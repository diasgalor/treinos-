import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
import plotly.express as px
from unidecode import unidecode
import xml.etree.ElementTree as ET
from pyproj import Transformer
from shapely.geometry import Polygon, LineString, Point
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import io
import gc

# Função para extrair metadados e geometria do KML
def extrair_dados_kml(kml_content):
    try:
        tree = ET.fromstring(kml_content)
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        dados = []
        for placemark in tree.findall('.//kml:Placemark', ns):
            props = {}
            name_elem = placemark.find('kml:name', ns)
            props['Name'] = name_elem.text if name_elem is not None else None
            for simple_data in placemark.findall('.//kml:SimpleData', ns):
                props[simple_data.get('name')] = simple_data.text

            geometry = None
            polygon_elem = placemark.find('.//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', ns)
            if polygon_elem is not None:
                coords_text = polygon_elem.text.strip()
                coords = [tuple(map(float, c.split(','))) for c in coords_text.split()]
                try:
                    geometry = Polygon([(c[0], c[1]) for c in coords])
                except Exception as geom_e:
                    st.warning(f"Erro ao criar geometria para placemark {props.get('Name', 'Sem Nome')}: {geom_e}")
                    geometry = None

            line_elem = placemark.find('.//kml:LineString/kml:coordinates', ns)
            if line_elem is not None:
                coords_text = line_elem.text.strip()
                coords = [tuple(map(float, c.split(','))) for c in coords_text.split()]
                try:
                    geometry = LineString([(c[0], c[1]) for c in coords])
                except Exception as geom_e:
                    st.warning(f"Erro ao criar geometria para placemark {props.get('Name', 'Sem Nome')}: {geom_e}")
                    geometry = None

            point_elem = placemark.find('.//kml:Point/kml:coordinates', ns)
            if point_elem is not None:
                coords_text = point_elem.text.strip()
                coords = tuple(map(float, coords_text.split(',')))
                try:
                    geometry = Point(coords[0], coords[1])
                except Exception as geom_e:
                    st.warning(f"Erro ao criar geometria para placemark {props.get('Name', 'Sem Nome')}: {geom_e}")
                    geometry = None

            if geometry:
                dados.append({**props, 'geometry': geometry})

        if not dados:
            st.warning("Nenhuma geometria válida encontrada no KML.")
            return gpd.GeoDataFrame(columns=['Name', 'geometry'])

        gdf = gpd.GeoDataFrame(dados, crs="EPSG:4326")
        return gdf

    except Exception as e:
        st.error(f"Erro ao processar KML: {e}")
        return gpd.GeoDataFrame(columns=['Name', 'geometry'])

# Função para padronizar nomes
def formatar_nome(nome):
    return unidecode(nome.upper()) if isinstance(nome, str) else nome

# Função para normalizar coordenadas
def normalizar_coordenadas(valor, scale_factor=1000000000):
    if isinstance(valor, str):
        try:
            valor_float = float(valor.replace(',', '')) / scale_factor
            valor_normalizado = round(valor_float, 6)
            return valor_normalizado
        except ValueError:
            st.warning(f"Não foi possível converter o valor: {valor}")
            return None
    return None

# Função para criar mapa interativo
def criar_mapa_interativo(df_csv, gdf_kml):
    required_columns = ['VL_LATITUDE', 'VL_LONGITUDE', 'DESC_TIPO_EQUIPAMENTO', 'FROTA', 'STATUS']
    missing_columns = [col for col in required_columns if col not in df_csv.columns]
    if missing_columns:
        st.error(f"Colunas ausentes em df_csv: {missing_columns}")
        return None

    df_csv = df_csv.dropna(subset=["VL_LATITUDE", "VL_LONGITUDE"])
    df_csv = df_csv[
        (df_csv["VL_LATITUDE"].between(-90, 90)) &
        (df_csv["VL_LONGITUDE"].between(-180, 180))
    ]

    if df_csv.empty:
        st.error("Nenhuma coordenada válida após filtragem.")
        return None

    mapa = folium.Map(
        location=[df_csv["VL_LATITUDE"].mean(), df_csv["VL_LONGITUDE"].mean()],
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True
    )

    marker_cluster = MarkerCluster().add_to(mapa)

    df_estacoes = df_csv[df_csv["DESC_TIPO_EQUIPAMENTO"].str.contains("ESTACAO", case=False, na=False)]
    for _, row in df_estacoes.iterrows():
        try:
            folium.Marker(
                location=[row["VL_LATITUDE"], row["VL_LONGITUDE"]],
                popup=f"{row['FROTA']}",
                icon=folium.Icon(color="blue", icon="cloud", prefix="fa")
            ).add_to(marker_cluster)
        except Exception as e:
            st.warning(f"Erro ao adicionar marcador para estação: {row['FROTA']}, {e}")

    df_pluviometros_ativos = df_csv[
        (df_csv["DESC_TIPO_EQUIPAMENTO"].str.contains("PLUVIOMETRO", case=False, na=False)) &
        (df_csv["STATUS"].str.upper() == "ATIVO")
    ]
    for _, row in df_pluviometros_ativos.iterrows():
        try:
            folium.Marker(
                location=[row["VL_LATITUDE"], row["VL_LONGITUDE"]],
                popup=f"{row['FROTA']}",
                icon=folium.Icon(color="green", icon="tint", prefix="fa")
            ).add_to(marker_cluster)
        except Exception as e:
            st.warning(f"Erro ao adicionar marcador para pluviômetro: {row['FROTA']}, {e}")

    if gdf_kml is not None and not gdf_kml.empty:
        if "Name" in gdf_kml.columns:
            folium.GeoJson(
                gdf_kml,
                name="Limites",
                tooltip=folium.GeoJsonTooltip(fields=["Name"], aliases=["Fazenda:"]),
                style_function=lambda x: {"fillColor": "blue", "color": "blue", "weight": 1, "fillOpacity": 0.1}
            ).add_to(mapa)
        else:
            folium.GeoJson(
                gdf_kml,
                name="Limites",
                style_function=lambda x: {"fillColor": "blue", "color": "blue", "weight": 1, "fillOpacity": 0.1}
            ).add_to(mapa)

    folium.LayerControl().add_to(mapa)
    return mapa

# Função para criar gráfico de firmwares
def criar_grafico_firmware(df_csv):
    df_firmware_fazenda = df_csv.groupby(['VL_FIRMWARE_EQUIPAMENTO', 'UNIDADE']).size().reset_index(name='Quantidade')
    df_firmware_fazenda = df_firmware_fazenda.sort_values(by='Quantidade', ascending=True)

    fig = px.bar(
        df_firmware_fazenda,
        x='Quantidade',
        y='UNIDADE',
        color='VL_FIRMWARE_EQUIPAMENTO',
        title='Distribuição de Firmwares por Unidade',
        labels={
            'VL_FIRMWARE_EQUIPAMENTO': 'Versão do Firmware',
            'UNIDADE': 'Unidade',
            'Quantidade': 'Qtd. de Equipamentos'
        },
        orientation='h',
        text='Quantidade',
        color_discrete_sequence=px.colors.qualitative.Dark2
    )

    fig.update_layout(
        title_font_size=16,
        title_font_family='Arial',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(250,250,250,1)',
        bargap=0.25,
        height=400,
        xaxis=dict(title='Quantidade de Equipamentos'),
        yaxis=dict(title=''),
        legend_title='Versão do Firmware',
        font=dict(size=12),
        margin=dict(l=10, r=10, t=50, b=10)
    )

    fig.update_traces(
        textposition='outside',
        textfont=dict(size=10, color='black')
    )

    return fig

# Função para criar gráficos de pluviômetros e comunicação
def criar_graficos_pluviometros(df_csv):
    cores_personalizadas = ["#2E86C1", "#28B463", "#8E44AD"]

    df_contagem_1 = df_csv[
        df_csv["DESC_TIPO_EQUIPAMENTO"].str.contains("ESTACAO|PLUVIOMETRO", case=False, na=False)
    ].groupby(['UNIDADE', 'DESC_TIPO_EQUIPAMENTO']).size().reset_index(name='Quantidade')

    df_contagem_2 = df_csv[df_csv['TIPO_COMUNICACAO'] != '4G'].groupby(['UNIDADE', 'TIPO_COMUNICACAO']).size().reset_index(name='Quantidade')

    contagem_moveis = df_csv['D_MOVEIS_AT'].value_counts()

    fig1 = px.bar(
        df_contagem_1,
        x='UNIDADE',
        y='Quantidade',
        color='DESC_TIPO_EQUIPAMENTO',
        title='Pluviômetros e Estações por Unidade',
        text='Quantidade',
        barmode='stack',
        color_discrete_sequence=cores_personalizadas
    )

    fig1.update_layout(
        height=300,
        legend_title='Tipo de Equipamento',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        margin=dict(l=10, r=10, t=50, b=10)
    )

    fig2 = px.bar(
        df_contagem_2,
        x='UNIDADE',
        y='Quantidade',
        color='TIPO_COMUNICACAO',
        title='Tipos de Comunicação por Unidade (Excluindo 4G)',
        text='Quantidade',
        barmode='stack',
        color_discrete_sequence=cores_personalizadas
    )

    fig2.update_layout(
        height=300,
        legend_title='Tipo de Comunicação',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        margin=dict(l=10, r=10, t=50, b=10)
    )

    fig3 = px.pie(
        values=contagem_moveis.values,
        names=contagem_moveis.index,
        title='Percentual de Equipamentos com Dados Móveis',
        hole=0.5,
        color_discrete_sequence=cores_personalizadas
    )

    fig3.update_traces(textinfo='percent+label', textfont_size=12)
    fig3.update_layout(showlegend=True, legend_title='Dados Móveis', height=300, margin=dict(l=10, r=10, t=50, b=10))

    return fig1, fig2, fig3

# Função para interpolação IDW
def interpolacao_idw(df, x_col='VL_LONGITUDE', y_col='VL_LATITUDE', val_col='DBM', resolution=0.002, buffer=0.05, geom_mask=None):
    df = df.dropna(subset=[val_col]).copy()
    if df.empty:
        return None, None, None, None

    if df[val_col].dropna().between(1, 4).all():
        df['class_num'] = df[val_col].astype(int)
    else:
        df['class_num'] = df[val_col].apply(lambda x: 4 if x > -70 else 3 if x > -85 else 2 if x > -100 else 1)

    minx, miny = df[x_col].min() - buffer, df[y_col].min() - buffer
    maxx, maxy = df[x_col].max() + buffer, df[y_col].max() + buffer

    x_grid = np.arange(minx, maxx, resolution)
    y_grid = np.arange(miny, maxy, resolution)

    if len(x_grid) * len(y_grid) > 1_000_000:
        st.warning("Resolução muito alta para interpolação. Reduza o tamanho da grade.")
        return None, None, None, None

    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    pontos = df[[x_col, y_col]].values
    valores = df['class_num'].values

    distances = cdist(grid_points, pontos)
    epsilon = 1e-9
    weights = 1 / (distances**2 + epsilon)
    denom = weights.sum(axis=1)
    numer = (weights * valores).sum(axis=1)
    interpolated = numer / denom
    interpolated = np.clip(np.round(interpolated), 1, 4)

    if geom_mask is not None:
        pontos_geom = [Point(xy) for xy in grid_points]
        mask = np.array([geom_mask.contains********

# Main Streamlit app
st.set_page_config(page_title="Mapa de Equipamentos Climáticos", layout="wide")

st.title("Mapa de Equipamentos Climáticos")
st.markdown("### Visualize estações e pluviômetros em um mapa interativo otimizado para dispositivos móveis.")

# File upload
col1, col2 = st.columns([1, 1])
with col1:
    excel_file = st.file_uploader("Carregar arquivo Excel (.xlsx)", type="xlsx")
with col2:
    kml_file = st.file_uploader("Carregar arquivo KML (.kml)", type="kml")

if excel_file:
    with st.spinner("Carregando arquivo Excel..."):
        df_csv = pd.read_excel(excel_file, dtype={'VL_LATITUDE': str, 'VL_LONGITUDE': str})
        df_csv.columns = df_csv.columns.str.strip()
        st.success(f"Arquivo Excel carregado! {df_csv.shape[0]} linhas, colunas: {list(df_csv.columns)}")

    expected_cols = ["VL_LATITUDE", "VL_LONGITUDE", "UNIDADE"]
    if not all(col in df_csv.columns for col in expected_cols):
        st.error(f"Colunas esperadas ausentes: {expected_cols}")
    else:
        df_csv["VL_LATITUDE"] = df_csv["VL_LATITUDE"].apply(normalizar_coordenadas)
        df_csv["VL_LONGITUDE"] = df_csv["VL_LONGITUDE"].apply(normalizar_coordenadas)
        df_csv["UNIDADE"] = df_csv["UNIDADE"].apply(formatar_nome)
        st.info(f"Linhas com coordenadas válidas: {df_csv.dropna(subset=['VL_LATITUDE', 'VL_LONGITUDE']).shape[0]}")

    gdf_kml = None
    if kml_file:
        with st.spinner("Carregando arquivo KML..."):
            kml_content = kml_file.read().decode('utf-8')
            gdf_kml = extrair_dados_kml(kml_content)
            if not gdf_kml.empty:
                st.success(f"Arquivo KML carregado! {len(gdf_kml)} feições")
            else:
                st.warning("Nenhum dado válido no KML.")

    # Mapa Interativo
    st.subheader("Mapa Interativo")
    mapa = criar_mapa_interativo(df_csv, gdf_kml)
    if mapa:
        st_folium(mapa, width="100%", height=400)

    # Gráficos
    st.subheader("Análise de Dados")
    tab1, tab2 = st.tabs(["Firmwares", "Pluviômetros e Comunicação"])
    
    with tab1:
        fig_firmware = criar_grafico_firmware(df_csv)
        st.plotly_chart(fig_firmware, use_container_width=True)

    with tab2:
        fig1, fig2, fig3 = criar_graficos_pluviometros(df_csv)
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)

    # Interpolação IDW
    st.subheader("Interpolação de Sinal por Fazenda")
    gdf_equipamentos = gpd.GeoDataFrame(
        df_csv,
        geometry=gpd.points_from_xy(df_csv['VL_LONGITUDE'], df_csv['VL_LATITUDE']),
        crs="EPSG:4326"
    )
    gdf_equipamentos['UNIDADE_Padronizada'] = gdf_equipamentos['UNIDADE'].apply(formatar_nome)

    if gdf_kml is not None:
        gdf_kml_com_nomes = gdf_kml.copy()
        gdf_kml_com_nomes['NomeFazendaExtraido'] = gdf_kml_com_nomes.get('NOME_FAZ', gdf_kml_com_nomes.get('Name', 'sem_nome'))
        gdf_kml_com_nomes['NomeFazendaKML_Padronizada'] = gdf_kml_com_nomes['NomeFazendaExtraido'].apply(formatar_nome)
        gdf_kml_com_nomes['geometry'] = gdf_kml_com_nomes['geometry'].apply(
            lambda geom: geom.buffer(0) if geom and not geom.is_valid else geom
        )

        unidades = sorted(gdf_equipamentos['UNIDADE_Padronizada'].dropna().unique())
        if unidades:
            unidade_selecionada = st.selectbox("Selecione a Fazenda:", unidades)
            if st.button("Gerar Interpolação"):
                with st.spinner("Gerando interpolação..."):
                    nome_formatado = formatar_nome(unidade_selecionada)
                    df_fazenda = gdf_equipamentos[gdf_equipamentos['UNIDADE_Padronizada'] == nome_formatado].copy()
                    if not df_fazenda.empty:
                        geom_df = gdf_kml_com_nomes[gdf_kml_com_nomes['NomeFazendaKML_Padronizada'] == nome_formatado]
                        if not geom_df.empty:
                            fazenda_geom = geom_df.unary_union
                            df_fazenda['DBM'] = pd.to_numeric(df_fazenda['DBM'], errors='coerce')
                            grid_x, grid_y, grid_numerico, bounds = interpolacao_idw(
                                df_fazenda, geom_mask=fazenda_geom
                            )
                            if grid_x is not None:
                                fig, ax = plt.subplots(figsize=(10, 8))
                                colors = {1: '#fc8d59', 2: '#fee08b', 3: '#91cf60', 4: '#1a9850'}
                                cmap = plt.matplotlib.colors.ListedColormap([colors[i] for i in range(1, 5)])
                                im = ax.imshow(grid_numerico, extent=bounds, origin='lower', cmap=cmap, alpha=0.8)
                                if fazenda_geom and not fazenda_geom.is_empty:
                                    gpd.GeoDataFrame(geometry=[fazenda_geom]).boundary.plot(ax=ax, color='black', linewidth=2)
                                for _, row in df_fazenda.iterrows():
                                    cor, marcador = ("blue", "o") if "estacao" in row["DESC_TIPO_EQUIPAMENTO"].lower() else ("green", "o")
                                    ax.scatter(row["VL_LONGITUDE"], row["VL_LATITUDE"], c=cor, marker=marcador, s=50, edgecolor="k")
                                cbar = plt.colorbar(im, ax=ax, ticks=[1.5, 2.5, 3.5, 4.5])
                                cbar.ax.set_yticklabels(['Ruim', 'Regular', 'Bom', 'Ótimo'])
                                st.pyplot(fig)
                            else:
                                st.error("Falha na interpolação. Verifique os dados.")
                        else:
                            st.error("Nenhuma geometria encontrada para a fazenda selecionada.")
                    else:
                        st.error("Nenhum dado encontrado para a fazenda selecionada.")
        else:
            st.error("Nenhuma unidade válida encontrada.")

else:
    st.info("Por favor, carregue um arquivo Excel para continuar.")
