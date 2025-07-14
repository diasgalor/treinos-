import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from folium.plugins import MarkerCluster
from shapely.geometry import Point
from scipy.spatial.distance import cdist
from unidecode import unidecode
import xml.etree.ElementTree as ET
import gc
import re
import tempfile
import streamlit.components.v1 as components

# Configuração para layout responsivo
st.set_page_config(page_title="Mapa de Equipamentos Climáticos", layout="wide", initial_sidebar_state="collapsed")

# Estilo CSS minimalista com Tailwind via CDN
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        body { font-family: 'Arial', sans-serif; }
        .stApp { max-width: 100%; padding: 1rem; }
        .sidebar .sidebar-content { padding: 1rem; }
        h1, h2, h3 { font-weight: 500; color: #1f2937; }
        .plotly-chart { width: 100% !important; height: auto !important; }
        .folium-map { width: 100% !important; height: 400px; }
        @media (max-width: 640px) {
            .stApp { padding: 0.5rem; }
            h1 { font-size: 1.5rem; }
            h2 { font-size: 1.25rem; }
            .folium-map { height: 300px; }
        }
    </style>
""", unsafe_allow_html=True)

# Funções auxiliares
def formatar_nome(nome):
    if nome is None:
        return "N/A"
    return unidecode(str(nome)).strip().lower().replace(" ", "_")

def corrigir_coordenadas(valor):
    if isinstance(valor, str):
        valor_limpo = re.sub(r"[^0-9\-.]", "", valor).replace(",", ".")
        partes = valor_limpo.split(".")
        if len(partes) > 2:
            valor_limpo = partes[0] + "." + "".join(partes[1:])
        try:
            return float(valor_limpo)
        except ValueError:
            return None
    return valor

def extrair_metadados_kml(content):
    tree = ET.fromstring(content)
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    dados = []
    for placemark in tree.findall('.//kml:Placemark', ns):
        item = {}
        name = placemark.find('kml:name', ns)
        item['Name'] = name.text if name is not None else None
        for sd in placemark.findall('.//kml:SimpleData', ns):
            item[sd.attrib['name']] = sd.text
        dados.append(item)
    return pd.DataFrame(dados)

def classificar_dbm(valor):
    if pd.isna(valor):
        return np.nan
    elif valor > -70:
        return 4  # ótimo
    elif valor > -85:
        return 3  # bom
    elif valor > -100:
        return 2  # regular
    else:
        return 1  # ruim

def interpolacao_idw(df, x_col='VL_LONGITUDE', y_col='VL_LATITUDE', val_col='DBM', resolution=0.002, buffer=0.05, geom_mask=None):
    df = df.dropna(subset=[x_col, y_col, val_col]).copy()
    if df.empty or len(df) < 3:  # Mínimo de 3 pontos para interpolação
        return None, None, None, None

    if df[val_col].dropna().between(1, 4).all():
        df['class_num'] = df[val_col].astype(int)
    else:
        df['class_num'] = df[val_col].apply(classificar_dbm)

    if df['class_num'].isna().all():
        return None, None, None, None

    minx, miny = df[x_col].min() - buffer, df[y_col].min() - buffer
    maxx, maxy = df[x_col].max() + buffer, df[y_col].max() + buffer
    if not all(np.isfinite([minx, miny, maxx, maxy])):
        return None, None, None, None

    x_grid = np.arange(minx, maxx, resolution)
    y_grid = np.arange(miny, maxy, resolution)

    if len(x_grid) * len(y_grid) > 1_000_000:
        st.warning("Grade muito grande. Ajuste a resolução ou buffer.")
        return None, None, None, None

    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    pontos = df[[x_col, y_col]].values
    valores = df['class_num'].values

    distances = cdist(grid_points, pontos)
    epsilon = 1e-6  # Aumentado para evitar divisão por zero
    weights = 1 / (distances ** 2 + epsilon)
    denom = weights.sum(axis=1)
    if not np.all(denom > 0):
        return None, None, None, None
    numer = (weights * valores).sum(axis=1)
    interpolated = np.clip(np.round(numer / denom), 1, 4)

    if geom_mask is not None:
        pontos_geom = [Point(xy) for xy in grid_points]
        mask = np.array([geom_mask.contains(pt) for pt in pontos_geom])
        interpolated[~mask] = np.nan

    grid_numerico = interpolated.reshape(grid_x.shape)
    del distances, weights, interpolated
    gc.collect()
    return grid_x, grid_y, grid_numerico, (minx, maxx, miny, maxy)

# Upload de arquivos
st.sidebar.header("Upload de Arquivos")
csv_file = st.sidebar.file_uploader("CSV dos Equipamentos", type="csv")
kml_file = st.sidebar.file_uploader("KML com Limites das Fazendas", type="kml")

if csv_file and kml_file:
    # Leitura e pré-processamento
    df_csv = pd.read_csv(csv_file, delimiter=';')
    df_csv.columns = df_csv.columns.str.strip()
    df_csv['VL_LATITUDE'] = df_csv['VL_LATITUDE'].apply(corrigir_coordenadas)
    df_csv['VL_LONGITUDE'] = df_csv['VL_LONGITUDE'].apply(corrigir_coordenadas)
    df_csv['UNIDADE'] = df_csv['UNIDADE'].apply(formatar_nome)
    df_csv = df_csv.dropna(subset=['VL_LATITUDE', 'VL_LONGITUDE'])
    df_csv = df_csv[(df_csv['VL_LATITUDE'].between(-90, 90)) & (df_csv['VL_LONGITUDE'].between(-180, 180))]

    gdf_kml = gpd.read_file(kml_file, driver='KML')
    if gdf_kml.crs is None or gdf_kml.crs.to_epsg() != 4326:
        gdf_kml = gdf_kml.to_crs(epsg=4326)
    content = kml_file.getvalue().decode('utf-8')
    df_metadados = extrair_metadados_kml(content)
    col_nome = 'NOME_FAZ' if 'NOME_FAZ' in df_metadados.columns else 'Name'
    gdf_kml['NomeFazendaKML'] = df_metadados[col_nome].apply(formatar_nome)

    # Interface
    st.markdown("<h1 class='text-2xl font-semibold mb-4'>Mapa de Equipamentos Climáticos</h1>", unsafe_allow_html=True)
    aba = st.radio("Visualização:", ['Dashboard', 'Mapa', 'Interpolação'], horizontal=True, label_visibility="collapsed")

    if aba == 'Dashboard':
        st.markdown("<h2 class='text-xl font-medium mb-3'>Gráficos Interativos</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1], gap="small")

        with col1:
            df_firmware = df_csv.groupby(['VL_FIRMWARE_EQUIPAMENTO', 'UNIDADE']).size().reset_index(name='Quantidade')
            fig1 = px.bar(df_firmware, x='Quantidade', y='UNIDADE', color='VL_FIRMWARE_EQUIPAMENTO', orientation='h',
                          title='Firmwares por Unidade', height=300, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig1.update_layout(margin=dict(l=20, r=20, t=50, b=20), legend_title="Firmware", font=dict(size=10))
            fig1.update_traces(textposition='outside', textfont=dict(size=8))
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            df_tipo = df_csv[df_csv["DESC_TIPO_EQUIPAMENTO"].str.contains("ESTACAO|PLUVIOMETRO", na=False, case=False)]
            df_tipo = df_tipo.groupby(['UNIDADE', 'DESC_TIPO_EQUIPAMENTO']).size().reset_index(name='Quantidade')
            fig2 = px.bar(df_tipo, x='UNIDADE', y='Quantidade', color='DESC_TIPO_EQUIPAMENTO', barmode='stack',
                          title='Equipamentos por Unidade', height=300, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig2.update_layout(margin=dict(l=20, r=20, t=50, b=20), legend_title="Tipo", font=dict(size=10))
            fig2.update_traces(textposition='outside', textfont=dict(size=8))
            st.plotly_chart(fig2, use_container_width=True)

        if 'D_MOVEIS_AT' in df_csv.columns:
            contagem = df_csv['D_MOVEIS_AT'].value_counts()
            fig3 = px.pie(values=contagem.values, names=contagem.index, hole=0.5, title='Equipamentos com Dados Móveis',
                          height=300, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig3.update_traces(textinfo='percent+label', textfont_size=10)
            fig3.update_layout(margin=dict(l=20, r=20, t=50, b=20), legend_title="Dados Móveis", font=dict(size=10))
            st.plotly_chart(fig3, use_container_width=True)

    elif aba == 'Mapa':
        st.markdown("<h2 class='text-xl font-medium mb-3'>Mapa Interativo</h2>", unsafe_allow_html=True)
        if not df_csv.empty and not df_csv[['VL_LATITUDE', 'VL_LONGITUDE']].isna().any().any():
            mapa = folium.Map(location=[df_csv['VL_LATITUDE'].mean(), df_csv['VL_LONGITUDE'].mean()], zoom_start=10, tiles="CartoDB Positron")
            cluster = MarkerCluster().add_to(mapa)

            for _, row in df_csv.iterrows():
                if pd.isna(row['VL_LATITUDE']) or pd.isna(row['VL_LONGITUDE']):
                    continue
                tipo = str(row['DESC_TIPO_EQUIPAMENTO']).lower()
                cor = 'green' if 'pluviometro' in tipo else 'blue' if 'estacao' in tipo else 'red'
                folium.Marker(
                    location=[row['VL_LATITUDE'], row['VL_LONGITUDE']],
                    popup=f"{row['FROTA']}",
                    icon=folium.Icon(color=cor, icon='circle', prefix='fa')
                ).add_to(cluster)

            folium.GeoJson(
                gdf_kml,
                name="Limites",
                tooltip=folium.GeoJsonTooltip(fields=["NomeFazendaKML"], aliases=["Fazenda:"]),
                style_function=lambda x: {"fillColor": "blue", "color": "blue", "weight": 1, "fillOpacity": 0.1}
            ).add_to(mapa)
            folium.LayerControl().add_to(mapa)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                mapa.save(tmp.name)
                with open(tmp.name, 'r') as f:
                    components.html(f.read(), height=400)
        else:
            st.warning("Nenhum dado de coordenadas válido para exibir o mapa.")

    elif aba == 'Interpolação':
        st.markdown("<h2 class='text-xl font-medium mb-3'>Interpolação do Sinal por Fazenda</h2>", unsafe_allow_html=True)
        fazendas = sorted(gdf_kml['NomeFazendaKML'].dropna().unique())
        escolha = st.selectbox("Selecione uma fazenda:", fazendas, key="fazenda_select")

        df_csv['UNIDADE_Padronizada'] = df_csv['UNIDADE'].apply(formatar_nome)
        gdf_kml['NomeFazendaKML_Padronizada'] = gdf_kml['NomeFazendaKML'].apply(formatar_nome)
        df_faz = df_csv[df_csv['UNIDADE_Padronizada'] == formatar_nome(escolha)].copy()
        geom = gdf_kml[gdf_kml['NomeFazendaKML_Padronizada'] == formatar_nome(escolha)].geometry.union_all()

        if not df_faz.empty and geom is not None:
            df_faz['DBM'] = pd.to_numeric(df_faz['DBM'], errors='coerce')
            has_dbm = 'DBM' in df_faz.columns and not df_faz['DBM'].dropna().empty
            has_intensidade = 'INTENSIDADE' in df_faz.columns and not df_faz['INTENSIDADE'].dropna().empty

            mapping = {"ruim": 1, "regular": 2, "bom": 3, "otimo": 4}
            if has_intensidade:
                df_faz['INTENSIDADE_MAP'] = df_faz['INTENSIDADE'].apply(
                    lambda x: mapping.get(unidecode(str(x)).strip().lower(), np.nan)
                )
                if has_dbm:
                    medias_dbm = df_faz.dropna(subset=['DBM']).groupby('INTENSIDADE_MAP')['DBM'].mean()
                    df_faz['DBM'] = df_faz.apply(
                        lambda row: medias_dbm.get(row['INTENSIDADE_MAP'], row['DBM']) if pd.isna(row['DBM']) else row['DBM'], axis=1
                    )

            val_col = 'DBM' if has_dbm else 'INTENSIDADE_MAP' if has_intensidade else None
            if val_col is None:
                st.warning(f"Nenhum dado válido de intensidade para a fazenda '{escolha}'.")
            else:
                grid_x, grid_y, grid_numerico, bounds = interpolacao_idw(df_faz, val_col=val_col, geom_mask=geom)
                if grid_x is not None:
                    colors = {1: '#fc8d59', 2: '#fee08b', 3: '#91cf60', 4: '#1a9850'}
                    cmap = plt.matplotlib.colors.ListedColormap([colors[i] for i in range(1, 5)])
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(grid_numerico, extent=bounds, origin='lower', cmap=cmap, interpolation='nearest', alpha=0.8)
                    gpd.GeoSeries(geom).boundary.plot(ax=ax, color='black', linewidth=1.5)
                    for _, row in df_faz.iterrows():
                        if pd.isna(row['VL_LONGITUDE']) or pd.isna(row['VL_LATITUDE']):
                            continue
                        tipo = str(row['DESC_TIPO_EQUIPAMENTO']).lower()
                        cor = 'blue' if 'pluviometro' in tipo else 'purple' if 'estacao' in tipo else 'red'
                        ax.scatter(row['VL_LONGITUDE'], row['VL_LATITUDE'], c=cor, s=50, edgecolor='k', linewidth=0.5, label=tipo if tipo not in ax.get_legend_handles_labels()[1] else "")
                    ax.set_xlabel("Longitude", fontsize=8)
                    ax.set_ylabel("Latitude", fontsize=8)
                    ax.set_title(f"Interpolação - {escolha}", fontsize=10)
                    cbar = plt.colorbar(im, ax=ax, ticks=[1.5, 2.5, 3.5, 4.5])
                    cbar.ax.set_yticklabels(['Ruim', 'Regular', 'Bom', 'Ótimo'], fontsize=8)
                    cbar.set_label('Qualidade do Sinal', fontsize=8)
                    ax.legend(title="Equipamento", fontsize=8, title_fontsize=8, loc='upper right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    gc.collect()
                else:
                    st.warning(f"Falha na interpolação para a fazenda '{escolha}'. Verifique se há dados válidos suficientes.")
        else:
            st.warning(f"Nenhum dado ou geometria válida para a fazenda '{escolha}'.")
else:
    st.info("Envie os arquivos CSV e KML na barra lateral para iniciar.")
