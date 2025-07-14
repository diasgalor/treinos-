# streamlit_app.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
from shapely.geometry import Polygon, Point
import xml.etree.ElementTree as ET
import requests
import io
from unidecode import unidecode
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# --- ESTILO APP ---
st.set_page_config(layout="wide", page_title="Mapa SLC Climáticos")
st.markdown("""
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background-color: #f5f5f7;
    color: #1d1d1f;
}
[data-testid="stHeader"] { background: none; }
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
    margin: auto;
}
h1, h2, h3 { font-weight: 600; color: #1d1d1f; }
.stButton>button {
    background-color: #007aff;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
}
.stButton>button:hover {
    background-color: #005bb5;
}
</style>
""", unsafe_allow_html=True)

st.title("🌱 Monitoramento Climático SLC")

# --- FUNÇÕES ---
def formatar_nome(nome):
    return unidecode(str(nome)).strip().upper() if isinstance(nome, str) else nome

def corrigir_coord(valor):
    try:
        return round(float(str(valor).replace(',', '')) / 1e9, 6)
    except:
        return None

def extrair_kml(url):
    try:
        r = requests.get(url)
        tree = ET.fromstring(r.content)
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        dados = []
        for placemark in tree.findall('.//kml:Placemark', ns):
            props = {}
            name_elem = placemark.find('kml:name', ns)
            props['Name'] = name_elem.text if name_elem is not None else None
            coords_elem = placemark.find('.//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', ns)
            if coords_elem is not None:
                coords_text = coords_elem.text.strip()
                coords = [tuple(map(float, c.split(',')[:2])) for c in coords_text.split()]
                geometry = Polygon(coords)
                dados.append({**props, 'geometry': geometry})
        return gpd.GeoDataFrame(dados, crs="EPSG:4326")
    except Exception as e:
        st.error(f"Erro ao processar KML: {e}")
        return gpd.GeoDataFrame(columns=['Name', 'geometry'])

# --- LEITURA DE DADOS ---
excel_url = "https://solinfteccombr0.sharepoint.com/:x:/s/ged/Ee54FYVqqh9Hh3zUEuGJJOsBcAAAsZne4aXjTe6sAyQJvA?download=1"
kml_url = "https://solinfteccombr0.sharepoint.com/:u:/s/ged/EQmWrLecAxZKk8NDqpykPaUBG9v7VE1LqL5e9AtW_zbMgg?download=1"

@st.cache_data
def carregar_dados():
    try:
        response = requests.get(excel_url)
        df = pd.read_excel(io.BytesIO(response.content), dtype=str, engine='openpyxl')
        df.columns = df.columns.str.strip()
        df['VL_LATITUDE'] = df['VL_LATITUDE'].apply(corrigir_coord)
        df['VL_LONGITUDE'] = df['VL_LONGITUDE'].apply(corrigir_coord)
        df['UNIDADE'] = df['UNIDADE'].apply(formatar_nome)
        gdf = extrair_kml(kml_url)
        return df.dropna(subset=['VL_LATITUDE', 'VL_LONGITUDE']), gdf
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None

df_csv, gdf_kml = carregar_dados()

if df_csv is None or gdf_kml is None:
    st.stop()

# Validação de colunas
required_columns = ['VL_LATITUDE', 'VL_LONGITUDE', 'UNIDADE', 'DESC_TIPO_EQUIPAMENTO', 'FROTA', 'STATUS']
missing_columns = [col for col in required_columns if col not in df_csv.columns]
if missing_columns:
    st.error(f"Colunas ausentes no Excel: {missing_columns}")
    st.stop()

# --- MENU ---
opcao = st.sidebar.radio("Visualização:", ["Mapa", "Dashboard", "Interpolação Sinal"])

# --- MAPA INTERATIVO ---
if opcao == "Mapa":
    st.subheader("🗺️ Mapa Interativo")
    try:
        mapa = folium.Map(
            location=[df_csv["VL_LATITUDE"].mean(), df_csv["VL_LONGITUDE"].mean()],
            zoom_start=10,
            tiles="CartoDB Positron"
        )
        marker_cluster = MarkerCluster().add_to(mapa)

        # Estações meteorológicas
        df_estacoes = df_csv[df_csv["DESC_TIPO_EQUIPAMENTO"].str.contains("ESTACAO", case=False, na=False)]
        for _, row in df_estacoes.iterrows():
            folium.Marker(
                location=[row["VL_LATITUDE"], row["VL_LONGITUDE"]],
                popup=f"{row['FROTA']}",
                icon=folium.Icon(color="blue", icon="cloud", prefix="fa")
            ).add_to(marker_cluster)

        # Pluviômetros ativos
        df_pluviometros_ativos = df_csv[
            (df_csv["DESC_TIPO_EQUIPAMENTO"].str.contains("PLUVIOMETRO", case=False, na=False)) &
            (df_csv["STATUS"].str.upper() == "ATIVO")
        ]
        for _, row in df_pluviometros_ativos.iterrows():
            folium.Marker(
                location=[row["VL_LATITUDE"], row["VL_LONGITUDE"]],
                popup=f"{row['FROTA']}",
                icon=folium.Icon(color="green", icon="tint", prefix="fa")
            ).add_to(marker_cluster)

        # Limites KML
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
        st_folium(mapa, width=1200, height=600)
    except Exception as e:
        st.error(f"Erro ao criar mapa: {e}")

# --- DASHBOARD ---
elif opcao == "Dashboard":
    st.subheader("📊 Dashboard")
    
    # Gráfico 1: Firmwares por Unidade
    df_firmware_fazenda = df_csv.groupby(['VL_FIRMWARE_EQUIPAMENTO', 'UNIDADE']).size().reset_index(name='Quantidade')
    df_firmware_fazenda = df_firmware_fazenda.sort_values(by='Quantidade', ascending=True)
    fig1 = px.bar(
        df_firmware_fazenda,
        x='Quantidade',
        y='UNIDADE',
        color='VL_FIRMWARE_EQUIPAMENTO',
        title='<b>Distribuição de Firmwares por Unidade</b>',
        labels={'VL_FIRMWARE_EQUIPAMENTO': 'Versão do Firmware', 'UNIDADE': 'Unidade', 'Quantidade': 'Qtd. de Equipamentos'},
        orientation='h',
        text='Quantidade',
        color_discrete_sequence=px.colors.qualitative.Dark2
    )
    fig1.update_layout(
        title_font_size=20,
        title_font_family='Arial',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(250,250,250,1)',
        bargap=0.25,
        height=600,
        xaxis=dict(title='Quantidade de Equipamentos'),
        yaxis=dict(title=''),
        legend_title='Versão do Firmware'
    )
    fig1.update_traces(textposition='outside', textfont=dict(size=12, color='black'))
    st.plotly_chart(fig1, use_container_width=True)

    # Gráfico 2: Pluviômetros e Estações por Unidade
    df_contagem_1 = df_csv[
        df_csv["DESC_TIPO_EQUIPAMENTO"].str.contains("ESTACAO|PLUVIOMETRO", case=False, na=False)
    ].groupby(['UNIDADE', 'DESC_TIPO_EQUIPAMENTO']).size().reset_index(name='Quantidade')
    fig2 = px.bar(
        df_contagem_1,
        x='UNIDADE',
        y='Quantidade',
        color='DESC_TIPO_EQUIPAMENTO',
        title='Pluviômetros e Estações por Unidade',
        text='Quantidade',
        barmode='stack',
        color_discrete_sequence=["#2E86C1", "#28B463"]
    )
    fig2.update_layout(height=450, legend_title='Tipo de Equipamento', plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig2, use_container_width=True)

# --- INTERPOLAÇÃO SINAL ---
elif opcao == "Interpolação Sinal":
    st.subheader("📡 Interpolação de Sinal")

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
        df = df.dropna(subset=[val_col]).copy()
        if df.empty:
            return None, None, None, None
        df['class_num'] = df[val_col].apply(classificar_dbm)
        minx, miny = df[x_col].min() - buffer, df[y_col].min() - buffer
        maxx, maxy = df[x_col].max() + buffer, df[y_col].max() + buffer
        x_grid = np.arange(minx, maxx, resolution)
        y_grid = np.arange(miny, maxy, resolution)
        if len(x_grid) * len(y_grid) > 1_000_000:
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
            mask = np.array([geom_mask.contains(pt) for pt in pontos_geom])
            interpolated[~mask] = np.nan
        grid_numerico = interpolated.reshape(grid_x.shape)
        return grid_x, grid_y, grid_numerico, (minx, maxx, miny, maxy)

    def plotar_interpolacao(grid_x, grid_y, grid_numerico, geom_fazenda, bounds, df_pontos):
        minx, maxx, miny, maxy = bounds
        colors = {1: '#fc8d59', 2: '#fee08b', 3: '#91cf60', 4: '#1a9850'}
        cmap = plt.matplotlib.colors.ListedColormap([colors[i] for i in range(1, 5)])
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(grid_numerico, extent=(minx, maxx, miny, maxy), origin='lower', cmap=cmap, interpolation='nearest', alpha=0.8)
        if geom_fazenda is not None and not geom_fazenda.is_empty:
            gpd.GeoDataFrame(geometry=[geom_fazenda]).boundary.plot(ax=ax, color='black', linewidth=2)
        for _, row in df_pontos.iterrows():
            ax.scatter(row["VL_LONGITUDE"], row["VL_LATITUDE"], c='purple' if 'ESTACAO' in row["DESC_TIPO_EQUIPAMENTO"] else 'blue', marker='o', s=100, edgecolor='k')
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Interpolação IDW da Intensidade do Sinal")
        cbar = plt.colorbar(im, ax=ax, ticks=[1.5, 2.5, 3.5, 4.5])
        cbar.ax.set_yticklabels(['Ruim', 'Regular', 'Bom', 'Ótimo'])
        cbar.set_label('Classe de Sinal')
        plt.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)
        plt.close(fig)

    unidades = sorted(df_csv['UNIDADE'].dropna().unique())
    fazenda = st.selectbox("Selecione a Fazenda:", unidades)
    if st.button("Gerar Interpolação"):
        nome_formatado = formatar_nome(fazenda)
        df_fazenda = df_csv[df_csv['UNIDADE'] == nome_formatado].copy()
        if df_fazenda.empty:
            st.warning("Nenhuma estação encontrada para a fazenda selecionada.")
        else:
            geom_df = gdf_kml[gdf_kml['Name'].apply(formatar_nome) == nome_formatado]
            fazenda_geom = geom_df.unary_union if not geom_df.empty else None
            df_fazenda['DBM'] = pd.to_numeric(df_fazenda.get('DBM', pd.Series()), errors='coerce')
            if 'DBM' in df_fazenda.columns and not df_fazenda['DBM'].dropna().empty:
                grid_x, grid_y, grid_numerico, bounds = interpolacao_idw(df_fazenda, geom_mask=fazenda_geom)
                if grid_x is not None:
                    plotar_interpolacao(grid_x, grid_y, grid_numerico, fazenda_geom, bounds, df_fazenda)
                else:
                    st.warning("Não foi possível gerar a interpolação devido a dados insuficientes.")
            else:
                st.warning("Dados de sinal (DBM) não disponíveis para a fazenda selecionada.")
