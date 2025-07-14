import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
from shapely.geometry import Polygon, Point
import xml.etree.ElementTree as ET
from unidecode import unidecode

# --- ESTILO MINIMALISTA APPLE ---
st.set_page_config(layout="wide", page_title="Monitoramento Climático SLC", initial_sidebar_state="collapsed")
st.markdown("""
<style>
body { font-family: 'SF Pro Display', 'Segoe UI', Arial, sans-serif; background: #f8f8fa; color: #222; }
.block-container { padding: 0.5rem; max-width: 100%; }
h1, h2, h3 { font-weight: 700; color: #222; }
.stButton>button {
    background: #007aff;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 14px 0;
    font-size: 18px;
    width: 100%;
    margin-top: 8px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.04);
}
.stButton>button:hover { background: #005bb5; }
.stSelectbox, .stFileUploader { margin-bottom: 0.75rem; }
[data-testid="stHeader"] { background: none; }
@media (max-width: 600px) {
    h1 { font-size: 1.3rem; }
    h2 { font-size: 1.1rem; }
    .block-container { padding: 0.2rem; }
    .stButton>button { font-size: 16px; padding: 12px 0; }
    label, .stSelectbox, .stFileUploader { font-size: 15px; }
}
</style>
""", unsafe_allow_html=True)

st.title("🌱 Monitoramento Climático SLC")

# --- FUNÇÕES ---
def formatar_nome(nome):
    return unidecode(str(nome)).strip().upper() if isinstance(nome, str) else nome

def extrair_kml(file_content):
    try:
        tree = ET.fromstring(file_content)
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
        gdf = gpd.GeoDataFrame(dados, crs="EPSG:4326")
        if gdf.empty:
            st.warning("Nenhum dado válido encontrado no KML.")
        return gdf
    except Exception as e:
        st.error(f"Erro ao processar KML: {e}")
        return gpd.GeoDataFrame(columns=['Name', 'geometry'])

# --- UPLOAD DE ARQUIVOS ---
st.header("Upload de Arquivos")
excel_file = st.file_uploader("Carregar consolidada.xlsx", type=["xlsx"])
kml_file = st.file_uploader("Carregar slc_mapa.kml", type=["kml"], accept_multiple_files=False)

# --- CARREGAR DADOS (CACHÊ) ---
@st.cache_data
def carregar_dados(excel_file, kml_file):
    df = pd.read_excel(excel_file) if excel_file is not None else None
    gdf = extrair_kml(kml_file.read().decode("utf-8")) if kml_file is not None else None
    return df, gdf

df, gdf = None, None
if excel_file and kml_file:
    df, gdf = carregar_dados(excel_file, kml_file)

if df is not None:
    st.subheader("Dados Climáticos")
    st.dataframe(df, use_container_width=True, hide_index=True)

# --- MAPA RESPONSIVO ---
if gdf is not None and not gdf.empty:
    st.subheader("Mapa das Áreas")
    map_center = gdf.geometry.centroid.iloc[0].xy if not gdf.empty else (-49, -15)
    m = folium.Map(
        location=[map_center[1][0], map_center[0][0]],
        zoom_start=10,
        width='100%', height=350 if st.experimental_get_query_params().get("mobile") else 500
    )
    for _, row in gdf.iterrows():
        folium.GeoJson(row['geometry'], name=row['Name']).add_to(m)
        folium.Marker([row['geometry'].centroid.y, row['geometry'].centroid.x], 
                      popup=row['Name'],
                      icon=folium.Icon(color='green', icon='leaf', prefix='fa')).add_to(m)
    st_folium(m, width=350 if st.experimental_get_query_params().get("mobile") else 700, height=350 if st.experimental_get_query_params().get("mobile") else 500)

# --- GRÁFICO AJUSTÁVEL ---
if df is not None and 'Temperatura' in df.columns and 'Data' in df.columns:
    st.subheader("Gráfico de Temperatura")
    fig = px.line(df, x='Data', y='Temperatura', markers=True, title='Variação da Temperatura')
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(size=16),
        plot_bgcolor='#f8f8fa',
        paper_bgcolor='#f8f8fa',
        height=350 if st.experimental_get_query_params().get("mobile") else 500
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div style='text-align:center; color:#aaa; font-size:14px; margin-top:20px'>
Feito para SLC | <span style="font-weight:600;">Design inspirado no iOS</span>
</div>
""", unsafe_allow_html=True)
