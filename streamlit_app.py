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
    padding: 1rem;
    max-width: 100%;
    margin: auto;
}
h1, h2, h3 { font-weight: 600; color: #1d1d1f; }
.stButton>button {
    background-color: #007aff;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 8px 16px;
    font-size: 16px;
}
.stButton>button:hover {
    background-color: #005bb5;
}
.stSelectbox, .stFileUploader {
    margin-bottom: 1rem;
}
@media (max-width: 600px) {
    .block-container {
        padding: 0.5rem;
    }
    h1 { font-size: 1.5rem; }
    h2 { font-size: 1.2rem; }
    .stButton>button {
        width: 100%;
        padding: 10px;
    }
    [data-testid="stFileUploader"] {
        font-size: 14px;
    }
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

@st.cache_data
def carregar_dados(excel_file, kml_file):
    try:
        # Carregar Excel
        if excel_file is None:
            st.error("Por favor, carregue o arquivo consolidada.xlsx.")
            return None, None
        try
