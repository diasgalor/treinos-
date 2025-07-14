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
    background-color: #fdfdfd;
    color: #222;
}
[data-testid="stHeader"] { background: none; }
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
    margin: auto;
}
h1, h2, h3 { font-weight: 600; color: #111; }
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
    r = requests.get(url)
    kml = r.content
    tree = ET.fromstring(kml)
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

# --- LEITURA DE DADOS AUTOMÁTICA DO ONEDRIVE ---
excel_url = "https://solinfteccombr0.sharepoint.com/:x:/s/ged/Ee54FYVqqh9Hh3zUEuGJJOsBcAAAsZne4aXjTe6sAyQJvA?download=1"
kml_url = "https://solinfteccombr0.sharepoint.com/:u:/s/ged/EQmWrLecAxZKk8NDqpykPaUBG9v7VE1LqL5e9AtW_zbMgg?download=1"

@st.cache_data
def carregar_dados():
    response = requests.get(excel_url)
    df = pd.read_excel(io.BytesIO(response.content), dtype=str, engine='openpyxl')
    df.columns = df.columns.str.strip()
    df['VL_LATITUDE'] = df['VL_LATITUDE'].apply(corrigir_coord)
    df['VL_LONGITUDE'] = df['VL_LONGITUDE'].apply(corrigir_coord)
    df['UNIDADE'] = df['UNIDADE'].apply(formatar_nome)
    gdf = extrair_kml(kml_url)
    return df.dropna(subset=['VL_LATITUDE', 'VL_LONGITUDE']), gdf

df_csv, gdf_kml = carregar_dados()

# --- MENU ---
opcao = st.sidebar.radio("Visualização:", ["Mapa", "Dashboard"])

if opcao == "Mapa":
    st.subheader("🗺️ Mapa Interativo")
    m = folium.Map(location=[df_csv['
