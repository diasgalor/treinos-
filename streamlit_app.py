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
    df = pd.read_excel(excel_url, dtype=str)
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
    m = folium.Map(location=[df_csv['VL_LATITUDE'].mean(), df_csv['VL_LONGITUDE'].mean()], zoom_start=9)
    cluster = MarkerCluster().add_to(m)

    for _, row in df_csv.iterrows():
        tipo = str(row.get("DESC_TIPO_EQUIPAMENTO", "")).lower()
        cor = 'green' if 'pluvi' in tipo else 'blue' if 'estacao' in tipo else 'red'
        folium.Marker(
            location=[row['VL_LATITUDE'], row['VL_LONGITUDE']],
            popup=row.get('FROTA', 'Sem frota'),
            icon=folium.Icon(color=cor)
        ).add_to(cluster)

    folium.GeoJson(
        gdf_kml,
        tooltip=folium.GeoJsonTooltip(fields=["Name"], aliases=["Fazenda"]),
        style_function=lambda x: {"color": "#444", "weight": 1, "fillOpacity": 0.05}
    ).add_to(m)

    st_folium(m, height=600, width=1100)

elif opcao == "Dashboard":
    st.subheader("📊 Painel de Equipamentos")

    col1, col2 = st.columns(2)

    if 'VL_FIRMWARE_EQUIPAMENTO' in df_csv.columns:
        firmware = df_csv.groupby(['VL_FIRMWARE_EQUIPAMENTO', 'UNIDADE']).size().reset_index(name='Qtd')
        fig1 = px.bar(firmware, x='Qtd', y='UNIDADE', color='VL_FIRMWARE_EQUIPAMENTO',
                      title='Firmwares por Unidade', orientation='h')
        col1.plotly_chart(fig1, use_container_width=True)

    if 'DESC_TIPO_EQUIPAMENTO' in df_csv.columns:
        tipos = df_csv[df_csv['DESC_TIPO_EQUIPAMENTO'].str.contains("PLUVI|ESTACAO", na=False, case=False)]
        tipos = tipos.groupby(['UNIDADE', 'DESC_TIPO_EQUIPAMENTO']).size().reset_index(name='Qtd')
        fig2 = px.bar(tipos, x='UNIDADE', y='Qtd', color='DESC_TIPO_EQUIPAMENTO',
                      title='Equipamentos por Tipo')
        col2.plotly_chart(fig2, use_container_width=True)

    if 'D_MOVEIS_AT' in df_csv.columns:
        dist = df_csv['D_MOVEIS_AT'].value_counts()
        fig3 = px.pie(values=dist.values, names=dist.index, hole=0.5,
                      title='Equipamentos com Dados Móveis')
        st.plotly_chart(fig3, use_container_width=True)
