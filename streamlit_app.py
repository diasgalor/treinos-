import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from unidecode import unidecode

# --- Configuração da Página ---
st.set_page_config(page_title="🗺️ Talhões", layout="wide", initial_sidebar_state="collapsed")

# --- Estilo moderno e leve inspirado em apps móveis ---
st.markdown("""
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background-color: #f9fafb;
    color: #111;
}
h1, h2, h3 {
    font-weight: 600;
    color: #222;
}
.stButton>button {
    background-color: #2c7be5;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    font-size: 1rem;
    transition: 0.2s ease-in-out;
}
.stButton>button:hover {
    background-color: #1a5fd0;
}
.stFileUploader, .stMultiSelect {
    border-radius: 10px !important;
    border: 1px solid #ddd !important;
    background: white !important;
}
@media (max-width: 600px) {
    h1 { font-size: 1.2rem; }
    .stButton>button { font-size: 0.9rem; padding: 0.6rem 1rem; }
}
</style>
""", unsafe_allow_html=True)

st.title("🗺️ Visualizador de Talhões")

# --- Funções ---
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
        return gdf
    except Exception as e:
        st.error(f"Erro ao processar o KML: {e}")
        return gpd.GeoDataFrame(columns=['Name', 'geometry'])

# --- Upload do KML ---
kml_file = st.file_uploader("Carregue o arquivo KML dos talhões", type="kml")

if kml_file:
    gdf_kml = extrair_kml(kml_file.read().decode("utf-8"))
    if gdf_kml.empty:
        st.warning("Nenhum polígono encontrado.")
        st.stop()

    gdf_kml["Name"] = gdf_kml["Name"].apply(formatar_nome)
    talhoes = sorted(gdf_kml["Name"].dropna().unique())

    selecao = st.multiselect("Selecione os talhões para visualizar no mapa", talhoes, default=talhoes[:1])

    if selecao:
        talhoes_sel = gdf_kml[gdf_kml["Name"].isin(selecao)]
        centro = talhoes_sel.geometry.centroid.iloc[0]
        m = folium.Map(location=[centro.y, centro.x], zoom_start=14, tiles="CartoDB positron")

        for _, row in talhoes_sel.iterrows():
            nome = row["Name"]
            folium.GeoJson(
                row["geometry"],
                name=nome,
                tooltip=folium.Tooltip(nome),
                style_function=lambda x: {
                    "fillColor": "#38b000",
                    "color": "#22577a",
                    "weight": 2,
                    "fillOpacity": 0.3
                }
            ).add_to(m)

        st.markdown("### 🧭 Mapa dos Talhões Selecionados")
        st_folium(m, height=500, width="100%")
    else:
        st.info("Selecione pelo menos um talhão para exibir no mapa.")
else:
    st.warning("Carregue o arquivo `.kml` com os talhões para iniciar.")

# --- Rodapé ---
st.markdown("""
<hr style="margin-top:2rem;margin-bottom:1rem;">
<div style='text-align:center; font-size:0.9rem; color:#999'>
🌱 Tecnologia a favor do campo • Versão simplificada para visualização de talhões
</div>
""", unsafe_allow_html=True)
