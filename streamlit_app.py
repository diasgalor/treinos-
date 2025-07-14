import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Polygon, Point
import xml.etree.ElementTree as ET
from unidecode import unidecode

# --- ESTILO MOBILE APPLE-INSPIRED ---
st.set_page_config(
    page_title="🌱 Monitoramento Climático SLC",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #f5f5f7; color: #222;}
.block-container { padding: 0.5rem; max-width: 100%; margin: auto; }
h1, h2, h3 { font-weight: 700; color: #222; }
.stButton>button {
    background: linear-gradient(90deg, #007aff 70%, #34c759 100%);
    color: white;
    border-radius: 18px;
    border: none;
    padding: 14px 0;
    font-size: 18px;
    width: 100%;
    margin-top: 8px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.08);
    transition: background 0.2s;
}
.stButton>button:hover { background: #005bb5; }
.stSelectbox, .stFileUploader, .stTextInput, .stNumberInput {
    margin-bottom: 1rem;
    border-radius: 14px !important;
    border: 1px solid #e5e5ea !important;
    box-shadow: 0 1px 8px rgba(0,0,0,0.04);
}
.stDataFrame, .stTable {
    border-radius: 14px !important;
    border: 1.5px solid #e5e5ea !important;
    box-shadow: 0 1px 8px rgba(0,0,0,0.04);
}
@media (max-width: 600px) {
    .block-container { padding: 0.2rem; }
    h1 { font-size: 1.2rem; }
    h2 { font-size: 1rem; }
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

@st.cache_data
def carregar_dados(excel_file, kml_file):
    df = pd.read_excel(excel_file) if excel_file is not None else None
    gdf = extrair_kml(kml_file.read().decode("utf-8")) if kml_file is not None else None
    return df, gdf

df, gdf = None, None
if excel_file and kml_file:
    df, gdf = carregar_dados(excel_file, kml_file)

# --- DADOS EXCEL ---
if df is not None:
    st.subheader("📋 Dados Climáticos")
    st.dataframe(df, use_container_width=True, hide_index=True)

# --- MAPA RESPONSIVO E IMPACTANTE ---
if gdf is not None and not gdf.empty:
    st.subheader("🗺️ Mapa das Áreas")
    centroid = gdf.geometry.centroid.iloc[0].xy if not gdf.empty else (-49, -15)
    m = folium.Map(
        location=[centroid[1][0], centroid[0][0]],
        zoom_start=11,
        width='100%',
        height=430
    )
    for _, row in gdf.iterrows():
        folium.GeoJson(
            row['geometry'],
            name=row['Name'],
            style_function=lambda x: {
                "fillColor": "#007aff40", "color": "#007aff", "weight": 2, "fillOpacity": 0.22
            }
        ).add_to(m)
        folium.Marker(
            location=[row['geometry'].centroid.y, row['geometry'].centroid.x],
            popup=row['Name'],
            icon=folium.Icon(color='green', icon='leaf', prefix='fa')
        ).add_to(m)
    st_folium(m, width=370, height=430)

# --- GRÁFICO DE TEMPERATURA ---
if df is not None and 'Temperatura' in df.columns and 'Data' in df.columns:
    st.subheader("🌡️ Variação de Temperatura")
    fig = px.line(
        df, x='Data', y='Temperatura', markers=True,
        title='Variação da Temperatura',
        template='plotly_white',
        color_discrete_sequence=['#007aff'],
    )
    fig.update_layout(
        title_font_size=17,
        font=dict(family='-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Arial', size=15),
        plot_bgcolor='#f5f5f7',
        paper_bgcolor='#f5f5f7',
        margin=dict(l=24, r=24, t=40, b=24),
        height=360,
        xaxis=dict(title='Data', showgrid=False, showline=True, linecolor='#e5e5ea'),
        yaxis=dict(title='Temperatura', showgrid=False, showline=True, linecolor='#e5e5ea'),
    )
    fig.update_traces(line=dict(width=4), marker=dict(size=10, color='#34c759'))
    st.plotly_chart(fig, use_container_width=True)

# --- GRÁFICO DE BARRAS IMPACTANTE ---
if df is not None and 'VL_FIRMWARE_EQUIPAMENTO' in df.columns and 'UNIDADE' in df.columns:
    st.subheader("🔢 Distribuição de Firmwares por Unidade")
    df_firmware = df.groupby(['VL_FIRMWARE_EQUIPAMENTO', 'UNIDADE']).size().reset_index(name='Quantidade')
    fig_b = px.bar(
        df_firmware,
        x='Quantidade',
        y='UNIDADE',
        color='VL_FIRMWARE_EQUIPAMENTO',
        orientation='h',
        text='Quantidade',
        color_discrete_sequence=['#007aff', '#34c759', '#1d1d1f', '#e5e5ea'],
        title='Distribuição de Firmwares por Unidade'
    )
    fig_b.update_layout(
        title_font_size=16,
        font=dict(family='-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Arial', size=15),
        plot_bgcolor='#f5f5f7',
        paper_bgcolor='#f5f5f7',
        bargap=0.28,
        height=360,
        margin=dict(l=24, r=24, t=50, b=24),
        xaxis=dict(title='Quantidade', showgrid=False, showline=True, linecolor='#e5e5ea'),
        yaxis=dict(title='', automargin=True, showgrid=False, showline=True, linecolor='#e5e5ea'),
        legend_title='Firmware'
    )
    fig_b.update_traces(
        textposition='outside',
        marker=dict(line=dict(color='#e5e5ea', width=2)),
        textfont=dict(size=13, color='#222')
    )
    st.plotly_chart(fig_b, use_container_width=True)

# --- GRÁFICO DE ROSCA/DADOS MÓVEIS ---
if df is not None and 'D_MOVEIS_AT' in df.columns:
    st.subheader("📡 Percentual de Equipamentos com Dados Móveis")
    contagem_moveis = df['D_MOVEIS_AT'].value_counts()
    fig3 = px.pie(
        values=contagem_moveis.values,
        names=contagem_moveis.index,
        title='Dados Móveis',
        hole=0.5,
        color_discrete_sequence=['#007aff', '#34c759', '#e5e5ea']
    )
    fig3.update_traces(
        textinfo='percent+label',
        textfont_size=15,
        marker=dict(line=dict(color='#e5e5ea', width=2))
    )
    fig3.update_layout(
        showlegend=True,
        legend_title='Dados Móveis',
        font=dict(family='-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Arial', size=14),
        plot_bgcolor='#f5f5f7',
        paper_bgcolor='#f5f5f7',
        margin=dict(l=20, r=20, t=50, b=20),
        height=320
    )
    st.plotly_chart(fig3, use_container_width=True)

# --- RODAPÉ ---
st.markdown("""
<div style='text-align:center; color:#aaa; font-size:15px; margin-top:22px'>
Feito para SLC | <span style="font-weight:600;">Design mobile inspirado em iOS</span>
</div>
""", unsafe_allow_html=True)
