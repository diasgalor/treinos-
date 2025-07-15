import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from unidecode import unidecode
import ee
import geemap.foliumap as geemap

# --- Configuração da Página ---
st.set_page_config(page_title="NDVI e Biomassa SLC", layout="wide", initial_sidebar_state="collapsed")

# --- Estilização Inspirada em iOS ---
st.markdown("""
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; background: #f5f5f7; color: #222;}
.stButton>button { background: linear-gradient(90deg, #007aff 70%, #34c759 100%); color:white; border-radius:18px; border:none; padding:14px 0; font-size:18px; width:100%; margin-top:8px; box-shadow:0 2px 16px rgba(0,0,0,0.08);}
.stButton>button:hover { background: #005bb5; }
.stSelectbox, .stFileUploader, .stMultiSelect { margin-bottom:1rem; border-radius:14px !important; border:1px solid #e5e5ea !important; box-shadow:0 1px 8px rgba(0,0,0,0.04);}
@media (max-width: 600px) {
    h1 { font-size:1.2rem; }
    h2 { font-size:1rem; }
    .stButton>button { font-size:16px; padding:12px 0;}
}
.stTabs > div[role='tablist'] > div { background: #e5e5ea; border-radius: 12px; padding: 4px; }
.stTabs > div[role='tablist'] > div > button { border-radius: 10px; font-size: 14px; padding: 8px; }
.stTabs > div[role='tablist'] > div > button[data-baseweb='tab-highlight'] { background: #007aff; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🛰️ NDVI e Biomassa por Talhão")

# --- Inicializar Earth Engine ---
try:
    ee.Initialize(project='ee-diasgalor')
except Exception as e:
    st.error("Erro ao conectar com o Google Earth Engine. Verifique a autenticação.")
    st.stop()

# --- Funções Utilitárias ---
def formatar_nome(nome):
    """Formata o nome removendo acentos e convertendo para maiúsculas."""
    return unidecode(str(nome)).strip().upper() if isinstance(nome, str) else nome

def extrair_kml(file_content):
    """Extrai geometrias e nomes de um arquivo KML."""
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
        st.error(f"Erro ao processar o arquivo KML: {e}")
        return gpd.GeoDataFrame(columns=['Name', 'geometry'])

# --- Upload do Arquivo KML ---
kml_file = st.file_uploader("Carregue o arquivo KML dos talhões", type="kml")

if kml_file:
    gdf_kml = extrair_kml(kml_file.read().decode("utf-8"))
    gdf_kml["Name"] = gdf_kml["Name"].apply(formatar_nome)

    talhoes_disponiveis = sorted(gdf_kml["Name"].dropna().unique())
    talhoes_selecionados = st.multiselect(
        "Selecione o(s) talhão(ões) para análise",
        talhoes_disponiveis,
        default=talhoes_disponiveis[:1],
        help="Escolha um ou mais talhões para análise de NDVI e biomassa."
    )

    if talhoes_selecionados:
        st.info("Buscando imagens Sentinel-2 (≥90% cobertura, ≤20% nuvem)...")

        # Criar FeatureCollection para os talhões selecionados
        talhoes_filtrados = gdf_kml[gdf_kml["Name"].isin(talhoes_selecionados)]
        fc_talhoes = ee.FeatureCollection(talhoes_filtrados.__geo_interface__)

        # Obter imagens Sentinel-2
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(fc_talhoes.geometry())
              .filterDate('2025-07-01', '2025-07-15')
              .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 20)))

        # Calcular percentual de cobertura
        def calcular_cobertura(img):
            area_inter = img.geometry().intersection(fc_talhoes.geometry(), 1).area()
            area_total = fc_talhoes.geometry().area()
            return img.set('percentual_cobertura', area_inter.divide(area_total).multiply(100))

        s2_coberta = s2.map(calcular_cobertura).filter(ee.Filter.gte('percentual_cobertura', 90))
        melhor_img = s2_coberta.sort('CLOUDY_PIXEL_PERCENTAGE').first()

        if melhor_img:
            data_melhor = ee.Date(melhor_img.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            st.success(f"Melhor imagem encontrada: {data_melhor}")

            # Inicializar mapa centrado nos talhões
            centroid = talhoes_filtrados.geometry.centroid.iloc[0]
            mapa = geemap.Map(
                center=[centroid.y, centroid.x],
                zoom=12,
                add_google_map=False,
                width="100%",
                height="400px"
            )

            # Adicionar camada RGB
            rgb_params = {
                "bands": ["B4", "B3", "B2"],
                "min": 0,
                "max": 3000,
                "gamma": 1.4
            }
            mapa.add_layer(melhor_img.clip(fc_talhoes), rgb_params, name=f"RGB - {data_melhor}")

            # Calcular e adicionar camada NDVI
            ndvi = melhor_img.normalizedDifference(['B8', 'B4']).rename('NDVI')
            ndvi_params = {
                "min": 0.0,
                "max": 1.0,
                "palette": ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
            }
            mapa.add_layer(ndvi.clip(fc_talhoes), ndvi_params, name=f"NDVI - {data_melhor}")

            # Adicionar limites dos talhões
            mapa.add_layer(fc_talhoes.style(**{
                "color": "blue",
                "fillColor": "00000000",
                "width": 2
            }), name="Talhões")

            # Exibir mapa
            st_folium(mapa, width="100%", height=400)

            # Calcular percentual de biomassa (NDVI ≥ 0.5)
            def percentual_biomassa(feat):
                geom = feat.geometry()
                ndvi_clip = ndvi.clip(geom)
                stats = ndvi_clip.reduceRegion(
                    reducer=ee.Reducer.frequencyHistogram(),
                    geometry=geom,
                    scale=10,
                    maxPixels=1e9
                )
                hist = ee.Dictionary(stats.get('NDVI'))
                total_pixels = hist.values().reduce(ee.Reducer.sum())
                pixels_verde = hist.keys().map(lambda k: ee.Number.parse(k)).filter(ee.Filter.gte('', 0.5))
                valores_verde = pixels_verde.map(lambda k: hist.get(ee.String(k)))
                soma_verde = ee.List(valores_verde).reduce(ee.Reducer.sum())
                pct_verde = ee.Number(soma_verde).divide(total_pixels).multiply(100)
                return feat.set('biomassa_pct', pct_verde)

            fc_resultado = fc_talhoes.map(percentual_biomassa)
            biomass_data = fc_resultado.getInfo()['features']
            biomass_dict = {
                f['properties']['Name']: f['properties']['biomassa_pct']
                for f in biomass_data if 'biomassa_pct' in f['properties']
            }
            biomass_df = pd.DataFrame.from_dict(
                biomass_dict,
                orient='index',
                columns=['Biomassa (%)']
            ).reset_index().rename(columns={'index': 'Talhão'})
            biomass_df['Biomassa (%)'] = biomass_df['Biomassa (%)'].round(2)

            # Exibir gráfico de biomassa
            st.subheader("📊 Percentual de Biomassa (NDVI ≥ 0.5)")
            st.markdown("Percentual de pixels com NDVI ≥ 0.5 por talhão, indicando vegetação saudável.")
            fig = px.bar(
                biomass_df,
                x="Talhão",
                y="Biomassa (%)",
                text="Biomassa (%)",
                color="Biomassa (%)",
                color_continuous_scale="Greens",
                range_y=[0, 100],
                height=360
            )
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig.update_layout(
                font=dict(family='-apple-system', size=15),
                plot_bgcolor='#f5f5f7',
                paper_bgcolor='#f5f5f7',
                margin=dict(l=10, r=10, t=30, b=10),
                title="Percentual de Biomassa por Talhão"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Exibir tabela de dados de biomassa
            st.subheader("📋 Dados de Biomassa")
            st.dataframe(biomass_df.style.format({"Biomassa (%)": "{:.2f}%"}), use_container_width=True)
        else:
            st.warning("Nenhuma imagem Sentinel-2 encontrada com os critérios de nuvem e cobertura.")
    else:
        st.warning("Selecione pelo menos um talhão para análise.")
else:
    st.warning("Carregue o arquivo KML para iniciar a análise.")

# --- Rodapé com Mensagem Motivacional ---
st.markdown("""
<div style='text-align:center; color:#aaa; font-size:15px; margin-top:22px'>
Transformando o futuro do campo com tecnologia, nosso projeto leva inovação e sustentabilidade aos talhões!
</div>
""", unsafe_allow_html=True)
