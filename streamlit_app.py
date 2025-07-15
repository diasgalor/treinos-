import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, Point
from unidecode import unidecode
import io
import numpy as np
import ee
import geemap.foliumap as geemap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- Page Configuration ---
st.set_page_config(page_title="SLC Mobile", layout="wide", initial_sidebar_state="collapsed")

# --- Apple-Inspired Styling ---
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
.radial-card {
    background: #232323;
    border-radius: 18px;
    padding: 24px 8px 12px 8px;
    color: #fff;
    width: 210px;
    margin: auto;
    box-shadow: 0 2px 16px rgba(0,0,0,0.12);
}
.stTabs > div[role='tablist'] > div { background: #e5e5ea; border-radius: 12px; padding: 4px; }
.stTabs > div[role='tablist'] > div > button { border-radius: 10px; font-size: 14px; padding: 8px; }
.stTabs > div[role='tablist'] > div > button[data-baseweb='tab-highlight'] { background: #007aff; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🌱 Monitoramento Climático SLC Mobile")

# --- UTILS ---
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
        st.error(f"Erro ao processar KML: {e}")
        return gpd.GeoDataFrame(columns=['Name', 'geometry'])

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

def interpolacao_idw(df, x_col='VL_LONGITUDE', y_col='VL_LATITUDE', val_col='DBM',
                     resolution=0.002, buffer=0.05, geom_mask=None):
    df = df.dropna(subset=[val_col]).copy()
    if df.empty:
        return None, None, None, None

    if df[val_col].dropna().between(1, 4).all():
        df['class_num'] = df[val_col].astype(int)
    else:
        df['class_num'] = df[val_col].apply(classificar_dbm)

    minx, miny = df[x_col].min() - buffer, df[y_col].min() - buffer
    maxx, maxy = df[x_col].max() + buffer, df[y_col].max() + buffer

    x_grid = np.arange(minx, maxx, resolution)
    y_grid = np.arange(miny, maxy, resolution)

    if len(x_grid) * len(y_grid) > 1_000_000:
        return None, None, None, None

    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    from scipy.spatial.distance import cdist
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

# --- Initialize Earth Engine ---
try:
    ee.Initialize(project='ee-diasgalor')
except Exception as e:
    st.error("Erro ao conectar com o Earth Engine. Verifique autenticação.")
    st.stop()

# --- Upload Section ---
st.header("Upload dos Arquivos")
excel_file = st.file_uploader("Excel consolidada (.xlsx)", type=["xlsx"])
kml_file = st.file_uploader("Mapa Limites (.kml)", type=["kml"])
df, gdf_kml = None, None

@st.cache_data
def carregar_dados(excel_file, kml_file):
    df = pd.read_excel(excel_file) if excel_file else None
    gdf = extrair_kml(kml_file.read().decode("utf-8")) if kml_file else None
    return df, gdf

if excel_file and kml_file:
    df, gdf_kml = carregar_dados(excel_file, kml_file)

if df is not None:
    df["VL_LATITUDE"] = pd.to_numeric(df.get("VL_LATITUDE", None), errors="coerce")
    df["VL_LONGITUDE"] = pd.to_numeric(df.get("VL_LONGITUDE", None), errors="coerce")
    df["UNIDADE"] = df["UNIDADE"].apply(formatar_nome)

# --- Tabs ---
tabs = st.tabs(["📊 Gráficos", "🗺️ Mapa Interativo", "🎯 Interpolação IDW", "🛰️ NDVI e Biomassa"])

# --- Tab 1: Gráficos ---
with tabs[0]:
    st.subheader("🔢 Distribuição de Firmwares por Unidade")
    if df is not None and "VL_FIRMWARE_EQUIPAMENTO" in df.columns and "UNIDADE" in df.columns:
        df_firmware = df.groupby(['VL_FIRMWARE_EQUIPAMENTO', 'UNIDADE']).size().reset_index(name='Quantidade')
        fig = px.bar(
            df_firmware, x='Quantidade', y='UNIDADE', color='VL_FIRMWARE_EQUIPAMENTO',
            orientation='h', text='Quantidade',
            color_discrete_sequence=['#007aff', '#34c759', '#e5e5ea', '#1d1d1f'],
            title='Distribuição de Firmwares por Unidade'
        )
        fig.update_layout(
            font=dict(family='-apple-system', size=15),
            plot_bgcolor='#f5f5f7', paper_bgcolor='#f5f5f7',
            bargap=0.25, height=340, margin=dict(l=16, r=16, t=40, b=20),
            legend_title='Firmware'
        )
        fig.update_traces(textposition='outside', marker=dict(line=dict(color='#e5e5ea', width=2)), textfont=dict(size=13, color='#222'))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🌦️ Pluviômetros e Estações por Unidade")
    if df is not None and "DESC_TIPO_EQUIPAMENTO" in df.columns:
        df_contagem_1 = df[df["DESC_TIPO_EQUIPAMENTO"].str.contains("ESTACAO|PLUVIOMETRO", case=False, na=False)]
        df_contagem_1 = df_contagem_1.groupby(['UNIDADE', 'DESC_TIPO_EQUIPAMENTO']).size().reset_index(name='Quantidade')
        fig1 = px.bar(
            df_contagem_1, x='UNIDADE', y='Quantidade', color='DESC_TIPO_EQUIPAMENTO',
            text='Quantidade', barmode='stack',
            color_discrete_sequence=['#007aff', '#34c759', '#8E44AD'],
            title='Pluviômetros e Estações por Unidade'
        )
        fig1.update_layout(
            height=340, legend_title='Tipo de Equipamento', plot_bgcolor='#f5f5f7', paper_bgcolor='#f5f5f7'
        )
        fig1.update_traces(textposition='outside')
        st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📶 Tipos de Comunicação por Unidade (Sem 4G)")
    if df is not None and "TIPO_COMUNICACAO" in df.columns:
        df_contagem_2 = df[df['TIPO_COMUNICACAO'] != '4G'].groupby(['UNIDADE', 'TIPO_COMUNICACAO']).size().reset_index(name='Quantidade')
        fig2 = px.bar(
            df_contagem_2, x='UNIDADE', y='Quantidade', color='TIPO_COMUNICACAO',
            text='Quantidade', barmode='stack',
            color_discrete_sequence=['#2E86C1', '#28B463', '#8E44AD'],
            title='Tipos de Comunicação por Unidade'
        )
        fig2.update_layout(
            height=340, legend_title='Tipo de Comunicação', plot_bgcolor='#f5f5f7', paper_bgcolor='#f5f5f7'
        )
        fig2.update_traces(textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📡 Percentual de Equipamentos com Dados Móveis")
    if df is not None and "D_MOVEIS_AT" in df.columns:
        total_equipamentos = len(df)
        equipamentos_moveis = df["D_MOVEIS_AT"].str.upper().eq("SIM").sum()
        percentual = round((equipamentos_moveis / total_equipamentos) * 100) if total_equipamentos > 0 else 0

        st.markdown('<div class="radial-card">', unsafe_allow_html=True)
        st.markdown('**Percentual Dados Móveis**', unsafe_allow_html=True)

        fig3 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = percentual,
            number = {'suffix': "%", 'font': {'size': 36, 'color': "#fff"}},
            gauge = {
                'shape': 'angular',
                'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': "#fff"},
                'bar': {'color': "#d1d5db", 'thickness': 0.23},
                'bgcolor': "#1a1a1a",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, percentual], 'color': '#d1d5db'},
                    {'range': [percentual, 100], 'color': '#232323'}
                ],
                'threshold': {'line': {'color': "#d1d5db", 'width': 6}, 'thickness': 0.6, 'value': percentual}
            },
            domain = {'x': [0, 1], 'y': [0, 1]}
        ))
        fig3.update_layout(
            height=180, width=180,
            margin=dict(l=10, r=10, t=0, b=0),
            paper_bgcolor="#232323",
            font=dict(color="#fff", size=18)
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown(f'<div style="text-align:center;">Conquistado: <b>{equipamentos_moveis}/{total_equipamentos}</b> equipamentos</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 2: Mapa Interativo ---
with tabs[1]:
    st.subheader("🗺️ Mapa de Equipamentos e Limites")
    if df is not None and gdf_kml is not None:
        map_center = [df["VL_LATITUDE"].mean(), df["VL_LONGITUDE"].mean()]
        mapa = folium.Map(location=map_center, zoom_start=11, height=430)

        marker_cluster = MarkerCluster().add_to(mapa)
        if "DESC_TIPO_EQUIPAMENTO" in df.columns and "FROTA" in df.columns:
            df_estacoes = df[df["DESC_TIPO_EQUIPAMENTO"].str.contains("ESTACAO", case=False, na=False)]
            for _, row in df_estacoes.iterrows():
                folium.Marker(
                    location=[row["VL_LATITUDE"], row["VL_LONGITUDE"]],
                    popup=str(row["FROTA"]),
                    icon=folium.Icon(color="blue", icon="cloud", prefix="fa")
                ).add_to(marker_cluster)
        if "STATUS" in df.columns:
            df_pluviometros_ativos = df[
                (df["DESC_TIPO_EQUIPAMENTO"].str.contains("PLUVIOMETRO", case=False, na=False)) &
                (df["STATUS"].str.upper() == "ATIVO")
            ]
            for _, row in df_pluviometros_ativos.iterrows():
                folium.Marker(
                    location=[row["VL_LATITUDE"], row["VL_LONGITUDE"]],
                    popup=str(row["FROTA"]),
                    icon=folium.Icon(color="green", icon="tint", prefix="fa")
                ).add_to(marker_cluster)
        if not gdf_kml.empty:
            folium.GeoJson(
                gdf_kml,
                name="Limites",
                tooltip=folium.GeoJsonTooltip(fields=["Name"], aliases=["Fazenda:"]),
                style_function=lambda x: {"fillColor": "#007aff40", "color": "#007aff", "weight": 2, "fillOpacity": 0.19}
            ).add_to(mapa)
        folium.LayerControl().add_to(mapa)
        st_folium(mapa, width=370, height=430)

# --- Tab 3: Interpolação IDW ---
with tabs[2]:
    st.subheader("🎯 Interpolação IDW do Sinal")
    if df is not None and gdf_kml is not None and "DBM" in df.columns and "UNIDADE" in df.columns:
        unidades = sorted(df["UNIDADE"].dropna().unique())
        unidade_sel = st.selectbox("Selecione a Fazenda para interpolação", unidades)
        df_fazenda = df[df['UNIDADE'] == unidade_sel].copy()
        fazenda_mask = None
        if not gdf_kml.empty and "Name" in gdf_kml.columns:
            gdf_kml["NamePad"] = gdf_kml["Name"].apply(formatar_nome)
            geom_df = gdf_kml[gdf_kml["NamePad"] == unidade_sel]
            if not geom_df.empty:
                fazenda_mask = geom_df.unary_union

        df_fazenda['DBM'] = pd.to_numeric(df_fazenda.get('DBM', None), errors='coerce')
        has_dbm = 'DBM' in df_fazenda.columns and not df_fazenda['DBM'].dropna().empty
        if has_dbm and not df_fazenda.empty and fazenda_mask is not None:
            grid_x, grid_y, grid_numerico, bounds = interpolacao_idw(
                df_fazenda,
                x_col='VL_LONGITUDE',
                y_col='VL_LATITUDE',
                val_col='DBM',
                resolution=0.002,
                buffer=0.05,
                geom_mask=fazenda_mask
            )

            minx, maxx, miny, maxy = bounds
            colors = {
                1: '#fc8d59',  # ruim
                2: '#fee08b',  # regular
                3: '#91cf60',  # bom
                4: '#1a9850'   # ótimo
            }
            cmap = ListedColormap([colors[i] for i in range(1, 5)])

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(grid_numerico,
                           extent=(minx, maxx, miny, maxy),
                           origin='lower',
                           cmap=cmap,
                           interpolation='nearest',
                           alpha=0.8)
            if fazenda_mask is not None and not getattr(fazenda_mask, "is_empty", False):
                gpd.GeoDataFrame(geometry=[fazenda_mask]).boundary.plot(ax=ax, color='black', linewidth=2)
            ax.scatter(df_fazenda["VL_LONGITUDE"], df_fazenda["VL_LATITUDE"], c="black", s=55, edgecolors='w')
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title(f"IDW do Sinal ({unidade_sel})")
            cbar = plt.colorbar(im, ax=ax, ticks=[1.5, 2.5, 3.5, 4.5])
            cbar.ax.set_yticklabels(['Ruim', 'Regular', 'Bom', 'Ótimo'])
            cbar.set_label('Classe de Sinal')
            plt.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
        else:
            st.info("Não há dados de DBM ou geometria para esta unidade.")
    else:
        st.info("Faça upload dos arquivos e selecione uma unidade para interpolação.")

# --- Tab 4: NDVI e Biomassa ---
with tabs[3]:
    st.subheader("🛰️ NDVI e Biomassa por Talhão")
    if gdf_kml is not None and not gdf_kml.empty:
        gdf_kml["Name"] = gdf_kml["Name"].apply(formatar_nome)
        talhoes_disponiveis = sorted(gdf_kml["Name"].dropna().unique())
        talhoes_selecionados = st.multiselect(
            "Selecione Talhão(ns) para análise NDVI",
            talhoes_disponiveis,
            default=talhoes_disponiveis[:1],
            help="Escolha um ou mais talhões para análise."
        )

        if talhoes_selecionados:
            st.info("Buscando imagens Sentinel-2 (cobertura ≥90%, nuvens ≤20%)...")

            # Create FeatureCollection for selected plots
            talhoes_filtrados = gdf_kml[gdf_kml["Name"].isin(talhoes_selecionados)]
            fc_talhoes = ee.FeatureCollection(talhoes_filtrados.__geo_interface__)

            # Fetch Sentinel-2 imagery
            s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(fc_talhoes.geometry())
                  .filterDate('2025-07-01', '2025-07-15')
                  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 20)))

            # Calculate coverage percentage
            def calcular_cobertura(img):
                area_inter = img.geometry().intersection(fc_talhoes.geometry(), 1).area()
                area_total = fc_talhoes.geometry().area()
                return img.set('percentual_cobertura', area_inter.divide(area_total).multiply(100))

            s2_coberta = s2.map(calcular_cobertura).filter(ee.Filter.gte('percentual_cobertura', 90))
            melhor_img = s2_coberta.sort('CLOUDY_PIXEL_PERCENTAGE').first()

            if melhor_img:
                data_melhor = ee.Date(melhor_img.get('system:time_start')).format('YYYY-MM-dd').getInfo()
                st.success(f"Melhor imagem encontrada: {data_melhor}")

                # Initialize map centered on the plots
                centroid = talhoes_filtrados.geometry.centroid.iloc[0]
                mapa = geemap.Map(
                    center=[centroid.y, centroid.x],
                    zoom=12,
                    add_google_map=False,
                    width="100%",
                    height="400px"
                )

                # Add RGB composite
                rgb_params = {
                    'bands': ['B4', 'B3', 'B2'],
                    'min': 0,
                    'max': 3000,
                    'gamma': 1.4
                }
                rgb_tile = geemap.ee_tile_layer(melhor_img.clip(fc_talhoes), rgb_params, f'RGB - {data_melhor}')
                mapa.add_child(rgb_tile)

                # Calculate and add NDVI layer
                ndvi = melhor_img.normalizedDifference(['B8', 'B4']).rename('NDVI')
                ndvi_params = {
                    'min': 0.0,
                    'max': 1.0,
                    'palette': ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
                }
                ndvi_tile = geemap.ee_tile_layer(ndvi.clip(fc_talhoes), ndvi_params, f'NDVI - {data_melhor}')
                mapa.add_child(ndvi_tile)
                mapa.keep_in_front(ndvi_tile)

                # Add plot boundaries
                talhoes_layer = geemap.ee_tile_layer(fc_talhoes.style(**{
                    'color': 'blue',
                    'fillColor': '00000000',
                    'width': 2
                }), {}, 'Talhões')
                mapa.add_child(talhoes_layer)

                # Display map
                st_folium(mapa, width="100%", height=400)

                # Calculate biomass percentage (NDVI ≥ 0.5)
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

                # Prepare data for chart
                biomass_df = pd.DataFrame.from_dict(
                    biomass_dict,
                    orient='index',
                    columns=['Biomassa (%)']
                ).reset_index().rename(columns={'index': 'Talhão'})
                biomass_df['Biomassa (%)'] = biomass_df['Biomassa (%)'].round(2)

                # Display biomass chart
                st.subheader("📊 Índice de Biomassa (NDVI ≥ 0.5)")
                st.markdown("Percentual de pixels com NDVI ≥ 0.5 por talhão, indicando vegetação saudável.")

                ```chartjs
                {
                    "type": "bar",
                    "data": {
                        "labels": ${biomass_df['Talhão'].to_json(orient='values')},
                        "datasets": [{
                            "label": "Biomassa (%)",
                            "data": ${biomass_df['Biomassa (%)'].to_json(orient='values')},
                            "backgroundColor": [
                                "#66bd63",
                                "#a6d96a",
                                "#d9ef8b",
                                "#fee08b"
                            ],
                            "borderColor": [
                                "#1a9850",
                                "#66bd63",
                                "#a6d96a",
                                "#d9ef8b"
                            ],
                            "borderWidth": 1
                        }]
                    },
                    "options": {
                        "responsive": true,
                        "maintainAspectRatio": false,
                        "scales": {
                            "y": {
                                "beginAtZero": true,
                                "max": 100,
                                "title": {
                                    "display": true,
                                    "text": "Biomassa (%)"
                                }
                            },
                            "x": {
                                "title": {
                                    "display": true,
                                    "text": "Talhão"
                                }
                            }
                        },
                        "plugins": {
                            "legend": {
                                "display": false
                            },
                            "title": {
                                "display": true,
                                "text": "Percentual de Biomassa por Talhão"
                            }
                        }
                    }
                }
