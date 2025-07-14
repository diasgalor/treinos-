import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, Point
from unidecode import unidecode
import io
import numpy as np

# --- ESTILO MOBILE APPLE-INSPIRED ---
st.set_page_config(page_title="SLC Mobile", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; background: #f5f5f7; color: #222;}
.stButton>button { background: linear-gradient(90deg, #007aff 70%, #34c759 100%); color:white; border-radius:18px; border:none; padding:14px 0; font-size:18px; width:100%; margin-top:8px; box-shadow:0 2px 16px rgba(0,0,0,0.08);}
.stButton>button:hover { background: #005bb5; }
.stSelectbox, .stFileUploader { margin-bottom:1rem; border-radius:14px !important; border:1px solid #e5e5ea !important; box-shadow:0 1px 8px rgba(0,0,0,0.04);}
@media (max-width: 600px) {
    h1 { font-size:1.2rem; }
    h2 { font-size:1rem; }
    .stButton>button { font-size:16px; padding:12px 0;}
}
</style>
""", unsafe_allow_html=True)

st.title("🌱 Monitoramento Climático SLC Mobile")

# ----- UTILS -----
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

    pontos = df[[x_col, y_col]].values
    valores = df['class_num'].values

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

# ----- UPLOAD -----
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

# ----- TABS -----
tabs = st.tabs(["📊 Gráficos", "🗺️ Mapa Interativo", "🎯 Interpolação IDW"])

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
        contagem_moveis = df['D_MOVEIS_AT'].value_counts()
        fig3 = px.pie(
            values=contagem_moveis.values, names=contagem_moveis.index,
            title='Dados Móveis', hole=0.5,
            color_discrete_sequence=['#007aff', '#34c759', '#e5e5ea']
        )
        fig3.update_traces(textinfo='percent+label', textfont_size=15, marker=dict(line=dict(color='#e5e5ea', width=2)))
        fig3.update_layout(showlegend=True, legend_title='Dados Móveis', font=dict(family='-apple-system', size=14),
                        plot_bgcolor='#f5f5f7', paper_bgcolor='#f5f5f7', margin=dict(l=10, r=10, t=30, b=10), height=280)
        st.plotly_chart(fig3, use_container_width=True)

with tabs[1]:
    st.subheader("🗺️ Mapa de Equipamentos e Limites")
    if df is not None and gdf_kml is not None:
        map_center = [df["VL_LATITUDE"].mean(), df["VL_LONGITUDE"].mean()]
        mapa = folium.Map(location=map_center, zoom_start=11, height=430)

        marker_cluster = MarkerCluster().add_to(mapa)
        # Estações meteorológicas
        if "DESC_TIPO_EQUIPAMENTO" in df.columns and "FROTA" in df.columns:
            df_estacoes = df[df["DESC_TIPO_EQUIPAMENTO"].str.contains("ESTACAO", case=False, na=False)]
            for _, row in df_estacoes.iterrows():
                folium.Marker(
                    location=[row["VL_LATITUDE"], row["VL_LONGITUDE"]],
                    popup=str(row["FROTA"]),
                    icon=folium.Icon(color="blue", icon="cloud", prefix="fa")
                ).add_to(marker_cluster)
        # Pluviômetros ativos
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
        # Limites do KML
        if not gdf_kml.empty:
            folium.GeoJson(
                gdf_kml,
                name="Limites",
                tooltip=folium.GeoJsonTooltip(fields=["Name"], aliases=["Fazenda:"]),
                style_function=lambda x: {"fillColor": "#007aff40", "color": "#007aff", "weight": 2, "fillOpacity": 0.19}
            ).add_to(mapa)
        folium.LayerControl().add_to(mapa)
        st_folium(mapa, width=370, height=430)

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
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap

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
            # Add points
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

# ----- RODAPÉ -----
st.markdown("""
<div style='text-align:center; color:#aaa; font-size:15px; margin-top:22px'>
Feito para SLC | <span style="font-weight:600;">Design mobile inspirado em iOS</span>
</div>
""", unsafe_allow_html=True)
