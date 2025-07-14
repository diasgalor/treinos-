import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, Point
from unidecode import unidecode
import numpy as np
import matplotlib.pyplot as plt

# --- ESTILO MOBILE STRAVA-INSPIRED ---
st.set_page_config(page_title="SLC Mobile", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; background: #f5f5f7; color: #222;}
.stButton>button { background: linear-gradient(90deg, #FC4C02 70%, #34c759 100%); color:white; border-radius:18px; border:none; padding:14px 0; font-size:18px; width:100%; margin-top:8px; box-shadow:0 2px 16px rgba(0,0,0,0.08);}
.stButton>button:hover { background: #FC4C02; }
.stSelectbox, .stFileUploader { margin-bottom:1rem; border-radius:14px !important; border:1px solid #e5e5ea !important; box-shadow:0 1px 8px rgba(0,0,0,0.04);}
@media (max-width: 600px) {
    h1 { font-size:1.2rem; }
    h2 { font-size:1rem; }
    .stButton>button { font-size:16px; padding:12px 0;}
}
</style>
""", unsafe_allow_html=True)

st.title("🌱 Monitoramento Climático SLC (Strava Style)")

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

    from scipy.spatial.distance import cdist
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
    # --- Strava-style: Gráfico de Linha com Área (Temperatura) ---
    if df is not None and "Temperatura" in df.columns and "Data" in df.columns:
        st.subheader("🌡️ Temperatura (Strava Style)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Data"], y=df["Temperatura"],
            mode="lines",
            line=dict(width=5, color="#FC4C02"),
            fill="tozeroy",
            fillcolor="rgba(252,76,2,0.18)",
            hoverinfo="x+y",
            name="Temperatura"
        ))
        fig.update_layout(
            margin=dict(l=10, r=10, t=25, b=10),
            height=220,
            plot_bgcolor="#fff",
            paper_bgcolor="#fff",
            font=dict(size=18),
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=True, tickfont=dict(size=14), ticks="outside"),
            yaxis=dict(showgrid=False, showticklabels=True, tickfont=dict(size=14), ticks="outside"),
        )
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True)

    # --- Strava-style: Gráfico de Barras ---
    if df is not None and "VL_FIRMWARE_EQUIPAMENTO" in df.columns and "UNIDADE" in df.columns:
        st.subheader("🔢 Distribuição de Firmwares por Unidade (Strava Style)")
        df_firmware = df.groupby(['VL_FIRMWARE_EQUIPAMENTO', 'UNIDADE']).size().reset_index(name='Quantidade')
        fig = go.Figure()
        for fw in df_firmware["VL_FIRMWARE_EQUIPAMENTO"].unique():
            df_fw = df_firmware[df_firmware["VL_FIRMWARE_EQUIPAMENTO"] == fw]
            fig.add_trace(go.Bar(
                x=df_fw["UNIDADE"], y=df_fw["Quantidade"],
                name=str(fw),
                marker=dict(color="#FC4C02", line=dict(color="#fff", width=0)),
                width=0.55,
                text=df_fw["Quantidade"], textposition="outside",
                hoverinfo="x+y",
            ))
        fig.update_layout(
            barmode="stack",
            height=220,
            plot_bgcolor="#fff",
            paper_bgcolor="#fff",
            margin=dict(l=10, r=10, t=25, b=10),
            font=dict(size=18),
            showlegend=True,
            xaxis=dict(tickfont=dict(size=14)),
            yaxis=dict(tickfont=dict(size=14), rangemode="tozero"),
            legend_title="Firmware"
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Strava-style: Gráfico de Rosca/Dados Móveis ---
    if df is not None and "D_MOVEIS_AT" in df.columns:
        st.subheader("📡 Percentual de Equipamentos com Dados Móveis (Strava Style)")
        contagem_moveis = df['D_MOVEIS_AT'].value_counts()
        fig3 = go.Figure(go.Pie(
            values=contagem_moveis.values,
            labels=contagem_moveis.index,
            hole=0.55,
            marker=dict(colors=["#FC4C02", "#34c759", "#e5e5ea"]),
            textinfo='percent+label',
            textfont_size=17
        ))
        fig3.update_layout(
            showlegend=True,
            legend_title='Dados Móveis',
            font=dict(size=16),
            margin=dict(l=10, r=10, t=30, b=10),
            height=220,
            plot_bgcolor="#fff",
            paper_bgcolor="#fff"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # --- Strava-style: Gráfico de Barras por Tipo de Comunicação ---
    if df is not None and "TIPO_COMUNICACAO" in df.columns:
        st.subheader("📶 Tipos de Comunicação por Unidade (Strava Style)")
        df_contagem_2 = df[df['TIPO_COMUNICACAO'] != '4G'].groupby(['UNIDADE', 'TIPO_COMUNICACAO']).size().reset_index(name='Quantidade')
        fig2 = go.Figure()
        for tipo in df_contagem_2["TIPO_COMUNICACAO"].unique():
            df_tipo = df_contagem_2[df_contagem_2["TIPO_COMUNICACAO"] == tipo]
            fig2.add_trace(go.Bar(
                x=df_tipo["UNIDADE"], y=df_tipo["Quantidade"],
                name=str(tipo),
                marker=dict(color="#34c759", line=dict(color="#fff", width=0)),
                width=0.55,
                text=df_tipo["Quantidade"], textposition="outside",
                hoverinfo="x+y",
            ))
        fig2.update_layout(
            barmode="stack",
            height=220,
            plot_bgcolor="#fff",
            paper_bgcolor="#fff",
            margin=dict(l=10, r=10, t=25, b=10),
            font=dict(size=18),
            showlegend=True,
            xaxis=dict(tickfont=dict(size=14)),
            yaxis=dict(tickfont=dict(size=14), rangemode="tozero"),
            legend_title="Tipo"
        )
        st.plotly_chart(fig2, use_container_width=True)

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
                style_function=lambda x: {"fillColor": "#FC4C0220", "color": "#FC4C02", "weight": 2, "fillOpacity": 0.19}
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
                gpd.GeoDataFrame(geometry=[fazenda_mask]).boundary.plot(ax=ax, color='#FC4C02', linewidth=2)
            # Add points
            ax.scatter(df_fazenda["VL_LONGITUDE"], df_fazenda["VL_LATITUDE"], c="#FC4C02", s=55, edgecolors='w')
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
Feito para SLC | <span style="font-weight:600;">Design mobile inspirado no Strava/iOS</span>
</div>
""", unsafe_allow_html=True)
