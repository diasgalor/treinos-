# streamlit_app.py

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from folium.plugins import MarkerCluster
from shapely.geometry import Point
from io import StringIO
import xml.etree.ElementTree as ET
import base64
from unidecode import unidecode
import re
import gc

st.set_page_config(layout="wide")
st.title("Mapa de Equipamentos Climáticos")

# Funções auxiliares
def corrigir_coordenadas(valor):
    if isinstance(valor, str):
        valor_limpo = re.sub(r"[^0-9\-.]", "", valor).replace(",", ".")
        partes = valor_limpo.split(".")
        if len(partes) > 2:
            valor_limpo = partes[0] + "." + "".join(partes[1:])
        try:
            return float(valor_limpo)
        except ValueError:
            return None
    return valor

def formatar_nome(nome):
    return unidecode(str(nome)).strip().upper().replace(" ", "_") if isinstance(nome, str) else nome

def extrair_metadados_kml(content):
    tree = ET.fromstring(content)
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    dados = []
    for placemark in tree.findall('.//kml:Placemark', ns):
        item = {}
        name = placemark.find('kml:name', ns)
        item['Name'] = name.text if name is not None else None
        for sd in placemark.findall('.//kml:SimpleData', ns):
            item[sd.attrib['name']] = sd.text
        dados.append(item)
    return pd.DataFrame(dados)

# Upload de arquivos
st.sidebar.header("Upload de arquivos")
csv_file = st.sidebar.file_uploader("CSV dos equipamentos", type="csv")
kml_file = st.sidebar.file_uploader("KML com limites das fazendas", type="kml")

if csv_file:
    df_csv = pd.read_csv(csv_file, delimiter=';')
    df_csv.columns = df_csv.columns.str.strip()
    df_csv['VL_LATITUDE'] = df_csv['VL_LATITUDE'].apply(corrigir_coordenadas)
    df_csv['VL_LONGITUDE'] = df_csv['VL_LONGITUDE'].apply(corrigir_coordenadas)
    df_csv['UNIDADE'] = df_csv['UNIDADE'].apply(formatar_nome)

if kml_file:
    gdf_kml = gpd.read_file(kml_file, driver='KML')
    if gdf_kml.crs is None or gdf_kml.crs.to_epsg() != 4326:
        gdf_kml = gdf_kml.to_crs(epsg=4326)
    content = kml_file.getvalue().decode('utf-8')
    df_metadados = extrair_metadados_kml(content)
    col_nome = 'NOME_FAZ' if 'NOME_FAZ' in df_metadados.columns else 'Name'
    gdf_kml['NomeFazendaKML'] = df_metadados[col_nome].apply(formatar_nome)

if csv_file and kml_file:
    aba = st.sidebar.radio("Escolha a visualização:", ['Dashboard', 'Mapa', 'Interpolação'])

    if aba == 'Dashboard':
        st.header("Gráficos Interativos")
        col1, col2 = st.columns(2)

        df_firmware = df_csv.groupby(['VL_FIRMWARE_EQUIPAMENTO', 'UNIDADE']).size().reset_index(name='Quantidade')
        fig1 = px.bar(df_firmware, x='Quantidade', y='UNIDADE', color='VL_FIRMWARE_EQUIPAMENTO', orientation='h',
                      title='Firmwares por Unidade')
        col1.plotly_chart(fig1, use_container_width=True)

        df_tipo = df_csv[df_csv["DESC_TIPO_EQUIPAMENTO"].str.contains("ESTACAO|PLUVIOMETRO", na=False, case=False)]
        df_tipo = df_tipo.groupby(['UNIDADE', 'DESC_TIPO_EQUIPAMENTO']).size().reset_index(name='Quantidade')
        fig2 = px.bar(df_tipo, x='UNIDADE', y='Quantidade', color='DESC_TIPO_EQUIPAMENTO', barmode='stack',
                      title='Equipamentos por Unidade')
        col2.plotly_chart(fig2, use_container_width=True)

        st.subheader("Dados Móveis")
        if 'D_MOVEIS_AT' in df_csv.columns:
            contagem = df_csv['D_MOVEIS_AT'].value_counts()
            fig3 = px.pie(values=contagem.values, names=contagem.index, hole=0.5, title='Equipamentos com Dados Móveis')
            st.plotly_chart(fig3, use_container_width=True)

    elif aba == 'Mapa':
        st.header("Mapa Interativo")
        mapa = folium.Map(location=[df_csv['VL_LATITUDE'].mean(), df_csv['VL_LONGITUDE'].mean()], zoom_start=10)
        cluster = MarkerCluster().add_to(mapa)

        for _, row in df_csv.iterrows():
            tipo = str(row['DESC_TIPO_EQUIPAMENTO']).lower()
            cor = 'green' if 'pluviometro' in tipo else 'blue' if 'estacao' in tipo else 'red'
            folium.Marker(
                location=[row['VL_LATITUDE'], row['VL_LONGITUDE']],
                popup=f"{row['FROTA']}",
                icon=folium.Icon(color=cor)
            ).add_to(cluster)

        if gdf_kml is not None:
            folium.GeoJson(
                gdf_kml,
                name="Limites",
                tooltip=folium.GeoJsonTooltip(fields=["NomeFazendaKML"], aliases=["Fazenda:"])
            ).add_to(mapa)

        folium.LayerControl().add_to(mapa)

        # Exibir no Streamlit
        from streamlit_folium import st_folium
        st_data = st_folium(mapa, width=1100, height=600)

    elif aba == 'Interpolação':
        st.header("Interpolação do Sinal por Fazenda")
        fazendas = sorted(gdf_kml['NomeFazendaKML'].dropna().unique())
        escolha = st.selectbox("Selecione uma fazenda:", fazendas)

        df_csv['UNIDADE_Padronizada'] = df_csv['UNIDADE'].apply(formatar_nome)
        gdf_kml['NomeFazendaKML_Padronizada'] = gdf_kml['NomeFazendaKML'].apply(formatar_nome)

        df_faz = df_csv[df_csv['UNIDADE_Padronizada'] == formatar_nome(escolha)].copy()
        geom = gdf_kml[gdf_kml['NomeFazendaKML_Padronizada'] == formatar_nome(escolha)].geometry.unary_union

        if 'DBM' in df_faz.columns:
            from scipy.spatial.distance import cdist
            df = df_faz.dropna(subset=['DBM'])
            df['class'] = df['DBM'].apply(lambda x: 4 if x > -70 else 3 if x > -85 else 2 if x > -100 else 1)

            x, y = df['VL_LONGITUDE'], df['VL_LATITUDE']
            val = df['class']
            grid_x, grid_y = np.meshgrid(
                np.linspace(x.min(), x.max(), 100),
                np.linspace(y.min(), y.max(), 100)
            )
            coords = np.c_[grid_x.ravel(), grid_y.ravel()]
            dist = cdist(coords, df[['VL_LONGITUDE', 'VL_LATITUDE']])
            weights = 1 / (dist**2 + 1e-9)
            idw = np.dot(weights, val) / weights.sum(axis=1)
            idw = idw.reshape(grid_x.shape)

            fig, ax = plt.subplots(figsize=(10, 8))
            cmap = plt.cm.get_cmap('RdYlGn', 4)
            im = ax.imshow(idw, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap=cmap)
            plt.colorbar(im, ax=ax, ticks=[1, 2, 3, 4], label='Qualidade do Sinal')
            ax.set_title(f"Interpolação do Sinal - {escolha}")
            st.pyplot(fig)

        else:
            st.warning("Coluna 'DBM' não encontrada nos dados da fazenda selecionada.")

else:
    st.info("Envie os arquivos CSV e KML na barra lateral para iniciar.")
