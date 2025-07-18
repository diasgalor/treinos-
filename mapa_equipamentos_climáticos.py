# -*- coding: utf-8 -*-
"""mapa_equipamentos_climáticos.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jTmKIyzW5Nu8dgB6Ihndvp3Zlmzu4Pqz
"""

# @title Sempre que fizerem alterações nos arquivos dentro da pasta, realizem o upload destes novamente abaixo desta celula clicando no icone "PLAY", escolher arquivos.
!pip install unidecode
import folium
import geopandas as gpd
import pandas as pd
import io
import re
from google.colab import files
from unidecode import unidecode
from folium.plugins import MarkerCluster
import xml.etree.ElementTree as ET

# Fazer upload dos arquivos CSV e KML manualmente
print("Por favor, faça o upload do arquivo CSV e do arquivo KML.")
uploaded = files.upload()

# Identificar os arquivos carregados
csv_files = [file for file in uploaded if file.endswith('.csv')]
kml_files = [file for file in uploaded if file.endswith('.kml')]

if not csv_files:
    print("Erro: Nenhum arquivo CSV encontrado. Por favor, faça o upload de um arquivo CSV.")
    # You might want to exit or raise an error here if a CSV is mandatory
    raise FileNotFoundError("Nenhum arquivo CSV encontrado.")

if not kml_files:
    print("Aviso: Nenhum arquivo KML encontrado. A parte do código que depende do KML não será executada.")
    kml_file = None
    gdf_kml = None # Set gdf_kml to None if no KML is uploaded
else:
    csv_file = csv_files[0]
    kml_file = kml_files[0]

    # Ler o KML como GeoDataFrame
    try:
        gdf_kml = gpd.read_file(io.BytesIO(uploaded[kml_file]), driver='KML')
        # Garantir que os polígonos estão no sistema de coordenadas correto
        if gdf_kml.crs is None or gdf_kml.crs.to_epsg() != 4326:
            print("Reprojetando KML para EPSG:4326")
            gdf_kml = gdf_kml.to_crs(epsg=4326)

        # Ler o conteúdo do arquivo KML
        kml_content = uploaded[kml_file].decode('utf-8')
        df_metadados = extrair_metadados_kml(kml_content)

        # Verificar as colunas disponíveis nos metadados
        print("Colunas disponíveis nos metadados do KML:", df_metadados.columns.tolist())

        # Adicionar os metadados ao GeoDataFrame, assumindo que a ordem dos Placemarks corresponde
        if 'NOME_FAZ' in df_metadados.columns:
            print("Coluna 'NOME_FAZ' encontrada nos metadados do KML.")
            gdf_kml['NomeFazendaKML'] = df_metadados['NOME_FAZ'].apply(formatar_nome)
        elif 'Name' in df_metadados.columns:
            print("Coluna 'NOME_FAZ' não encontrada. Usando a coluna 'Name' (pode estar vazia).")
            gdf_kml['NomeFazendaKML'] = df_metadados['Name'].apply(formatar_nome)
        else:
            print("Nenhuma coluna de nome de fazenda ('NOME_FAZ' ou 'Name') encontrada no KML.")
            gdf_kml['NomeFazendaKML'] = None

    except Exception as e:
        print(f"Erro ao processar o arquivo KML: {e}")
        kml_file = None
        gdf_kml = None


# Ler o CSV corretamente
try:
    df_csv = pd.read_csv(io.StringIO(uploaded[csv_files[0]].decode('utf-8')), delimiter=';')
    df_csv.columns = df_csv.columns.str.strip()
except Exception as e:
    print(f"Erro ao ler o arquivo CSV: {e}")
    # You might want to exit or raise an error here
    raise


# Função para corrigir coordenadas
def corrigir_coordenadas(valor):
    if isinstance(valor, str):
        # Remover todos os caracteres não numéricos, exceto "-" e "."
        valor_limpo = re.sub(r"[^0-9\-.]", "", valor)
        # Substituir vírgulas por pontos, se houver
        valor_limpo = valor_limpo.replace(",", ".")
        # Garantir que há no máximo um ponto decimal
        partes = valor_limpo.split(".")
        if len(partes) > 2:
            valor_limpo = partes[0] + "." + "".join(partes[1:])
        # Verificar se o número é negativo
        if valor_limpo.startswith("-"):
            valor_corrigido = "-" + valor_limpo[1:].lstrip("0")
        else:
            valor_corrigido = valor_limpo.lstrip("0")
        # Garantir que o valor final seja um float válido
        try:
            return float(valor_corrigido)
        except ValueError:
            return None
    return None

# Aplicar correção de coordenadas
df_csv["VL_LATITUDE"] = df_csv["VL_LATITUDE"].apply(corrigir_coordenadas)
df_csv["VL_LONGITUDE"] = df_csv["VL_LONGITUDE"].apply(corrigir_coordenadas)

# Função para padronizar nomes
def formatar_nome(nome):
    return unidecode(nome.upper()) if isinstance(nome, str) else nome

# Padronizar nomes das fazendas no CSV
df_csv["UNIDADE"] = df_csv["UNIDADE"].apply(formatar_nome)


# Função para extrair metadados do KML
def extrair_metadados_kml(kml_file_content):
    # Parsear o KML como XML
    tree = ET.fromstring(kml_file_content)
    # Definir o namespace do KML
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    # Lista para armazenar os metadados
    metadados = []

    # Encontrar todos os elementos Placemark
    for placemark in tree.findall('.//kml:Placemark', ns):
        dados = {}
        # Extrair o nome do Placemark (se disponível)
        name_elem = placemark.find('kml:name', ns)
        if name_elem is not None:
            dados['Name'] = name_elem.text
        else:
            dados['Name'] = None
        # Extrair ExtendedData/SimpleData
        for simple_data in placemark.findall('.//kml:SimpleData', ns):
            nome_coluna = simple_data.get('name')
            valor = simple_data.text
            dados[nome_coluna] = valor
        metadados.append(dados)

    # Criar DataFrame com os metadados
    df_metadados = pd.DataFrame(metadados)
    return df_metadados


# Criar o mapa centralizado na média das coordenadas (apenas se houver dados CSV válidos)
if not df_csv.dropna(subset=["VL_LATITUDE", "VL_LONGITUDE"]).empty:
    mapa = folium.Map(location=[df_csv["VL_LATITUDE"].mean(), df_csv["VL_LONGITUDE"].mean()], zoom_start=10)

    # Adicionar os polígonos do KML ao mapa (apenas se gdf_kml foi carregado)
    if gdf_kml is not None and not gdf_kml.empty:
        if 'NomeFazendaKML' in gdf_kml.columns:
            folium.GeoJson(
                gdf_kml,
                name="Limites",
                tooltip=folium.GeoJsonTooltip(fields=["NomeFazendaKML"], aliases=["Fazenda:"]),
                style_function=lambda x: {"fillColor": "blue", "color": "blue", "weight": 1, "fillOpacity": 0.1}
            ).add_to(mapa)
        else:
            folium.GeoJson(
                gdf_kml,
                name="Limites",
                style_function=lambda x: {"fillColor": "blue", "color": "blue", "weight": 1, "fillOpacity": 0.1}
            ).add_to(mapa)
    else:
         print("KML não carregado ou vazio, limites não serão exibidos no mapa inicial.")


    # Exibir o mapa (opcional, pode ser comentado se o segundo bloco for usado)
    # mapa
else:
    print("Nenhum dado CSV válido com coordenadas encontrado para criar o mapa inicial.")

# @title
# Depuração: Verificar o estado após a correção
print("\n=== Depuração: Após correção de coordenadas ===")
print("Primeiras 5 linhas do CSV (após correção):")
print(df_csv[['VL_LATITUDE', 'VL_LONGITUDE']].head())
print("Tipos de dados após correção:")
print(df_csv[['VL_LATITUDE', 'VL_LONGITUDE']].dtypes)
print("Valores nulos após correção:")
print(df_csv[['VL_LATITUDE', 'VL_LONGITUDE']].isna().sum())

# @title Mapa Interativo SLC geral, com depuração
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster

# Verificar se as variáveis necessárias estão definidas
try:
    if 'df_csv' not in globals():
        raise NameError("df_csv não está definido. Execute o primeiro bloco para carregar o DataFrame.")
    if not isinstance(df_csv, pd.DataFrame):
        raise TypeError("df_csv deve ser um pandas DataFrame.")
    if 'gdf_kml' not in globals():
        raise NameError("gdf_kml não está definido. Execute o primeiro bloco para carregar o GeoDataFrame.")
    if not isinstance(gdf_kml, gpd.GeoDataFrame):
        raise TypeError("gdf_kml deve ser um GeoDataFrame.")
except (NameError, TypeError) as e:
    print(f"Erro: {e}")
    print("Certifique-se de que o primeiro bloco foi executado corretamente.")
    raise

# Depuração: Verificar colunas necessárias em df_csv
required_columns = ['VL_LATITUDE', 'VL_LONGITUDE', 'DESC_TIPO_EQUIPAMENTO', 'FROTA', 'STATUS']
missing_columns = [col for col in required_columns if col not in df_csv.columns]
if missing_columns:
    print(f"Erro: Colunas ausentes em df_csv: {missing_columns}")
    raise ValueError("Colunas necessárias não encontradas no DataFrame.")

# Depuração: Verificar coordenadas válidas
print("\n=== Depuração: Estado de df_csv ===")
print("Número de linhas antes de filtrar:", len(df_csv))
print("Valores nulos em VL_LATITUDE e VL_LONGITUDE:")
print(df_csv[['VL_LATITUDE', 'VL_LONGITUDE']].isna().sum())

# Filtrar coordenadas válidas (remover nulos e valores fora do intervalo)
df_csv = df_csv.dropna(subset=["VL_LATITUDE", "VL_LONGITUDE"])
df_csv = df_csv[
    (df_csv["VL_LATITUDE"].between(-90, 90)) &
    (df_csv["VL_LONGITUDE"].between(-180, 180))
]
print("Número de linhas após filtrar coordenadas válidas:", len(df_csv))

# Verificar se há dados válidos para o mapa
if df_csv.empty:
    print("Erro: Nenhuma coordenada válida após filtragem.")
    raise ValueError("DataFrame vazio após filtragem de coordenadas.")

# Depuração: Estatísticas das coordenadas
print("Estatísticas de VL_LATITUDE:")
print(df_csv["VL_LATITUDE"].describe())
print("Estatísticas de VL_LONGITUDE:")
print(df_csv["VL_LONGITUDE"].describe())

# Criar o mapa centralizado na média das coordenadas válidas
try:
    mapa = folium.Map(
        location=[df_csv["VL_LATITUDE"].mean(), df_csv["VL_LONGITUDE"].mean()],
        zoom_start=10
    )
except Exception as e:
    print(f"Erro ao criar o mapa: {e}")
    print("Valores de VL_LATITUDE:", df_csv["VL_LATITUDE"].describe())
    print("Valores de VL_LONGITUDE:", df_csv["VL_LONGITUDE"].describe())
    raise

# Adicionar o MarkerCluster ao mapa
marker_cluster = MarkerCluster().add_to(mapa)

# Estações meteorológicas
print("\n=== Depuração: Estações meteorológicas ===")
df_estacoes = df_csv[df_csv["DESC_TIPO_EQUIPAMENTO"].str.contains("ESTACAO", case=False, na=False)]
print(f"Número de estações meteorológicas: {len(df_estacoes)}")
if df_estacoes.empty:
    print("Aviso: Nenhuma estação meteorológica encontrada.")
else:
    for _, row in df_estacoes.iterrows():
        try:
            folium.Marker(
                location=[row["VL_LATITUDE"], row["VL_LONGITUDE"]],
                popup=f"{row['FROTA']}",
                icon=folium.Icon(color="blue", icon="cloud", prefix="fa")
            ).add_to(marker_cluster)
        except Exception as e:
            print(f"Erro ao adicionar marcador para estação: {row['FROTA']}, {row['VL_LATITUDE']}, {row['VL_LONGITUDE']}")
            print(f"Detalhes do erro: {e}")

# Pluviômetros ativos
print("\n=== Depuração: Pluviômetros ativos ===")
df_pluviometros_ativos = df_csv[
    (df_csv["DESC_TIPO_EQUIPAMENTO"].str.contains("PLUVIOMETRO", case=False, na=False)) &
    (df_csv["STATUS"].str.upper() == "ATIVO")
]
print(f"Número de pluviômetros ativos: {len(df_pluviometros_ativos)}")
if df_pluviometros_ativos.empty:
    print("Aviso: Nenhum pluviômetro ativo encontrado.")
else:
    for _, row in df_pluviometros_ativos.iterrows():
        try:
            folium.Marker(
                location=[row["VL_LATITUDE"], row["VL_LONGITUDE"]],  # Corrigido: osm_id -> VL_LATITUDE
                popup=f"{row['FROTA']}",
                icon=folium.Icon(color="green", icon="tint", prefix="fa")
            ).add_to(marker_cluster)
        except Exception as e:
            print(f"Erro ao adicionar marcador para pluviômetro: {row['FROTA']}, {row['VL_LATITUDE']}, {row['VL_LONGITUDE']}")
            print(f"Detalhes do erro: {e}")

# Adicionar limites (usando Name, conforme o primeiro bloco)
print("\n=== Depuração: Limites do KML ===")
if "Name" in gdf_kml.columns:
    print("Coluna 'Name' encontrada em gdf_kml. Adicionando limites com tooltip.")
    try:
        folium.GeoJson(
            gdf_kml,
            name="Limites",
            tooltip=folium.GeoJsonTooltip(fields=["Name"], aliases=["Fazenda:"]),
            style_function=lambda x: {"fillColor": "blue", "color": "blue", "weight": 1, "fillOpacity": 0.1}
        ).add_to(mapa)
    except Exception as e:
        print(f"Erro ao adicionar limites do KML: {e}")
else:
    print("Aviso: Coluna 'Name' não encontrada em gdf_kml. Adicionando limites sem tooltip.")
    try:
        folium.GeoJson(
            gdf_kml,
            name="Limites",
            style_function=lambda x: {"fillColor": "blue", "color": "blue", "weight": 1, "fillOpacity": 0.1}
        ).add_to(mapa)
    except Exception as e:
        print(f"Erro ao adicionar limites do KML: {e}")

# Controle de camadas
folium.LayerControl().add_to(mapa)

# Exibir o mapa
try:
    display(mapa)
except Exception as e:
    print(f"Erro ao exibir o mapa: {e}")
    raise

# @title Gráfico Interativo - Firmware por Unidade
import plotly.express as px

# Agrupar os dados
df_firmware_fazenda = df_csv.groupby(['VL_FIRMWARE_EQUIPAMENTO', 'UNIDADE']).size().reset_index(name='Quantidade')

# Ordenar por quantidade para visualização mais lógica
df_firmware_fazenda = df_firmware_fazenda.sort_values(by='Quantidade', ascending=True)

# Criar gráfico interativo
fig = px.bar(
    df_firmware_fazenda,
    x='Quantidade',
    y='UNIDADE',
    color='VL_FIRMWARE_EQUIPAMENTO',
    title='<b>Distribuição de Firmwares por Unidade</b>',
    labels={
        'VL_FIRMWARE_EQUIPAMENTO': 'Versão do Firmware',
        'UNIDADE': 'Unidade',
        'Quantidade': 'Qtd. de Equipamentos'
    },
    orientation='h',
    text='Quantidade',
    color_discrete_sequence=px.colors.qualitative.Dark2
)

# Personalizações visuais
fig.update_layout(
    title_font_size=20,
    title_font_family='Arial',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(250,250,250,1)',
    bargap=0.25,
    height=600,
    xaxis=dict(title='Quantidade de Equipamentos'),
    yaxis=dict(title=''),
    legend_title='Versão do Firmware'
)

# Melhoria nos rótulos de texto
fig.update_traces(
    textposition='outside',
    textfont=dict(size=12, color='black')
)

# Exibir
fig.show()

# @title Relação de pluviometros e tipo de comunicação
import plotly.express as px

# Definição da paleta de cores suave e elegante
cores_personalizadas = ["#2E86C1", "#28B463", "#8E44AD"]  # Azul, Verde, Roxo

# Prepare data for the first chart: Pluviômetros e Estações por Fazenda
df_contagem_1 = df_csv[
    df_csv["DESC_TIPO_EQUIPAMENTO"].str.contains("ESTACAO|PLUVIOMETRO", case=False, na=False)
].groupby(['UNIDADE', 'DESC_TIPO_EQUIPAMENTO']).size().reset_index(name='Quantidade')

# Prepare data for the second chart: Tipo de Comunicação por Unidade (excluding '4G')
df_contagem_2 = df_csv[df_csv['TIPO_COMUNICACAO'] != '4G'].groupby(['UNIDADE', 'TIPO_COMUNICACAO']).size().reset_index(name='Quantidade')

# Prepare data for the third chart: Percentual de Equipamentos com Dados Móveis
contagem_moveis = df_csv['D_MOVEIS_AT'].value_counts()


# --- GRÁFICO 1: Pluviômetros e Estações por Fazenda ---
fig1 = px.bar(
    df_contagem_1,
    x='UNIDADE',
    y='Quantidade',
    color='DESC_TIPO_EQUIPAMENTO',
    title='Pluviômetros e Estações por Unidade',
    text='Quantidade',
    barmode='stack',
    color_discrete_sequence=cores_personalizadas
)

fig1.update_layout(
    height=450,
    legend_title='Tipo de Equipamento',
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# --- GRÁFICO 2: Tipo de Comunicação por Unidade ---
fig2 = px.bar(
    df_contagem_2,
    x='UNIDADE',
    y='Quantidade',
    color='TIPO_COMUNICACAO',
    title='Tipos de Comunicação por Unidade (Excluindo 4G)',
    text='Quantidade',
    barmode='stack',
    color_discrete_sequence=cores_personalizadas
)

fig2.update_layout(
    height=450,
    legend_title='Tipo de Comunicação',
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# --- Gráfico de Rosca ---
fig3 = px.pie(
    values=contagem_moveis.values,
    names=contagem_moveis.index,
    title='Percentual de Equipamentos com Dados Móveis',
    hole=0.5,
    color_discrete_sequence=cores_personalizadas
)

fig3.update_traces(textinfo='percent+label', textfont_size=14)
fig3.update_layout(showlegend=True, legend_title='Dados Móveis')

# Exibir os gráficos
fig1.show()
fig2.show()
fig3.show()

#@title MAPA INTERATIVO SINAL

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from scipy.spatial.distance import cdist
import ipywidgets as widgets
from IPython.display import display, clear_output
from unidecode import unidecode
import gc

# --- Funções auxiliares ---

def formatar_nome(nome):
    if nome is None:
        return "N/A"
    return unidecode(str(nome)).strip().lower().replace(" ", "_")

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
    print("\n--- Iniciando interpolação IDW ---")
    print(f"Número de pontos para interpolar: {len(df)}")
    df = df.dropna(subset=[val_col]).copy()
    if df.empty:
        print("Erro: DataFrame vazio para interpolação!")
        return None, None, None, None

    # Se os valores já estão entre 1 e 4, não precisa classificar, só garantir que sejam inteiros
    if df[val_col].dropna().between(1,4).all():
        df['class_num'] = df[val_col].astype(int)
    else:
        df['class_num'] = df[val_col].apply(classificar_dbm)

    print(f"Valores únicos em class_num (classes para interpolação): {df['class_num'].unique()}")

    minx, miny = df[x_col].min() - buffer, df[y_col].min() - buffer
    maxx, maxy = df[x_col].max() + buffer, df[y_col].max() + buffer

    x_grid = np.arange(minx, maxx, resolution)
    y_grid = np.arange(miny, maxy, resolution)

    print(f"Grade criada com {len(x_grid)} x {len(y_grid)} pontos = {len(x_grid)*len(y_grid)} pontos totais")
    if len(x_grid)*len(y_grid) > 1_000_000:
        print("Aviso: Grade muito grande, ajuste resolução ou buffer.")
        return None, None, None, None

    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    pontos = df[[x_col, y_col]].values
    valores = df['class_num'].values

    print("Calculando distâncias para IDW...")
    distances = cdist(grid_points, pontos)
    epsilon = 1e-9
    weights = 1 / (distances**2 + epsilon)
    denom = weights.sum(axis=1)
    numer = (weights * valores).sum(axis=1)
    interpolated = numer / denom
    interpolated = np.clip(np.round(interpolated), 1, 4)

    # Aplicar máscara espacial, se disponível
    if geom_mask is not None:
        print("Aplicando máscara espacial (limite da fazenda)...")
        pontos_geom = [Point(xy) for xy in grid_points]
        mask = np.array([geom_mask.contains(pt) for pt in pontos_geom])
        interpolated[~mask] = np.nan

    grid_numerico = interpolated.reshape(grid_x.shape)
    print("Interpolação concluída.\n")

    del distances, weights, interpolated
    gc.collect()

    return grid_x, grid_y, grid_numerico, (minx, maxx, miny, maxy)

def estilo_ponto(row):
    mapa_frota_icones = {
        "pluviometro": ("blue", "o"),
        "estacao": ("purple", "o")
    }
    frota = str(row.get("DESC_TIPO_EQUIPAMENTO", "")).strip().lower()
    if frota in mapa_frota_icones:
        cor, marcador = mapa_frota_icones[frota]
    else:
        cor, marcador = ("red", "o")
    return cor, marcador

def plotar_interpolacao(grid_x, grid_y, grid_numerico, geom_fazenda, bounds, df_pontos):
    print("Iniciando plotagem da interpolação...")
    minx, maxx, miny, maxy = bounds

    colors = {
        1: '#fc8d59',  # ruim (vermelho claro)
        2: '#fee08b',  # regular (amarelo claro)
        3: '#91cf60',  # bom (verde claro)
        4: '#1a9850'   # ótimo (verde escuro)
    }
    cmap = plt.matplotlib.colors.ListedColormap([colors[i] for i in range(1, 5)])

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(grid_numerico,
                   extent=(minx, maxx, miny, maxy),
                   origin='lower',
                   cmap=cmap,
                   interpolation='nearest',
                   alpha=0.8)

    if geom_fazenda is not None and not geom_fazenda.is_empty:
        gpd.GeoSeries(geom_fazenda).boundary.plot(ax=ax, color='black', linewidth=2)

    legenda = {}

    for _, row in df_pontos.iterrows():
        cor, marcador = estilo_ponto(row)
        label = row.get("DESC_TIPO_EQUIPAMENTO", "N/A")
        if label not in legenda:
            legenda[label] = ax.scatter(
                row["VL_LONGITUDE"],
                row["VL_LATITUDE"],
                c=cor,
                marker=marcador,
                s=100,
                edgecolor="k",
                linewidth=0.7,
                label=label,
                alpha=0.9
            )
        else:
            ax.scatter(
                row["VL_LONGITUDE"],
                row["VL_LATITUDE"],
                c=cor,
                marker=marcador,
                s=100,
                edgecolor="k",
                linewidth=0.7,
                alpha=0.9
            )

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Interpolação IDW da Intensidade do Sinal com Equipamentos")

    cbar = plt.colorbar(im, ax=ax, ticks=[1.5, 2.5, 3.5, 4.5])
    cbar.ax.set_yticklabels(['Ruim', 'Regular', 'Bom', 'Ótimo'])
    cbar.set_label('Classe de Sinal')

    ax.legend(title="DESC_TIPO_EQUIPAMENTO", loc='upper right', markerscale=1.2)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    plt.pause(0.1)
    plt.close(fig)
    print("Plotagem finalizada.")
    gc.collect()

def gerar_interpolacao_por_fazenda(nome_fazenda):
    print(f"\n=== Iniciando processo para a fazenda: '{nome_fazenda}' ===")
    nome_formatado = formatar_nome(nome_fazenda)
    df_fazenda = gdf_equipamentos[gdf_equipamentos['UNIDADE_Padronizada'] == nome_formatado].copy()

    if df_fazenda.empty:
        print(f"Não foram encontrados pontos para a fazenda '{nome_fazenda}'. Abortando.")
        return

    print(f"Pontos encontrados na fazenda: {len(df_fazenda)}")

    geom_df = gdf_kml_com_nomes[gdf_kml_com_nomes['NomeFazendaKML_Padronizada'] == nome_formatado]
    if geom_df.empty:
        print(f"Geometria da fazenda '{nome_fazenda}' não encontrada. Abortando.")
        return
    fazenda_geom = geom_df.unary_union

    df_fazenda['DBM'] = pd.to_numeric(df_fazenda['DBM'], errors='coerce')
    has_dbm = 'DBM' in df_fazenda.columns and not df_fazenda['DBM'].dropna().empty
    has_intensidade = 'INTENSIDADE' in df_fazenda.columns and not df_fazenda['INTENSIDADE'].dropna().empty

    mapping = {
        "ruim": 1,
        "regular": 2,
        "bom": 3,
        "otimo": 4
    }

    if has_intensidade:
        print("Mapeando valores da coluna 'INTENSIDADE' para números...")
        df_fazenda['INTENSIDADE_MAP'] = df_fazenda['INTENSIDADE'].apply(
            lambda x: mapping.get(unidecode(str(x)).strip().lower(), np.nan)
        )
        print(f"Classes encontradas em INTENSIDADE_MAP: {df_fazenda['INTENSIDADE_MAP'].unique()}")

    if has_dbm and has_intensidade:
        print("Calculando médias de DBM para cada classe de INTENSIDADE_MAP (somente onde DBM está presente)...")
        df_com_dbm = df_fazenda.dropna(subset=['DBM'])
        medias_dbm = df_com_dbm.groupby('INTENSIDADE_MAP')['DBM'].mean()
        print(medias_dbm)

        def preencher_dbm(row):
            if pd.isna(row['DBM']) and not pd.isna(row.get('INTENSIDADE_MAP', np.nan)):
                return medias_dbm.get(row['INTENSIDADE_MAP'], np.nan)
            return row['DBM']

        antes = df_fazenda['DBM'].isna().sum()
        df_fazenda['DBM'] = df_fazenda.apply(preencher_dbm, axis=1)
        depois = df_fazenda['DBM'].isna().sum()
        print(f"Valores DBM nulos antes do preenchimento: {antes}")
        print(f"Valores DBM nulos após preenchimento: {depois}")

        val_col = 'DBM'

    elif has_dbm:
        print("Usando apenas valores reais de DBM (sem preenchimento)...")
        val_col = 'DBM'

    elif has_intensidade:
        print("Usando INTENSIDADE_MAP como proxy para DBM (não há DBM disponível)...")
        df_fazenda['DBM'] = df_fazenda['INTENSIDADE_MAP']
        val_col = 'DBM'

    else:
        print(f"Nenhum dado válido de intensidade para a fazenda '{nome_fazenda}'. Abortando.")
        return

    df_fazenda = df_fazenda.dropna(subset=[val_col])
    if df_fazenda.empty:
        print(f"Sem dados válidos para interpolação após preenchimento para a fazenda '{nome_fazenda}'. Abortando.")
        return

    print(f"Número de pontos válidos para interpolação: {len(df_fazenda)}")
    print(f"Descrição estatística dos valores usados para interpolação:")
    print(df_fazenda[val_col].describe())
    print("\n--- DEBUG: Verificando dados antes da interpolação ---")
    print(f"Número total de pontos na fazenda: {len(df_fazenda)}")
    print(f"Número de pontos sem NaN na coluna '{val_col}': {df_fazenda[val_col].dropna().shape[0]}")
    print(f"Valores únicos na coluna '{val_col}': {sorted(df_fazenda[val_col].dropna().unique())}")
    print("Amostra dos valores (primeiras 10 linhas):")
    print(df_fazenda[[val_col, 'INTENSIDADE_MAP', 'DBM']].head(10))

    grid_x, grid_y, grid_numerico, bounds = interpolacao_idw(
        df_fazenda,
        x_col='VL_LONGITUDE',
        y_col='VL_LATITUDE',
        val_col=val_col,
        resolution=0.002,
        buffer=0.05,
        geom_mask=fazenda_geom
    )

    if grid_x is None:
        print("Falha na interpolação. Abortando.")
        return

    plotar_interpolacao(grid_x, grid_y, grid_numerico, fazenda_geom, bounds, df_fazenda)

# --- BLOCO DE LEITURA E VALIDAÇÃO DOS DADOS ---

# Substitua pelos seus arquivos/dados carregados
# Exemplo:
# df_csv = pd.read_csv('dados_maquinas.csv')
# gdf_kml = gpd.read_file('fazendas.kml')

try:
    if 'df_csv' not in globals():
        raise NameError("df_csv não está definido. Execute o bloco de carregamento de dados.")
    if not isinstance(df_csv, pd.DataFrame):
        raise TypeError("df_csv deve ser um pandas DataFrame.")
    if 'gdf_kml' not in globals():
        raise NameError("gdf_kml não está definido. Execute o bloco de carregamento do GeoDataFrame.")
    if not isinstance(gdf_kml, gpd.GeoDataFrame):
        raise TypeError("gdf_kml deve ser um GeoDataFrame.")
except (NameError, TypeError) as e:
    print(f"Erro: {e}")
    raise

required_columns = ['VL_LATITUDE', 'VL_LONGITUDE', 'UNIDADE', 'DESC_TIPO_EQUIPAMENTO']
missing_columns = [col for col in required_columns if col not in df_csv.columns]
if missing_columns:
    raise ValueError(f"Colunas ausentes em df_csv: {missing_columns}")

print("\n=== Verificando coordenadas válidas ===")
print(f"Número de linhas antes da filtragem: {len(df_csv)}")
df_csv = df_csv.dropna(subset=['VL_LATITUDE', 'VL_LONGITUDE'])
df_csv = df_csv[(df_csv['VL_LATITUDE'].between(-90, 90)) & (df_csv['VL_LONGITUDE'].between(-180, 180))]
print(f"Número de linhas após filtragem: {len(df_csv)}")

if df_csv.empty:
    raise ValueError("Nenhuma coordenada válida após filtragem.")

gdf_equipamentos = gpd.GeoDataFrame(
    df_csv,
    geometry=gpd.points_from_xy(df_csv['VL_LONGITUDE'], df_csv['VL_LATITUDE']),
    crs="EPSG:4326"
)
gdf_equipamentos['UNIDADE_Padronizada'] = gdf_equipamentos['UNIDADE'].apply(formatar_nome)

gdf_kml_com_nomes = gdf_kml.copy()
gdf_kml_com_nomes['NomeFazendaExtraido'] = gdf_kml_com_nomes.get('NomeFazendaKML', gdf_kml_com_nomes.get('Name')).apply(
    lambda x: x.strip().lower() if isinstance(x, str) else "sem_nome"
)
gdf_kml_com_nomes['NomeFazendaKML_Padronizada'] = gdf_kml_com_nomes['NomeFazendaExtraido'].apply(formatar_nome)

gdf_kml_com_nomes['geometry'] = gdf_kml_com_nomes['geometry'].apply(
    lambda geom: geom.buffer(0) if geom and not geom.is_valid else geom
)

print("\n=== Número de geometrias inválidas após correção ===")
print(len(gdf_kml_com_nomes[~gdf_kml_com_nomes['geometry'].is_valid]))

# --- Interface interativa ---

def selecionar_unidade():
    unidades = sorted(gdf_equipamentos['UNIDADE_Padronizada'].dropna().unique())
    if not unidades:
        print("Erro: Nenhuma unidade válida encontrada.")
        return
    dropdown = widgets.Dropdown(options=unidades, description="Fazenda:")
    button = widgets.Button(description="Gerar Interpolação", button_style='success')
    output = widgets.Output()

    def on_click(b):
        with output:
            clear_output(wait=True)
            gerar_interpolacao_por_fazenda(dropdown.value)

    button.on_click(on_click)
    display(widgets.VBox([dropdown, button, output]))

# --- Rodar a interface ---
selecionar_unidade()

