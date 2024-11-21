import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import folium
import branca.colormap as cm
import json
from folium.features import GeoJsonPopup, GeoJsonTooltip

data = pd.read_csv('dataset.csv', delimiter=',')

def normalizar_nombre(nombre):
    if isinstance(nombre, str):
        return nombre.strip().lower().replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
    return ""

data['ETAPA_CASO'] = data['ETAPA_CASO'].replace('SIN DATO', data[data['ETAPA_CASO'] != 'SIN DATO']['ETAPA_CASO'].mode()[0])
data['DEPARTAMENTO_HECHO'] = data['DEPARTAMENTO_HECHO'].replace({
    'CHOCÓ': 'CHOCO',
    'BOGOTÁ, D. C.': 'SANTAFE DE BOGOTA D.C',
    'BOYACÁ': 'BOYACA',
    'CÓRDOBA': 'CORDOBA',
    'BOLÍVAR': 'BOLIVAR',
    'CAQUETÁ': 'CAQUETA', 
    'GUAINÍA': 'GUAINIA', 
    'ATLÁNTICO': 'ATLANTICO', 
    'VAUPÉS': 'VAUPES', 
    'QUINDÍO': 'QUINDIO', 
    'Archipiélago de San Andrés, Providencia y Santa Catalina':'ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA'

})
data['SEXO'] = data['SEXO'].replace('SIN DATO', data[data['SEXO'] != 'SIN DATO']['SEXO'].mode()[0])
data['GRUPO_ETARIO'] = data['GRUPO_ETARIO'].replace('SIN DATO', data[data['GRUPO_ETARIO'] != 'SIN DATO']['GRUPO_ETARIO'].mode()[0])

data = data.drop(columns=['PAÍS_HECHO'])
data = data.drop(columns=['AÑO_HECHOS', 'AÑO_ENTRADA', 'AÑO_DENUNCIA'], errors='ignore')



# Variables para el modelo
variables = ['SEXO', 'GRUPO_ETARIO', 'DEPARTAMENTO_HECHO', 'PAÍS_NACIMIENTO', 'APLICA_LGBTI', 'INDÍGENA', 'AFRODESCENDIENTE', 'TOTAL_VÍCTIMAS']
data_model = data[variables]
data_model = pd.get_dummies(data_model)
X = data_model
y = data['GRUPO_DELITO']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelo
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluaciones del modelo
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

data['DEPARTAMENTO_HECHO'] = data['DEPARTAMENTO_HECHO'].apply(normalizar_nombre)

# Calcular el conteo y el porcentaje de delitos por grupo y departamento
conteo_por_grupo = data.groupby(['DEPARTAMENTO_HECHO', 'GRUPO_DELITO']).size().reset_index(name='conteo')
total_por_departamento = data.groupby('DEPARTAMENTO_HECHO').size().reset_index(name='total_delitos')

# Calcular el porcentaje
conteo_por_grupo = conteo_por_grupo.merge(total_por_departamento, on='DEPARTAMENTO_HECHO')
conteo_por_grupo['porcentaje'] = (conteo_por_grupo['conteo'] / conteo_por_grupo['total_delitos']) * 100

max_delitos = total_por_departamento['total_delitos'].max()
departamentos_max_delitos = total_por_departamento[
    total_por_departamento['total_delitos'] == max_delitos
]['DEPARTAMENTO_HECHO'].tolist()


# Crear un diccionario con los datos por departamento
data_por_departamento = {}
for _, row in conteo_por_grupo.iterrows():
    dep = row['DEPARTAMENTO_HECHO']
    delito = row['GRUPO_DELITO']
    if dep not in data_por_departamento:
        data_por_departamento[dep] = {}
    data_por_departamento[dep][delito] = {
        'conteo': row['conteo'],
        'porcentaje': row['porcentaje']
    }

# Leer el archivo GeoJSON
with open('Colombia.geo.json', encoding='utf-8') as f:
    geojson_data = json.load(f)

# Crear el mapa base
mapa = folium.Map(location=[4.570868, -74.297333], zoom_start=6)

# Función para asignar estilo a los departamentos
def estilo_departamento(feature):
    dep_geojson = normalizar_nombre(feature['properties'].get('NOMBRE_DPT', ''))
    if dep_geojson in departamentos_max_delitos:
        return {
            'fillColor': 'red',  # Color distintivo para los departamentos con más delitos
            'color': 'black',
            'weight': 2,
            'fillOpacity': 0.9,
        }
    return {
        'fillColor': '#74a9cf',  # Color estándar para otros departamentos
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.7,
    }

# Agregar los datos GeoJSON al mapa
geojson_layer = folium.GeoJson(
    geojson_data,
    name="Departamentos",
    style_function=estilo_departamento,
    tooltip=GeoJsonTooltip(
        fields=['NOMBRE_DPT'],
        aliases=['Departamento:'],
        localize=True
    )
)
geojson_layer.add_to(mapa)

# Normalizar los nombres de los departamentos en el GeoJSON y agregar marcadores
for feature in geojson_data['features']:
    dep_geojson = normalizar_nombre(feature['properties'].get('NOMBRE_DPT', ''))
    # Obtener la ubicación central del departamento (coordenadas aproximadas)
    geometry = feature['geometry']
    if geometry['type'] == 'Polygon':
        coordinates = geometry['coordinates'][0]
    elif geometry['type'] == 'MultiPolygon':
        coordinates = geometry['coordinates'][0][0]
    else:
        coordinates = []

    # Calcular el centro aproximado
    if coordinates:
        lat = sum(coord[1] for coord in coordinates) / len(coordinates)
        lon = sum(coord[0] for coord in coordinates) / len(coordinates)
    else:
        lat, lon = None, None

    # Preparar la información para el popup
    delitos_info = ""
    if dep_geojson in data_por_departamento:
        for delito, stats in data_por_departamento[dep_geojson].items():
            delitos_info += f"<b>{delito}:</b> {stats['conteo']} ({stats['porcentaje']:.2f}%)<br>"
    else:
        delitos_info = "No hay información de delitos."

    # Crear un marcador si las coordenadas son válidas
    if lat and lon:
        popup_content = folium.Popup(delitos_info, max_width=300)
        folium.Marker(
            location=[lat, lon],
            popup=popup_content,
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(mapa)

# Agregar control de capas
folium.LayerControl().add_to(mapa)

# Guardar el mapa en un archivo HTML
mapa.save('mapa_grupos_delito.html')
print("Mapa generado y guardado como 'mapa_grupos_delito.html'")

# Configuración de la aplicación Streamlit
st.title("Análisis de Datos: Conteo de víctimas en Colombia en 2023")
st.sidebar.title("Navegación")
menu = st.sidebar.radio("Selecciona una opción", ["Exploración de Datos", "Visualización", "Modelo de Clasificación", "Conclusiones"])

if menu == "Exploración de Datos":
    st.header("Exploración de Datos")
    
    # Mostrar primeras filas del DataFrame
    st.write("### Primeras filas del dataset:")
    st.dataframe(data.head())

    # Información general del DataFrame
    st.write("### Información general del dataset:")
    # Capturamos la información de las columnas, tipos de datos y valores nulos
    info_df = pd.DataFrame({
        "Columnas": data.columns,
        "Tipo de dato": data.dtypes,
        "Valores nulos": data.isnull().sum()
    })
    st.dataframe(info_df)

    # Estadísticas descriptivas de TOTAL_VÍCTIMAS
    estadisticas_clave = data['TOTAL_VÍCTIMAS'].agg(['count', 'mean', 'std', 'min', 'max'])
    st.write("### Estadísticas descriptivas para TOTAL_VÍCTIMAS:")
    st.write(estadisticas_clave)

    # diagrama de caja de outliers
    st.write("### Gráfica de Outliers:")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data['TOTAL_VÍCTIMAS'], vert=False, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            flierprops=dict(marker='o', color='red', markersize=5))
    ax.set_title('Identificación de Outliers en TOTAL_VÍCTIMAS')
    ax.set_xlabel('TOTAL_VÍCTIMAS')
    st.pyplot(fig)

 

    # Estadísticas para columnas categóricas
    st.write("### Estadísticas para columnas categóricas:")
    columnas_categoricas = data.select_dtypes(include=['object']).columns
    estadisticas_categoricas = []

    for columna in columnas_categoricas:
        valores_unicos = data[columna].nunique()
        moda = data[columna].mode()[0]
        estadisticas_categoricas.append({
            "Columna": columna,
            "Valores únicos": valores_unicos,
            "Moda": moda,
        })

    estadisticas_categoricas_df = pd.DataFrame(estadisticas_categoricas)
    st.dataframe(estadisticas_categoricas_df)


elif menu == "Visualización":
    st.header("Visualización de Datos")
    grafica = st.selectbox("Selecciona la gráfica que deseas visualizar", [
        "Distribución de Víctimas por Sexo y Grupo de Delito",
        "Distribución de Víctimas por Grupo Etario y Grupo de Delito",
        "Distribución Porcentual de Víctimas por Sexo",
        "Distribución Porcentual de Víctimas por Grupo Delito",
        "Distribución de Grupos de Delito por Departamento",
        "Distribución de Víctimas por Grupo Delito y Categoría",
        "Mapa de Distribución de Delitos"
    ])

    if grafica == "Mapa de Distribución de Delitos":
        st.markdown("### Mapa de Distribución de Delitos en Colombia")
        st.components.v1.html(open("mapa_grupos_delito.html", 'r').read(), height=600)

    elif grafica == "Distribución de Víctimas por Sexo y Grupo de Delito":
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(x='GRUPO_DELITO', hue='SEXO', data=data, ax=ax)
        ax.set_title('Distribución de Víctimas por Sexo y Grupo de Delito')
        plt.xlabel('Grupo de Delito')
        plt.ylabel('Número de Víctimas')
        st.pyplot(fig)

    elif grafica == "Distribución de Víctimas por Grupo Etario y Grupo de Delito":
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.countplot(x='GRUPO_ETARIO', hue='GRUPO_DELITO', data=data, ax=ax)
        ax.set_title('Distribución de Víctimas por Grupo Etario y Grupo de Delito')
        plt.xlabel('Grupo Etario')
        plt.ylabel('Número de Víctimas')
        st.pyplot(fig)

    elif grafica == "Distribución Porcentual de Víctimas por Sexo":
        distribucion_sexo = data['SEXO'].value_counts(normalize=True) * 100
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(distribucion_sexo, labels=distribucion_sexo.index, autopct='%1.1f%%', startangle=90)
        ax.set_title('Distribución Porcentual de Víctimas por Sexo')
        st.pyplot(fig)

    elif grafica == "Distribución Porcentual de Víctimas por Grupo Delito":
        distribucion_grupo_delito = data['GRUPO_DELITO'].value_counts(normalize=True) * 100
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(distribucion_grupo_delito, labels=distribucion_grupo_delito.index, autopct='%1.1f%%', startangle=90)
        ax.set_title('Distribución Porcentual de Víctimas por Grupo Delito')
        st.pyplot(fig)

    elif grafica == "Distribución de Grupos de Delito por Departamento":
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(x='DEPARTAMENTO_HECHO', hue='GRUPO_DELITO', data=data, ax=ax)
        ax.set_title('Distribución de Grupos de Delito por Departamento')
        plt.xlabel('Departamento')
        plt.ylabel('Número de Delitos')
        plt.xticks(rotation=90)
        st.pyplot(fig)

    elif grafica == "Distribución de Víctimas por Grupo Delito y Categoría":
        filtro = data[['GRUPO_DELITO', 'TOTAL_VÍCTIMAS', 'APLICA_LGBTI', 'INDÍGENA', 'AFRODESCENDIENTE']]

        totales = filtro.groupby('GRUPO_DELITO')[['TOTAL_VÍCTIMAS']].sum()
        totales['LGBTI'] = filtro[filtro['APLICA_LGBTI'] == 'SI'].groupby('GRUPO_DELITO')['TOTAL_VÍCTIMAS'].sum()
        totales['INDÍGENA'] = filtro[filtro['INDÍGENA'] == 'SI'].groupby('GRUPO_DELITO')['TOTAL_VÍCTIMAS'].sum()
        totales['AFRODESCENDIENTE'] = filtro[filtro['AFRODESCENDIENTE'] == 'SI'].groupby('GRUPO_DELITO')['TOTAL_VÍCTIMAS'].sum()

        fig, ax = plt.subplots(figsize=(12, 8))
        totales[['LGBTI', 'INDÍGENA', 'AFRODESCENDIENTE']].plot(kind='bar', ax=ax, stacked=True)


        ax.set_title('Total de Víctimas por Grupo de Delito y Categoría')
        ax.set_xlabel('Grupo de Delito')
        ax.set_ylabel('Total de Víctimas')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        st.pyplot(fig)


elif menu == "Modelo de Clasificación":
    st.header("Resultados del Modelo de Clasificación: Árbol de Decisión")
    st.write(f"### Exactitud del modelo: {accuracy * 100:.2f}%")
    st.write("### Reporte de Clasificación:")
    st.write(pd.DataFrame(classification_rep).transpose())
    st.write("### Matriz de Confusión:")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    ax.set_title('Matriz de Confusión')
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Valor Real')
    st.pyplot(fig)

elif menu == "Conclusiones":
    st.header("Conclusiones")
    st.write("""
     1.	La mayoría de las víctimas son mujeres, lo que confirma que los delitos asociados a violencias basadas en género afectan desproporcionadamente a este grupo. Se identificó que los adultos (de 27 a 59 años) representan los grupos etarios con mayor incidencia en la mayoría de los delitos.
             
    2. La mayoría de los delitos analizados tienen como principal población afectada a las mujeres, especialmente en casos de violencia intrafamiliar y abuso sexual. Los delitos de lesiones personales y violencia intrafamiliar muestran una mayor incidencia entre personas identificadas como LGBTI
             
    3.	Antioquia, Bogotá, Cundinamarca y Valle del Cauca fueron los territorios con mayor número de reportes
             
    4. En regiones como el Caribe y el Pacífico, las tradiciones culturales que normalizan la violencia de género contribuyen a su alta incidencia.
    """)