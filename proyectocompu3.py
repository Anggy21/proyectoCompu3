import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree
import folium
import branca.colormap as cm
from shapely.geometry import shape
import webbrowser


data = pd.read_csv('dataset.csv', delimiter=',')
print(data)

print("\n**********")
"""# FASE 2: Entendimiento de los datos"""

data.info()

data.head()
data.tail()

print("\n**********")

data_numeric = data.select_dtypes(include=[np.number])
numeric_cols = data_numeric.columns.values

data_non_numeric = data.select_dtypes(exclude=[np.number])
non_numeric_cols = data_non_numeric.columns.values

print("Columnas numéricas:", numeric_cols)
print("Columnas categóricas:", non_numeric_cols)

print("\n**********")
print("Estadísticas para el TOTAL_VICTIMAS:")
print(data['TOTAL_VÍCTIMAS'].describe())

print("\n**********")

columnas_categoricas = data.select_dtypes(include=['object']).columns

estadisticas_categoricas = {}
for columna in columnas_categoricas:
    frecuencia = data[columna].value_counts()
    valores_unicos = data[columna].nunique()
    moda = data[columna].mode()[0]

    estadisticas_categoricas[columna] = {
        'Frecuencia absoluta': frecuencia,
        'Valores únicos': valores_unicos,
        'Moda': moda
    }

print("Estadísticas para las columnas categoricas:")
for columna, estadisticas in estadisticas_categoricas.items():
    print(f"\nEstadísticas para la columna: {columna}")
    print("Frecuencia absoluta:")
    print(estadisticas['Frecuencia absoluta'])
    print(f"\nValores únicos: {estadisticas['Valores únicos']}")
    print(f"Moda: {estadisticas['Moda']}")

print("\n**********")

"""# FASE 3. Preparación y limpieza de los datos"""

print("Recuento de valores nulos por columna:")
print(data.isnull().sum())

print("Filas duplicadas:")
duplicados = data.duplicated().sum()
print(f'Registros duplicados: {duplicados}')

print("\n**********")

moda_etapa = data[data['ETAPA_CASO'] != 'SIN DATO']['ETAPA_CASO'].mode()[0]

data['ETAPA_CASO'] = data['ETAPA_CASO'].replace('SIN DATO', moda_etapa)

print(data['ETAPA_CASO'].value_counts())

print("\n**********")

data = data.drop(columns=['PAÍS_HECHO'])
data = data.drop(columns=['AÑO_HECHOS', 'AÑO_ENTRADA', 'AÑO_DENUNCIA'], errors='ignore')


print(data['DEPARTAMENTO_HECHO'].unique())

print("\n**********")

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
print(data['DEPARTAMENTO_HECHO'].unique())

print("\n**********")

mapeo_seccionales = {
    'Cundinamarca': 'DIRECCIÓN SECCIONAL DE CUNDINAMARCA',
    'Bogotá': 'DIRECCIÓN SECCIONAL DE BOGOTÁ',
    'Boyacá': 'DIRECCIÓN SECCIONAL DE BOYACÁ',
    'Atlántico': 'DIRECCIÓN SECCIONAL DE ATLÁNTICO',
    'Antioquia': 'DIRECCIÓN SECCIONAL DE ANTIOQUIA',
    'Tolima': 'DIRECCIÓN SECCIONAL DE TOLIMA',
    'Valle del Cauca': 'DIRECCIÓN SECCIONAL DE VALLE DEL CAUCA',
    'Cesar': 'DIRECCIÓN SECCIONAL DE CESAR',
    'Córdoba': 'DIRECCIÓN SECCIONAL DE CÓRDOBA',
    'Huila': 'DIRECCIÓN SECCIONAL DE HUILA',
    'Nariño': 'DIRECCIÓN SECCIONAL DE NARIÑO',
    'Bolívar': 'DIRECCIÓN SECCIONAL DE BOLÍVAR',
    'Santander': 'DIRECCIÓN SECCIONAL DE SANTANDER',
    'La Guajira': 'DIRECCIÓN SECCIONAL DE LA GUAJIRA',
    'Meta': 'DIRECCIÓN SECCIONAL DE META',
    'Caquetá': 'DIRECCIÓN SECCIONAL DE CAQUETÁ',
    'Magdalena': 'DIRECCIÓN SECCIONAL DE MAGDALENA',
    'Chocó': 'DIRECCIÓN SECCIONAL DE CHOCÓ',
    'Caldas': 'DIRECCIÓN SECCIONAL DE CALDAS',
    'Norte de Santander': 'DIRECCIÓN SECCIONAL DE NORTE DE SANTANDER',
    'Risaralda': 'DIRECCIÓN SECCIONAL DE RISARALDA',
    'Putumayo': 'DIRECCIÓN SECCIONAL DE PUTUMAYO',
    'Cauca': 'DIRECCIÓN SECCIONAL DE CAUCA',
    'Vaupés': 'DIRECCIÓN SECCIONAL DE VAUPÉS',
    'Casanare': 'DIRECCIÓN SECCIONAL DE CASANARE',
    'Guainía': 'DIRECCIÓN SECCIONAL DE GUAINÍA',
    'San Andrés y Providencia': 'DIRECCIÓN SECCIONAL DE SAN ANDRÉS Y PROVIDENCIA',
    'Arauca': 'DIRECCIÓN SECCIONAL DE ARAUCA',
    'Sucre': 'DIRECCIÓN SECCIONAL DE SUCRE',
    'Quindío': 'DIRECCIÓN SECCIONAL DE QUINDÍO',
    'Vichada': 'DIRECCIÓN SECCIONAL DE VICHADA',
    'Guaviare': 'DIRECCIÓN SECCIONAL DE GUAViare',
    'Amazonas': 'DIRECCIÓN SECCIONAL DE AMAZONAS'
}

data['SECCIONAL'] = data['SECCIONAL'].replace('SIN DATO', np.nan)

data['SECCIONAL'] = data.apply(
    lambda row: mapeo_seccionales[row['DEPARTAMENTO_HECHO']]
    if pd.isna(row['SECCIONAL']) and row['DEPARTAMENTO_HECHO'] in mapeo_seccionales
    else row['SECCIONAL'], axis=1
)

print(data['SECCIONAL'].value_counts())

print("\n**********")

moda_sexo = data[data['SEXO'] != 'SIN DATO']['SEXO'].mode()[0]

data['SEXO'] = data['SEXO'].replace('SIN DATO', moda_sexo)

print("Distribución después de reemplazar 'SIN DATO':")
print(data['SEXO'].value_counts())

print("\n**********")

moda_grupo_etario = data[data['GRUPO_ETARIO'] != 'SIN DATO']['GRUPO_ETARIO'].mode()[0]

data['GRUPO_ETARIO'] = data['GRUPO_ETARIO'].replace('SIN DATO', moda_grupo_etario)


"""# FASE 4: Analizando los datos / Modelado"""

mapeo_departamentos= {
    'CHOCÓ': 'CHOCO',
    'BOGOTÁ': 'SANTAFE DE BOGOTA D.C',
    'BOYACÁ': 'BOYACA',
    'CÓRDOBA': 'CORDOBA',
    'BOLÍVAR': 'BOLIVAR',
    'CAQUETÁ': 'CAQUETA', 
    'GUAINÍA': 'GUAINIA', 
    'ATLÁNTICO': 'ATLANTICO', 
    'VAUPÉS': 'VAUPES', 
    'QUINDÍO': 'QUINDIO', 
    'SAN ANDRÉS Y PROVIDENCIA':'ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA'
}

data['DEPARTAMENTO_HECHO'] = data['DEPARTAMENTO_HECHO'].replace(mapeo_departamentos)

# Calcular el conteo y el porcentaje de delitos por grupo y departamento
conteo_por_grupo = data.groupby(['DEPARTAMENTO_HECHO', 'GRUPO_DELITO']).size().reset_index(name='conteo')
total_por_departamento = data.groupby('DEPARTAMENTO_HECHO').size().reset_index(name='total_delitos')
conteo_por_grupo = conteo_por_grupo.merge(total_por_departamento, on='DEPARTAMENTO_HECHO')
conteo_por_grupo['porcentaje'] = (conteo_por_grupo['conteo'] / conteo_por_grupo['total_delitos']) * 100

# Ruta del archivo GeoJSON de los departamentos de Colombia
geojson_url = "Colombia.geo.json"

# Crear un diccionario con los datos de cada departamento y grupo de delito
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

# Crear un mapa base centrado en Colombia
def generar_mapa():
    mapa = folium.Map(location=[4.570868, -74.297333], zoom_start=6)

    # Colores únicos para cada grupo de delito
    grupos_delitos = data['GRUPO_DELITO'].unique()
    color_map = cm.LinearColormap(['red', 'blue', 'green', 'purple', 'orange'], vmin=0, vmax=len(grupos_delitos)-1)
    color_dict = {grupo: color_map(i) for i, grupo in enumerate(grupos_delitos)}

    # Función para asignar color a cada departamento basado en el primer grupo de delito
    def estilo_departamento(feature):
        dep = feature['properties']['NOMBRE_DPT']  # Asegúrate de que 'NOMBRE_DPT' es el campo correcto
        if dep in data_por_departamento:
            # Usar el color del primer grupo de delito encontrado
            primer_delito = next(iter(data_por_departamento[dep].keys()))
            return {
                'fillColor': color_dict[primer_delito],
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7
            }
        return {'fillColor': 'white', 'color': 'black', 'weight': 1, 'fillOpacity': 0.3}

    # Añadir al mapa los departamentos como GeoJSON
    geojson_layer = folium.GeoJson(
        geojson_url,
        name='Distribución de delitos',
        style_function=estilo_departamento,
        highlight_function=lambda x: {'weight': 3, 'color': 'yellow'},
        tooltip=folium.GeoJsonTooltip(
            fields=['NOMBRE_DPT'],  # Asegúrate de que 'NOMBRE_DPT' es el campo correcto
            aliases=['Departamento:'],
            sticky=True
        )
    ).add_to(mapa)

    # Extraer los centroides para los popups
    for feature in geojson_layer.data['features']:
        dep = feature['properties']['NOMBRE_DPT']  # Asegúrate de que 'NOMBRE_DPT' es el campo correcto
        if dep in data_por_departamento:
            # Calcular el centroide del departamento
            geometria = shape(feature['geometry'])
            centroide = geometria.centroid
            popup_html = f"<b>Departamento:</b> {dep}<br><br>"
            for delito, stats in data_por_departamento[dep].items():
                popup_html += f"<b>{delito}:</b> {stats['conteo']} ({stats['porcentaje']:.2f}%)<br>"
            
            # Reemplazar caracteres especiales
            popup_html = popup_html.replace("Ã±", "ñ").replace("Ã“", "Ó")

            # Agregar popup al mapa
            folium.Marker(
                location=[centroide.y, centroide.x],
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(mapa)

    # Guardar el mapa como archivo HTML y abrirlo en el navegador
    mapa.save("mapa_grupos_delito.html")
    webbrowser.open("mapa_grupos_delito.html")  # Esto abre el archivo en el navegador automáticamente

    print("El mapa ha sido generado y abierto en el navegador.")
# Funciones de visualización
def plot_victimas_sexo_grupo_delito():
    plt.figure(figsize=(12, 6))
    sns.countplot(x='GRUPO_DELITO', hue='SEXO', data=data)
    plt.title('Distribución de Víctimas por Sexo y Grupo de Delito')
    plt.xlabel('Grupo de Delito')
    plt.ylabel('Número de Víctimas')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Sexo')
    plt.tight_layout()
    plt.show()

def plot_victimas_grupo_etario():
    plt.figure(figsize=(12, 8))
    sns.countplot(x='GRUPO_ETARIO', hue='GRUPO_DELITO', data=data)
    plt.title('Distribución de Víctimas por Grupo Etario y Grupo de Delito')
    plt.xlabel('Grupo Etario')
    plt.ylabel('Número de Víctimas')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Grupo de Delito')
    plt.tight_layout()
    plt.show()

def plot_distribucion_sexo():
    distribucion_sexo = data['SEXO'].value_counts(normalize=True) * 100
    print("Distribución porcentual de víctimas por sexo:")
    print(distribucion_sexo)
    plt.figure(figsize=(8, 8))
    plt.pie(distribucion_sexo, labels=distribucion_sexo.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribución Porcentual de Víctimas por Sexo')
    plt.show()

def plot_distribucion_grupo_delito():
    distribucion_grupo_delito = data['GRUPO_DELITO'].value_counts(normalize=True) * 100
    print("Distribución porcentual de víctimas por grupo delito:")
    print(distribucion_grupo_delito)
    plt.figure(figsize=(8, 8))
    plt.pie(distribucion_grupo_delito, labels=distribucion_grupo_delito.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribución Porcentual de Víctimas por Grupo Delito')
    plt.show()

def plot_delitos_departamento():
    plt.figure(figsize=(12, 6))
    sns.countplot(x='DEPARTAMENTO_HECHO', hue='GRUPO_DELITO', data=data)
    plt.title('Distribución de Grupos de Delito por Departamento')
    plt.xlabel('Departamento')
    plt.ylabel('Número de Delitos')
    plt.xticks(rotation=90, ha='right')
    plt.legend(title='Grupo de Delito')
    plt.tight_layout()
    plt.show()

# Menú interactivo
def menu():
    while True:
        print("\nSeleccione la opción:")
        print("1. Distribución de Víctimas por Sexo y Grupo de Delito")
        print("2. Distribución de Víctimas por Grupo Etario y Grupo de Delito")
        print("3. Distribución Porcentual de Víctimas por Sexo")
        print("4. Distribución Porcentual de Víctimas por Grupo Delito")
        print("5. Distribución de Grupos de Delito por Departamento")
        print("6. Generar Mapa de Distribución de Delitos")
        print("7. Salir")

        opcion = input("Ingrese el número de la opción: ")

        if opcion == '1':
            plot_victimas_sexo_grupo_delito()
        elif opcion == '2':
            plot_victimas_grupo_etario()
        elif opcion == '3':
            plot_distribucion_sexo()
        elif opcion == '4':
            plot_distribucion_grupo_delito()
        elif opcion == '5':
            plot_delitos_departamento()
        elif opcion == '6':
            generar_mapa()
        elif opcion == '7':
            print("Saliendo del programa...")
            break
        else:
            print("Opción inválida, intente de nuevo.")


menu()

print("\n**********")

"""Modelado"""

variables = ['SEXO', 'GRUPO_ETARIO', 'DEPARTAMENTO_HECHO','PAÍS_NACIMIENTO','APLICA_LGBTI','INDÍGENA','AFRODESCENDIENTE','TOTAL_VÍCTIMAS']
data_model = data[variables]
data_model = pd.get_dummies(data_model)
X = data_model
y = data['GRUPO_DELITO']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar el modelo de árbol de decisión
model = DecisionTreeClassifier(random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

""""Fase 5: Evaluación del modelo"""""
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitud del modelo: {accuracy * 100:.2f}%")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

# Crear un heatmap de la matriz de confusión
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.show()

plt.figure(figsize=(15,10))
tree.plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
""""plt.show()"""""

