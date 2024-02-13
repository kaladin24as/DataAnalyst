import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

def limpiar_datos(df, threshold, umbral_significativo):
    """
    Limpia los datos de un DataFrame eliminando columnas con valores nulos que superan un umbral
    y encuentra correlaciones significativas entre las columnas restantes.

    Parámetros:
    df (DataFrame): El DataFrame que contiene los datos.
    threshold (float): El umbral de valores nulos para descartar columnas.
    umbral_significativo (float): El umbral de correlación significativa. Por defecto es 0.3.

    Retorna:
    DataFrame: Un DataFrame con las columnas que no superan el umbral de valores nulos.
    list: Una lista de columnas eliminadas.
    list: Una lista de tuplas que contienen las correlaciones significativas.
    """
    null_count = df.isnull().sum()
    null_percentage = (null_count / len(df)) 
    columns_to_drop = null_percentage[null_percentage > threshold].index
    columns_dropped = list(columns_to_drop)
    df = df.drop(columns=columns_to_drop)
    
    # Rellenar espacios en blanco con el promedio de cada columna
    if len(columns_dropped) == 0:
        df = df.fillna(df.mean())
    
    columnas_numericas = df.select_dtypes(include=['number'])
    correlacion = columnas_numericas.corr()
    correlaciones_significativas = correlacion[(correlacion >= umbral_significativo) | (correlacion <= -umbral_significativo)]
    correlaciones_list = []
    for col1 in correlaciones_significativas.columns:
        for col2 in correlaciones_significativas.index:
            if col1 != col2:
                valor_correlacion = correlaciones_significativas.loc[col2, col1]
                if abs(valor_correlacion) >= umbral_significativo:
                    correlaciones_list.append((col1, col2, valor_correlacion))
    return df, columns_dropped, correlaciones_list

def classify(model):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(x_train, y_train)
    print('Accuracy:', model.score(x_test, y_test))
    score = cross_val_score(model, X, y, cv=5)
    print('CV Score:', np.mean(score))

def convertir_columna_a_numerica(df, column_name):
    """
    Convierte una columna categórica en valores numéricos utilizando LabelEncoder.

    Parámetros:
    df (DataFrame): El DataFrame que contiene los datos.
    column_name (str): El nombre de la columna que se desea convertir.

    Retorna:
    DataFrame: El DataFrame con la columna convertida a valores numéricos si la columna existe.
    None: Si la columna especificada no existe en los datos.
    """
    if column_name in df.columns:
        le = LabelEncoder()
        df[column_name] = le.fit_transform(df[column_name])
        return df
    else:
        print("La columna especificada no existe en los datos.")
        return None
    
def identificar_variables_objetivo(df, num_variables_objetivo, threshold):
    """
    Identifica las variables objetivo en un DataFrame encontrando las columnas con la mayor correlación con otras columnas.

    Parámetros:
    df (DataFrame): El DataFrame que contiene los datos.
    num_variables_objetivo (int): El número de variables objetivo que se desean identificar.
    threshold (float): El umbral de correlación mínima para considerar una columna como una variable objetivo.

    Retorna:
    list: Una lista de los nombres de las variables objetivo identificadas.
    """
    # Excluir columnas no numéricas
    columnas_numericas = df.select_dtypes(include=['number'])

    # Calcular la matriz de correlación
    correlacion = columnas_numericas.corr()

    # Encontrar las columnas con la mayor correlación con otras columnas
    correlaciones_maximas = correlacion.abs().max()
    variables_objetivo = correlaciones_maximas[correlaciones_maximas > threshold].nlargest(num_variables_objetivo).index.tolist()

    return variables_objetivo
    
    

models = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(),
          ExtraTreesClassifier(), XGBClassifier()]

file_name = input("Ingresa el nombre del archivo de entrenamiento de datos:")
test_name = input("Ingresa el nombre del archivo de prueba de datos:")

data_cleaned = pd.read_csv(file_name)

print("Datos originales:")
print(data_cleaned.head())

columns_to_convert = input("¿Deseas convertir alguna columna categórica en valores numéricos? (yes/no): ")

if columns_to_convert.lower() == 'yes':
    column_name = input("Ingresa el nombre de la columna que deseas convertir: ")
    data_cleaned = convertir_columna_a_numerica(data_cleaned, column_name)

data_cleaned, columns_dropped, correlaciones_significativas = limpiar_datos(data_cleaned, threshold=0.5, umbral_significativo=0.3)

print("Datos limpios después de la conversión y la limpieza:")
print(data_cleaned.head())

print("\nColumnas eliminadas:")
print(columns_dropped)

print("\nCorrelaciones significativas:")
for correlacion in correlaciones_significativas:
    print(correlacion)

print(data_cleaned.info())
# Solicitar al usuario el nombre de la variable objetivo
target_variable = input("Ingresa el nombre de la variable objetivo: ")

# Solicitar al usuario los nombres de las variables predictoras
predictor_variables = input("Ingresa los nombres de las variables predictoras separadas por coma: ")
predictor_variables = [var.strip() for var in predictor_variables.split(",")]

# Crear la matriz de características X y el vector objetivo y
X = data_cleaned[predictor_variables]
y = data_cleaned[target_variable]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Entrenar modelos predictivos
for model in models:
    print("\nEntrenando modelo:", model.__class__.__name__)
    model.fit(X_train, y_train)
    print('Accuracy en el conjunto de prueba:', model.score(X_test, y_test))
    score = cross_val_score(model, X, y, cv=5)
    print('CV Score:', np.mean(score))

# Cargar el conjunto de datos de prueba
test_data = pd.read_csv(test_name)

# Realizar el mismo preprocesamiento que hiciste en los datos de entrenamiento
test_data_cleaned = test_data.copy()  # Hacer una copia de los datos de prueba

# Convertir columnas categóricas a numéricas
test_data_cleaned = convertir_columna_a_numerica(test_data_cleaned, 'Sex')  # Convertir 'Sex' a numérico u otra columna si es necesario

# Aplicar el mismo preprocesamiento adicional que hiciste en los datos de entrenamiento (limpieza, eliminación de columnas, etc.)

# Guardar los IDs de los pasajeros para el DataFrame de resultados
test_ids = test_data_cleaned['PassengerId']

# Realizar predicciones en el conjunto de datos de prueba
predictions = model.predict(test_data_cleaned[predictor_variables])

# Crear un DataFrame con los resultados
results_df = pd.DataFrame({'PassengerId': test_ids, 'Survived': predictions})

# Guardar el DataFrame de resultados en un archivo CSV
results_df.to_csv('results.csv', index=False)

# Imprimir una muestra de los resultados
print(results_df.head())

