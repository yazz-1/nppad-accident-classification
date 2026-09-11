import pandas as pd
import os
from numpy import polyfit
from src.utils import limpiar_nombre_archivo
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# LISTAS PREDEFINIDAS DE SENSORES Y UNIDADES
# =============================================================================

COLUMNAS_12 = ['TIME', 'P', 'TAVG', 'LVPZ', 'WBK', 'LSGA', 'LSGB', 'PSGA', 'WSTA', 'QMWT', 'PWNT', 'LWRB', 'TPCT']
UNIDADES_12 = ['s', 'bar', '°C', '%', 't/h', 'm', 'm', 'bar', 't/h', 'MW', '%', 'm', '°C']
COLUMNAS_5 = COLUMNAS_12[:6]
UNIDADES_5 = UNIDADES_12[:6]
sensores = COLUMNAS_12[1:]

def seleccionar_columnas(df_list, columnas):
    """
    Filtra cada DataFrame de la lista para quedarse solo con las columnas indicadas.

    Args:
        df_list (list[pd.DataFrame]): Lista de DataFrames.
        columnas (list[str]): Lista de nombres de columnas a conservar.

    Returns:
        list[pd.DataFrame]: Nueva lista de DataFrames filtrados y ordenados
            por 'TIME' si la columna existe.
    """
    df_list_filtrada = []
    for df in df_list:
        columnas_existentes = [col for col in columnas if col in df.columns]
        df_filtrado = df[columnas_existentes].copy()
        if 'TIME' in df_filtrado.columns:
            df_filtrado = df_filtrado.sort_values('TIME')
        df_list_filtrada.append(df_filtrado)
    return df_list_filtrada

def extraer_estadisticas(df, sensores):
    """
    Extrae estadísticas resumen de cada sensor de un DataFrame.

    Para cada sensor, calcula: media, desviación estándar, máximo, mínimo,
    valor inicial, valor final, pendiente de regresión lineal y tiempos
    hasta el máximo y el mínimo.

    Args:
        df (pd.DataFrame): DataFrame con columna 'TIME' y los sensores.
        sensores (list[str]): Lista de nombres de sensores a procesar.

    Returns:
        dict: Diccionario con claves del tipo '{sensor}_{estadistica}' y
            valores numéricos.
    """
    stats = {}
    for sensor in sensores:
        if sensor not in df.columns:
            continue
        serie = df[sensor]
        stats[f"{sensor}_media"] = serie.mean()
        stats[f"{sensor}_std"] = serie.std()
        stats[f"{sensor}_max"] = serie.max()
        stats[f"{sensor}_min"] = serie.min()
        stats[f"{sensor}_valor_inicial"] = serie.iloc[0]
        stats[f"{sensor}_valor_final"] = serie.iloc[-1]
        # Pendiente de regresión lineal
        pendiente, _ = polyfit(df['TIME'], serie, 1)
        stats[f"{sensor}_pendiente_regresion"] = pendiente
        # Tiempos hasta máximo y mínimo
        stats[f"{sensor}_tiempo_hasta_maximo"] = df.loc[serie.idxmax(), 'TIME']
        stats[f"{sensor}_tiempo_hasta_minimo"] = df.loc[serie.idxmin(), 'TIME']
    return stats

def procesar_todos_archivos(data_path, columns, sensores):
    """
    Recorre todos los archivos CSV y extrae estadísticas de cada uno.

    Para cada carpeta (accidente) y cada archivo CSV dentro de ella
    (severidad), carga el archivo, selecciona las columnas indicadas,
    extrae las estadísticas de los sensores y construye un diccionario
    con la información del accidente, severidad y estadísticas.

    Args:
        data_path (str): Ruta al directorio con las subcarpetas de accidentes.
        columns (list[str]): Columnas a seleccionar del CSV (incluye 'TIME').
        sensores (list[str]): Sensores sobre los que calcular estadísticas.

    Returns:
        list[dict]: Lista de diccionarios, uno por archivo procesado.
    """
    dict_list = []
    for carpeta in os.listdir(data_path):
        carpeta_path = os.path.join(data_path, carpeta)
        if not os.path.isdir(carpeta_path):
            continue
        archivos = sorted(os.listdir(carpeta_path))
        for archivo in archivos:
            if not archivo.endswith('.csv'):
                continue
            archivo_path = os.path.join(carpeta_path, archivo)
            # Limpiar nombre para obtener severidad
            nombre_limpio = limpiar_nombre_archivo(archivo.split('.')[0])
            try:
                severidad = int(nombre_limpio)
            except ValueError:
                severidad = -1  # o manejarlo de otra forma
            df = pd.read_csv(archivo_path)
            # Seleccionar columnas si existen
            columnas_existentes = [col for col in columns if col in df.columns]
            if not columnas_existentes:
                continue
            df = df[columnas_existentes].copy()
            df = df.sort_values('TIME')
            # Construir diccionario base
            dict_row = {"accidente": carpeta, "severidad": severidad}
            # Extraer estadísticas
            stats = extraer_estadisticas(df, sensores)
            dict_row.update(stats)
            dict_list.append(dict_row)
    return dict_list

def seleccionar_columnas_por_importancia(X_train, y_train, umbral=0.007, n_estimators=100, random_state=42):
    """
    Selecciona características según su importancia en un Random Forest.

    Entrena un Random Forest temporal sobre todas las características y
    devuelve aquellas cuya importancia supera el umbral indicado. Esto
    permite reducir la dimensionalidad manteniendo las variables más
    relevantes para la clasificación.

    Args:
        X_train (pd.DataFrame): Matriz de características de entrenamiento.
        y_train (np.ndarray): Etiquetas codificadas de entrenamiento.
        umbral (float, optional): Importancia mínima para conservar una
            característica. Por defecto 0.007.
        n_estimators (int, optional): Número de árboles del Random Forest.
            Por defecto 100.
        random_state (int, optional): Semilla para reproducibilidad.
            Por defecto 42.

    Returns:
        list[str]: Nombres de las columnas seleccionadas.
    """
    rf_temp = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_temp.fit(X_train, y_train)
    importancias = rf_temp.feature_importances_
    columnas_seleccionadas = X_train.columns[importancias > umbral].tolist()
    print(f"Seleccionadas {len(columnas_seleccionadas)} características con umbral {umbral}")
    return columnas_seleccionadas
