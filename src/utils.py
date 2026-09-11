import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder

def cargar_datos(ruta_carpeta):
    """
    Carga el primer archivo CSV de cada carpeta (un accidente por carpeta).

    Recorre el directorio indicado, entra en cada subcarpeta (que representa
    un tipo de accidente) y carga el primer archivo CSV que encuentra. Esto
    permite tener una muestra representativa de cada accidente para el EDA.

    Args:
        ruta_carpeta (str): Ruta al directorio que contiene las subcarpetas
            de accidentes (ej. "../data/processed/operation/").

    Returns:
        tuple: (df_list, accident_names)
            - df_list (list[pd.DataFrame]): Lista de DataFrames, uno por accidente.
            - accident_names (list[str]): Lista con los nombres de las carpetas
              (tipos de accidente) en el mismo orden que df_list.
    """
    df_list = []
    accident_names = []
    for carpeta in sorted(os.listdir(ruta_carpeta)):
        carpeta_path = os.path.join(ruta_carpeta, carpeta)
        if not os.path.isdir(carpeta_path):
            continue
        archivos = sorted(os.listdir(carpeta_path))
        if not archivos:
            continue
        archivo_path = os.path.join(carpeta_path, archivos[0])
        df = pd.read_csv(archivo_path)
        df_list.append(df)
        accident_names.append(carpeta)
    return df_list, accident_names

def cargar_dataset(ruta_csv):
    """
    Carga el dataset de características y separa X (predictoras) e y (objetivo).

    Lee el CSV generado en el notebook de Feature Engineering, separa la
    columna 'accidente' como variable objetivo y codifica las etiquetas con
    LabelEncoder para que sean numéricas.

    Args:
        ruta_csv (str): Ruta al archivo CSV con las características.

    Returns:
        tuple: (X, y, le)
            - X (pd.DataFrame): Matriz de características (todas las columnas
              excepto 'accidente').
            - y (np.ndarray): Vector de etiquetas codificadas numéricamente.
            - le (LabelEncoder): Objeto LabelEncoder ajustado, útil para
              recuperar los nombres originales de las clases.
    """
    df = pd.read_csv(ruta_csv)
    X = df.drop('accidente', axis=1)
    le = LabelEncoder()
    y = le.fit_transform(df['accidente'])
    return X, y, le

def graficar_series_temporales(df_list, accident_names, sensores, unidades,
                               estilos, carpeta_salida, prefijo="", dpi=150, show=False):
    """
    Genera un gráfico de series temporales por cada accidente y los guarda.

    Para cada DataFrame de la lista, dibuja las series temporales de los
    sensores indicados en un mismo eje. Cada sensor se representa con un
    estilo de línea y marcador diferente (para no depender del color).
    Los gráficos se guardan en la carpeta indicada.

    Args:
        df_list (list[pd.DataFrame]): Lista de DataFrames con columnas 'TIME'
            y los sensores.
        accident_names (list[str]): Nombres de los accidentes (uno por df).
        sensores (list[str]): Nombres de las columnas de sensores a graficar.
        unidades (list[str]): Unidades correspondientes a cada sensor.
        estilos (list[dict]): Lista de diccionarios con claves 'linestyle'
            y 'marker' para cada sensor.
        carpeta_salida (str): Ruta donde se guardarán las imágenes.
        prefijo (str, optional): Prefijo para el nombre de los archivos.
            Por defecto "".
        dpi (int, optional): Resolución de las imágenes. Por defecto 150.
        show (bool, optional): Si es True, muestra la figura en el notebook
            antes de cerrarla. Por defecto False.

    Returns:
        None
    """
    os.makedirs(carpeta_salida, exist_ok=True)
    for idx, df in enumerate(df_list):
        accidente = accident_names[idx]
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, sensor in enumerate(sensores):
            if sensor in df.columns:
                etiqueta = f"{sensor} ({unidades[i]})" if i < len(unidades) else sensor
                ax.plot(df['TIME'], df[sensor],
                        label=etiqueta,
                        linestyle=estilos[i]['linestyle'],
                        marker=estilos[i]['marker'],
                        markersize=4,
                        markevery=max(1, len(df)//30),
                        linewidth=1.5)
        ax.set_title(accidente, fontsize=14)
        ax.set_xlabel('Tiempo (s)', fontsize=12)
        ax.set_ylabel('Valor', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12)

        nombre_archivo = f"{prefijo}_{accidente}_5_sensors.png"
        ruta_completa = os.path.join(carpeta_salida, nombre_archivo)
        plt.savefig(ruta_completa, dpi=dpi, bbox_inches='tight')
        print(f"Guardado: {ruta_completa}")

        if show:
            plt.show()  # Muestra la figura en el notebook
        else:
            plt.close(fig)  # Cierra para liberar memoria

def crear_carpeta(ruta):
    """
    Crea una carpeta (y sus padres) si no existe.

    Args:
        ruta (str): Ruta de la carpeta a crear.

    Returns:
        None
    """
    os.makedirs(ruta, exist_ok=True)

def limpiar_nombre_archivo(nombre_archivo):
    """
    Elimina un guion inicial si existe en el nombre del archivo.

    Algunos archivos del dataset original tienen nombres como '-1.csv',
    este guion es un artefacto que debe eliminarse para poder convertir
    el nombre a entero (severidad).

    Args:
        nombre_archivo (str): Nombre del archivo sin extensión.

    Returns:
        str: Nombre limpio, sin guion inicial si lo tenía.
    """
    if nombre_archivo.startswith('-'):
        return nombre_archivo[1:]
    return nombre_archivo

def obtener_clases_modelo(model, le):
    """
    Devuelve los nombres de las clases que el modelo conoce.

    Dado un modelo entrenado (con atributo classes_) y un LabelEncoder
    ajustado, devuelve los nombres originales de las clases que el modelo
    puede predecir.

    Args:
        model: Modelo de clasificación entrenado (con atributo classes_).
        le (LabelEncoder): LabelEncoder ajustado.

    Returns:
        np.ndarray: Nombres de las clases que el modelo conoce.
    """
    clases_modelo_enteros = model.classes_
    return le.inverse_transform(clases_modelo_enteros)
