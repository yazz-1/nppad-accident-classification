# Datos del proyecto

Esta carpeta contiene los datos utilizados en el proyecto de clasificación de accidentes nucleares. Debido al tamaño de los archivos originales, no todos los datos se incluyen en el repositorio. A continuación se explica qué hay, qué falta y cómo obtenerlo.

## 📂 Estructura

```
data/
├── processed/
│   ├── features_12sensores_1217muestras.csv   # Dataset final de características (incluido)
│   ├── dose/                                  # Datos de dosis (NO incluidos por tamaño)
│   └── operation/                             # Datos de operación (NO incluidos por tamaño)
└── README.md                                  # Este archivo
```

### Archivos incluidos

- **`processed/features_12sensores_1217muestras.csv`**: Dataset tabular generado en el notebook `02_Feature_Engineering.ipynb`. Contiene 1217 filas (simulaciones) y 110 columnas (2 de metadatos + 108 estadísticas de los 12 sensores). Es la entrada para el modelado.

### Archivos NO incluidos (por tamaño)

- **`processed/operation/`**: Series temporales de los 96 sensores para cada simulación (~600 MB). Se generan a partir de los archivos `.mdb` originales.
- **`processed/dose/`**: Datos de dosis de radionucleidos (~100 MB). También se generan a partir de los `.mdb`.

## 📥 Cómo obtener los datos originales

El dataset original se llama **NPPAD** (*Nuclear Power Plant Accident Data*) y es de acceso público. Está disponible en:

- **GitHub**: [https://github.com/thu-inet/NuclearPowerPlantAccidentData](https://github.com/thu-inet/NuclearPowerPlantAccidentData)
- **Figshare**: [https://springernature.figshare.com/collections/NPPAD_An_Open_Time-series_Dataset_Covering_Various_Accidents_for_Nuclear_Power_Plants/6238473](https://springernature.figshare.com/collections/NPPAD_An_Open_Time-series_Dataset_Covering_Various_Accidents_for_Nuclear_Power_Plants/6238473)

### Pasos para reproducir los datos procesados

1. **Descarga el dataset NPPAD** desde el repositorio de GitHub (clona el repo o descarga el ZIP).

2. **Coloca la carpeta `NPPAD`** dentro de `data/raw/` (crea esta carpeta si no existe):

   ```
   data/
   └── raw/
       └── NPPAD/
           ├── ATWS/
           ├── FLB/
           ├── LOCA/
           └── ...
   ```

3. **Convierte los archivos `.mdb` a `.csv`** ejecutando el script:

   ```bash
   python scripts/convertir_mdb_a_csv.py
   ```

   Este script usa `mdbtools` (en Linux/Mac) o `pyodbc` (en Windows) para exportar las tablas `PlotData` y `ListDS` de cada archivo `.mdb` a CSV. Los archivos resultantes se guardan en `data/processed/operation/` y `data/processed/dose/`.

   **Requisito**: tener instalado `mdbtools` (Linux/Mac) o los drivers ODBC de Microsoft Access (Windows).

4. **Genera el dataset de características** ejecutando el notebook:

   ```
   notebooks/02_Feature_Engineering.ipynb
   ```

   Este notebook recorre todos los CSV de `data/processed/operation/`, extrae 9 estadísticas por cada uno de los 12 sensores y guarda el resultado en `data/processed/features_12sensores_1217muestras.csv`.

## 🔗 Enlaces útiles

- **Paper científico del dataset**: [Nature Scientific Data](https://www.nature.com/articles/s41597-022-01879-1)
- **Descripción de sensores**: consulta el README del repositorio original o el notebook `01_EDA.ipynb` para ver la lista completa de variables.

## 📌 Notas

- Los archivos `.mdb` originales ocupan **~15 GB**, pero tras la conversión a CSV se reducen a **~600 MB** (operación) + **~100 MB** (dosis), ya que los `.mdb` incluyen índices y metadatos que se eliminan al exportar.
- Si solo quieres reproducir el modelado sin regenerar los datos, puedes usar directamente `features_12sensores_1217muestras.csv`, que ya está incluido en el repositorio.

## 📄 Licencia de los datos

El dataset NPPAD se distribuye bajo la licencia **CC BY 4.0** (Creative Commons Attribution 4.0 International). Consulta el repositorio original para más detalles.
