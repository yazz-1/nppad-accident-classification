import os
import subprocess
import re

def convertir_mdb_a_csv(data_path, output_path):
    """
    Convierte todos los archivos .mdb dentro de data_path a CSV usando mdbtools.
    """
    for accident in os.listdir(data_path):
        accident_path = os.path.join(data_path, accident)
        if not os.path.isdir(accident_path):
            continue

        for filename in os.listdir(accident_path):
            if not filename.endswith('.mdb'):
                continue

            mdb_file = os.path.join(accident_path, filename)
            print(f"Procesando: {mdb_file}")

            # Determinar si es operación o dosis (por el nombre)
            if re.search(r'\d+dose\.mdb', filename):
                table = 'ListDS'
                subdir = 'dose'
            else:
                table = 'PlotData'
                subdir = 'operation'

            # Crear carpeta de salida
            out_dir = os.path.join(output_path, subdir, accident)
            os.makedirs(out_dir, exist_ok=True)

            # Nombre del archivo CSV de salida
            csv_name = filename.replace('.mdb', '.csv')
            csv_path = os.path.join(out_dir, csv_name)

            # Exportar con mdb-export
            cmd = ['mdb-export', '-D', '%Y-%m-%d %H:%M:%S', mdb_file, table]
            with open(csv_path, 'w') as f:
                subprocess.run(cmd, stdout=f, check=True)

            print(f"  -> Generado: {csv_path}")

if __name__ == "__main__":
    # Rutas dentro de tu proyecto
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(project_root, "data", "nppad", "raw")
    processed_path = os.path.join(project_root, "data", "nppad", "processed")

    convertir_mdb_a_csv(raw_path, processed_path)
