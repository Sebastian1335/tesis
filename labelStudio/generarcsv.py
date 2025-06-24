import os
import csv

# Ruta principal que contiene las subcarpetas con videos
carpeta_principal = 'D:/la-u/ciclo 2025-1/Seminario/DATASET'

# Nombre del archivo CSV de salida
csv_salida = 'dataset_errores.csv'

# Lista para guardar los nombres de los archivos .mp4
nombres_videos = []

# Recorrer todas las subcarpetas
for root, dirs, files in os.walk(carpeta_principal):
    for archivo in files:
        if archivo.lower().endswith('.mp4'):
            nombres_videos.append(archivo)

# Escribir los nombres en el archivo CSV
with open(csv_salida, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['nombre_video'])  # Encabezado
    for nombre in nombres_videos:
        writer.writerow([nombre])

print(f"CSV creado exitosamente con {len(nombres_videos)} nombres de video.")
