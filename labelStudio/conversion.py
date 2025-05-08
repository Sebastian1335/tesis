import pandas as pd
import json
import os

# Ruta al CSV exportado por Label Studio
input_csv = 'labelstudio_export.csv'
# (Opcional) Ruta donde quieres guardar el CSV resultante
output_csv = 'transformado.csv'

# 1. Carga el CSV
df = pd.read_csv(input_csv)

# 2. Función para parsear el campo 'errores'
def parse_errors(cell):
    # Si es un JSON con "choices", extrae la lista
    try:
        obj = json.loads(cell)
        if isinstance(obj, dict) and 'choices' in obj:
            return obj['choices']
    except (json.JSONDecodeError, TypeError):
        pass
    # Si viene como string plano, devuélvelo en lista
    if pd.notna(cell):
        return [cell]
    return []

# 3. Aplica el parseo y guarda la lista en una columna nueva
df['error_list'] = df['errores'].apply(parse_errors)

# 4. Crea columnas one‑hot para cada tipo de error
errors_dummies = (
    df['error_list']
      .explode()
      .pipe(lambda s: pd.get_dummies(s, prefix='errores___'))
      .groupby(level=0)
      .max()
      .fillna(0)
      .astype(int)
)

# 5. Extrae solo el nombre de archivo para la columna 'video'
df['video'] = df['video'].apply(lambda x: os.path.basename(x))

# 6. Monta el DataFrame final
result = pd.concat([
    df['video'],
    df['lado'],
    df['movimiento'],
    errors_dummies
], axis=1)

# 7. (Opcional) Guarda a CSV
result.to_csv(output_csv, index=False)

print("Transformación completada. Vista previa:")
print(result.head())
