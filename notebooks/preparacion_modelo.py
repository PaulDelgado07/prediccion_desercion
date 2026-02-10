#nuevo proyecto streamlit-python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

df = pd.read_excel(
    'C:/Users/Paul/OneDrive/Desktop/Proyecto_desercion/data/dataset.xlsx'
)

df.head(10)

#Conversión del promedio academico a formato numerico flotante
df['PROMEDIO'] = (
    df['PROMEDIO']
    .str.replace(',','.', regex= False)
    .astype(float) 
)

#Creación de una variable auxiliar para identificar materias reprobadas
df['reprobada'] = df['ESTADO'].apply(
    lambda x: 1 if x == 'REPROBADA' else 0
)

#Agregación de la informacion por estudiante
df_estudiantes = df.groupby('ESTUDIANTE').agg(
    prom_general=('PROMEDIO', 'mean'),
    asist_prom=('ASISTENCIA', 'mean'),
    materias_reprobadas=('reprobada', 'sum'),
    max_no_vez=('NO. VEZ', 'max')
).reset_index()

#total de materias cursadas por estudiante
total_materias_por_estudiante = (
    df.groupby('ESTUDIANTE')
    .size()
    .reset_index(name='total_materias')
)

df_estudiantes = df_estudiantes.merge(
    total_materias_por_estudiante, 
    on='ESTUDIANTE'
)

#Calculo de la tasa de reprobación
df_estudiantes['tasa_reprobacion'] = (
    df_estudiantes['materias_reprobadas'] / df_estudiantes['total_materias']
) 

print(f"Estudiantes procesados: {len(df_estudiantes)}")

# Definir deserción
condiciones_desercion = (
    (df_estudiantes['tasa_reprobacion'] >= 0.25) |      # Bajado de 0.30
    (df_estudiantes['asist_prom'] < 70) |               # Subido de 60 ← MÁS ESTRICTO
    (df_estudiantes['prom_general'] < 6.5) |            # Subido de 6.0
    (df_estudiantes['max_no_vez'] >= 2) |               # Bajado de 3
    (df_estudiantes['materias_reprobadas'] >= 3)        # Bajado de 4
)

df_estudiantes['desercion'] = condiciones_desercion.astype(int)

# Verificar distribución
n_desertores = df_estudiantes['desercion'].sum()
n_no_desertores = len(df_estudiantes) - n_desertores
tasa_desercion = (n_desertores / len(df_estudiantes)) * 100

print(f"\n--- Distribución de Deserción ---")
print(f"Desertores (clase 1): {n_desertores} ({tasa_desercion:.1f}%)")
print(f"No desertores (clase 0): {n_no_desertores} ({100-tasa_desercion:.1f}%)")

if tasa_desercion > 60:
    print("\n ADVERTENCIA: Tasa de deserción muy alta (>60%)")
    print("    Considera ajustar los criterios si es necesario")
elif tasa_desercion < 15:
    print("\n ADVERTENCIA: Tasa de deserción muy baja (<15%)")
    print("    Considera ajustar los criterios si es necesario")
else:
    print("\n Tasa de deserción en rango razonable (15-60%)")

df_final = df_estudiantes

# Variables independientes (features)
X = df_final[[
    'prom_general', 
    'asist_prom', 
    'materias_reprobadas', 
    'max_no_vez'
]]

# Variable dependiente (target)
y = df_final['desercion']

print(f"\nVariables independientes (X): {list(X.columns)}")
print(f"Variable dependiente (y): desercion")

# Division del conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y
)
print(f"\nConjunto de entrenamiento: {len(X_train)} estudiantes")
print(f"Conjunto de prueba: {len(X_test)} estudiantes")
print(f"\nDistribución en entrenamiento:")
print(f"  - Desertores: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
print(f"  - No desertores: {len(y_train)-y_train.sum()} ({(len(y_train)-y_train.sum())/len(y_train)*100:.1f}%)")

# Definición del modelo Random forest 
modelo = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced'
)
modelo.fit(X_train, y_train)
print("\nModelo entrenado exitosamente")

y_pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred, target_names=['No Desertor', 'Desertor']))

print("\n--- Matriz de Confusión ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

importancias = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': modelo.feature_importances_
}).sort_values(by='Importancia', ascending=False)

print("\n", importancias.to_string(index=False))

joblib.dump(modelo, 'C:/Users/Paul/OneDrive/Desktop/Proyecto_desercion/model/modelo_desercion.pkl')
print("\nModelo guardado: modelo_desercion.pkl")


# Guardar también las columnas para uso en Streamlit
metadata = {
    'columnas': list(X.columns),
    'accuracy': accuracy,
    'criterios_desercion': {
        'tasa_reprobacion': 0.30,
        'asistencia_minima': 60,
        'promedio_minimo': 6.0,
        'max_repeticiones': 3,
        'materias_reprobadas_max': 4
    }
}
joblib.dump(
    metadata,
    'C:/Users/Paul/OneDrive/Desktop/Proyecto_desercion/model/metadata.pkl'
)

# Gráfico de importancia
# plt.figure(figsize=(8, 5))
# plt.barh(importancias['Variable'], importancias['Importancia'], color='steelblue')
# plt.xlabel('Importancia')
# plt.title('Importancia de las Variables en la Predicción', fontweight='bold')
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.savefig('C:/Users/Paul/OneDrive/Desktop/Proyecto_desercion/importancia_variables.png', 
#             dpi=300, bbox_inches='tight')
# print("\n Gráfico guardado: importancia_variables.png")
# plt.close()

# # Matriz de confusión visual
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['No Desertor', 'Desertor'],
#             yticklabels=['No Desertor', 'Desertor'])
# plt.title('Matriz de Confusión', fontweight='bold', fontsize=14)
# plt.ylabel('Valor Real')
# plt.xlabel('Predicción')
# plt.tight_layout()
# plt.savefig('C:/Users/Paul/OneDrive/Desktop/Proyecto_desercion/matriz_confusion.png',
#             dpi=300, bbox_inches='tight')
# print(" Gráfico guardado: matriz_confusion.png")
# plt.close()


# print("\n" + "="*80)
# print("GUARDANDO MODELO...")
# print("="*80)

# joblib.dump(modelo, 'C:/Users/Paul/OneDrive/Desktop/Proyecto_desercion/model/modelo_desercion.pkl')
# print("\n Modelo guardado: modelo_desercion.pkl")

# # Guardar también las columnas para uso en Streamlit
# metadata = {
#     'columnas': list(X.columns),
#     'accuracy': accuracy,
#     'criterios_desercion': {
#         'tasa_reprobacion': 0.30,
#         'asistencia_minima': 60,
#         'promedio_minimo': 6.0,
#         'max_repeticiones': 3,
#         'materias_reprobadas_max': 4
#     }
# }

# print("\n" + "="*80)
# print("¡PROCESO COMPLETADO EXITOSAMENTE!")
# print("="*80)
# print(f"\nResumen:")
# print(f"  • Estudiantes analizados: {len(df_estudiantes)}")
# print(f"  • Tasa de deserción: {tasa_desercion:.1f}%")
# print(f"  • Accuracy del modelo: {accuracy:.4f}")
# print(f"  • Modelo guardado correctamente")
# print("\nArchivos generados:")
# print("  • modelo_desercion.pkl")
# print("  • metadata.pkl")
# print("  • importancia_variables.png")
# print("  • matriz_confusion.png")
# print("\n¡Ahora puedes ejecutar tu aplicación Streamlit!")
# print("="*80)
