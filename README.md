# Predicción de Deserción Estudiantil

Este proyecto tiene como objetivo desarrollar un modelo de aprendizaje automático capaz de predecir el riesgo de deserción estudiantil, utilizando variables académicas como el promedio general, la asistencia, el número de materias reprobadas y el número de repeticiones de asignaturas.

El sistema permite entrenar un modelo predictivo y posteriormente utilizarlo en una aplicación interactiva desarrollada con Streamlit.

## Metodología

El proyecto sigue la metodología "CRISP-DM", abarcando las siguientes fases:

1. **Comprensión del negocio**  
   Identificación del problema de la deserción estudiantil y su impacto académico.

2. **Comprensión de los datos**  
   Análisis de un conjunto de datos académicos históricos de estudiantes.

3. **Preparación de los datos**  
   Limpieza, transformación y agregación de datos a nivel de estudiante.

4. **Modelado**  
   Entrenamiento de un modelo **Random Forest Classifier** para la predicción de deserción.

5. **Evaluación**  
   Evaluación del modelo mediante métricas como accuracy, matriz de confusión y reporte de clasificación.

6. **Despliegue**  
   Implementación del modelo en una aplicación web con Streamlit.

## Tecnologías utilizadas

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

