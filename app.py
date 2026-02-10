import streamlit as st
import pandas as pd
import joblib

# Configuracion de todo el programa 
st.set_page_config(
    page_title="Sistema de Predicción de Deserción", 
    page_icon="C:/Users/Paul/OneDrive/Desktop/Proyecto_desercion/images/UG.png",
    layout="centered"
)

# Carga del modelo
modelo = joblib.load("C:/Users/Paul/OneDrive/Desktop/Proyecto_desercion/model/modelo_desercion.pkl")

# SIDEBAR
st.sidebar.title("Proyecto Académico")
st.sidebar.markdown("""
**Sistema de Predicción de Deserción Estudiantil**

 - Modelo: Random Forest  
 - Metodología: CRISP-DM  
 - Objetivo: Identificar estudiantes en riesgo de abandono académico
""")

st.sidebar.info(
    "Esta herramienta nos permite estimar el riesgo de deserción "
    "a partir de variables académicas históricas."
    
)

# Encabezado principal
st.image(
    "C:/Users/Paul/OneDrive/Desktop/Proyecto_desercion/images/foto_desercion.png",
    width= 700
)

st.title("Predicción de Deserción Estudiantil")
st.markdown(
    "Sistema de apoyo a la toma de decisiones académicas "
    "basado en técnicas de **minería de datos**."
)

st.markdown("---")


# Formulario de ingreso de datos
st.subheader("📋 Ingreso de datos del estudiante")
st.write("Complete la información académica para realizar la predicción.")

col1, col2 = st.columns(2)

with col1:
    prom_general = st.slider(
        "📘 Promedio académico",
        0.0, 10.0, 6.0
    )
    asist_prom = st.slider(
        "📊 Asistencia promedio (%)",
        0, 100, 80
    )

with col2:
    materias_reprobadas = st.number_input(
        "❌ Materias reprobadas",
        0, 10, 0
    )
    max_no_vez = st.number_input(
        "🔁 Máximo número de repeticiones",
        0, 5, 0
    )


# predicción
st.markdown("")

if st.button("Predecir riesgo de deserción"):
    X = pd.DataFrame([[
        prom_general,
        asist_prom,
        materias_reprobadas,
        max_no_vez
    ]], columns=[
        "prom_general",
        "asist_prom",
        "materias_reprobadas",
        "max_no_vez"
    ])

    pred = modelo.predict(X)[0]
    prob = modelo.predict_proba(X)[0][1]

    st.markdown("---")
    st.subheader("Resultado de la predicción")

    if pred == 1:
        st.error(
            f"⚠️ **Alto riesgo de deserción**\n\n"
            #f"Probabilidad estimada: **{prob:.2%}**"
        )
        st.metric(
            label="Probabilidad de deserción",
            value=f"{prob:.2%}"
        )

        st.progress(int(prob * 100))
    else:
        st.success(
            f"✅ **Bajo riesgo de deserción**\n\n",
            #f"Probabilidad estimada: **{prob:.2%}**"    
        )
        st.metric(
            label="Probabilidad de deserción",
            value=f"{prob:.2%}"
        )
        st.progress(int(prob * 100))

# pie de la pagina
st.markdown("---")
st.markdown(
    "<center>Proyecto desarrollado en Python y Streamlit | "
    "Ciencias de Datos e Inteligencia Artificial</center>",
    unsafe_allow_html=True
)

