import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from pipeline import eda  # Importamos el pipeline

def cargar_datos():
    """
    Obtiene los datos procesados desde el pipeline y lo carga desde session_state.
    """
    if "df_procesado" in st.session_state:
        return st.session_state["df_procesado"]
    else:
        st.error("No hay datos procesados disponibles. Ejecuta primero el Pipeline.")
        return None

def entrenar_modelo(X, y):
    """
    Entrena un modelo XGBoost con monitoreo de pérdida.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    eval_set = [(X_train, y_train), (X_test, y_test)]
    modelo.fit(X_train, y_train, eval_metric="logloss", eval_set=eval_set, verbose=False)

    resultados = modelo.evals_result()

    return modelo, resultados, (X_test, y_test, y_pred)

def analizar_resultados(metricas):
    """
    Analiza los resultados del modelo y proporciona una interpretación en base a las métricas.
    """
    precision = metricas["Precisión"]
    exactitud = metricas["Exactitud"]

    st.header("📊 Análisis de Resultados")

    if precision > 0.90:
        st.success(f"🌟 Excelente rendimiento del modelo con una precisión de {precision:.2%}.")
    elif 0.75 <= precision <= 0.90:
        st.info(f"✅ Buen rendimiento con una precisión de {precision:.2%}.")
    else:
        st.warning(f"⚠️ Precisión baja ({precision:.2%}). Considera revisar los datos y ajustar hiperparámetros.")

    if exactitud > 0.90:
        st.success(f"🔍 Alta exactitud ({exactitud:.2%}).")
    elif 0.75 <= exactitud <= 0.90:
        st.info(f"📌 Exactitud moderada ({exactitud:.2%}).")
    else:
        st.warning(f"⚠️ Exactitud baja ({exactitud:.2%}). Revisa posibles problemas en los datos.")

def graficar_perdida(resultados):
    """
    Grafica la evolución de la pérdida durante el entrenamiento.
    """
    st.header("📉 Evolución de la Pérdida del Modelo")

    fig = px.line(
        x=range(len(resultados["validation_0"]["logloss"])),
        y=resultados["validation_0"]["logloss"],
        labels={"x": "Iteraciones", "y": "Pérdida LogLoss"},
        title="Curva de Pérdida en el Entrenamiento"
    )
    fig.add_scatter(
        x=range(len(resultados["validation_1"]["logloss"])),
        y=resultados["validation_1"]["logloss"],
        mode="lines",
        name="Validación"
    )

    st.plotly_chart(fig)

def graficar_importancia(modelo, df_original):
    """
    Grafica la importancia de las características.
    """
    st.header("🔎 Importancia de las Características")

    importancia = modelo.feature_importances_
    fig = px.bar(
        x=df_original.columns,
        y=importancia,
        labels={"x": "Características", "y": "Importancia"},
        title="Importancia de Características en la Predicción"
    )

    st.plotly_chart(fig)

def graficar_shap(modelo, X_test, df_original):
    """
    Grafica los valores SHAP para interpretar el modelo.
    """
    st.header("🤖 Interpretabilidad con SHAP")

    explainer = shap.Explainer(modelo)
    shap_values = explainer(X_test)

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, df_original, plot_type="bar", show=False)
    st.pyplot(fig)

def graficar_predicciones(y_test, y_pred, df_original):
    """
    Muestra un histograma de predicciones del modelo.
    """
    st.header("📊 Distribución de Predicciones")

    df_pred = pd.DataFrame({"Real": y_test, "Predicción": y_pred})
    df_pred["Real"] = df_pred["Real"].map(lambda x: df_original["Pais Destino"].unique()[x])
    df_pred["Predicción"] = df_pred["Predicción"].map(lambda x: df_original["Pais Destino"].unique()[x])

    fig = px.histogram(
        df_pred,
        x="Predicción",
        title="Distribución de Predicciones por País Destino"
    )

    st.plotly_chart(fig)

def mostrar_resultados():
    """
    Módulo principal para mostrar resultados del modelo.
    """
    st.title("Predicción de Destino de Exportación de Fertilizantes")
    
    df = cargar_datos()
    if df is None:
        return
    
    # La data ya viene preprocesada del pipeline
    X = df.drop(columns=["Pais Destino"])
    y = df["Pais Destino"]

    modelo, resultados, (X_test, y_test) = entrenar_modelo(X, y)

    analizar_resultados({"Precisión": modelo.score(X_test, y_test), "Exactitud": modelo.score(X_test, y_test)})
    
    graficar_perdida(resultados)
    graficar_importancia(modelo, df)
    graficar_shap(modelo, X_test, df)
    graficar_predicciones(y_test, modelo.predict(X_test), df)

if __name__ == "__main__":
    mostrar_resultados()
    
    st.header("Informe de Clasificación")
    reporte = classification_report(y_test, y_pred, target_names=paises_destino)
    st.text(reporte)
