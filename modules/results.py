import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class analysis:
    @staticmethod
    def load_data():
        """Carga los datos procesados desde session_state."""
        if "df_procesado" in st.session_state:
            return st.session_state["df_procesado"]
        else:
            st.error("❌ No hay datos procesados disponibles. Ejecuta primero el Pipeline.")
            return None
        
    @staticmethod
    def train_model(X, y):
        """Entrena un modelo XGBoost con monitoreo de pérdida y manejo de excepciones."""
        if X.empty or y.empty:
            st.error("⚠️ No hay suficientes datos para entrenar el modelo.")
            return None, None, (None, None, None)

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            modelo = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

            eval_set = [(X_train, y_train), (X_test, y_test)]
            modelo.fit(X_train, y_train, eval_metric="logloss", eval_set=eval_set, verbose=False)

            resultados = modelo.evals_result()
            y_pred = modelo.predict(X_test)

            return modelo, resultados, (X_test, y_test, y_pred)
        except Exception as e:
            st.error(f"❌ Error durante el entrenamiento del modelo: {e}")
            return None, None, (None, None, None)

    @staticmethod
    def analyze_results(modelo, X_test, y_test):
        """Muestra métricas del modelo con validaciones previas."""
        if modelo is None or X_test is None or y_test is None:
            st.error("❌ No se pueden analizar los resultados porque el modelo no se entrenó correctamente.")
            return

        precision = modelo.score(X_test, y_test)

        st.header("📊 Análisis de Resultados")
        if precision > 0.90:
            st.success(f"🌟 Excelente rendimiento con una precisión de {precision:.2%}.")
        elif 0.75 <= precision <= 0.90:
            st.info(f"✅ Buen rendimiento con una precisión de {precision:.2%}.")
        else:
            st.warning(f"⚠️ Precisión baja ({precision:.2%}). Considera revisar los datos y ajustar hiperparámetros.")

    @staticmethod
    def plot_loss(resultados):
        """Grafica la evolución de la pérdida del modelo, con validaciones."""
        if not resultados or "validation_0" not in resultados:
            st.error("❌ No hay resultados de entrenamiento disponibles para graficar la pérdida.")
            return

        st.header("📉 Evolución de la Pérdida del Modelo")
        fig = px.line(
            x=range(len(resultados["validation_0"]["logloss"])),
            y=resultados["validation_0"]["logloss"],
            labels={"x": "Iteraciones", "y": "Pérdida LogLoss"},
            title="Curva de Pérdida en el Entrenamiento"
        )
        st.plotly_chart(fig)

    @staticmethod
    def graficar_importancia(modelo, df_original):
        """Muestra la importancia de las características."""
        if modelo is None or df_original.empty:
            st.error("❌ No se puede calcular la importancia de características.")
            return

        st.header("🔎 Importancia de las Características")
        importancia = modelo.feature_importances_
        fig = px.bar(
            x=df_original.columns,
            y=importancia,
            labels={"x": "Características", "y": "Importancia"},
            title="Importancia de Características en la Predicción"
        )
        st.plotly_chart(fig)

    @staticmethod
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

    @staticmethod
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

    def show_results():
        """Ejecuta todas las funciones de análisis y visualización."""
        st.markdown('<h3 class="custom-h3">Predicción de Destino de Exportación de Fertilizantes 🌱</h3>', unsafe_allow_html=True)

        df = analysis.load_data()
        if df is None:
            return

        X = df.drop(columns=["Pais Destino"])
        y = df["Pais Destino"]

        modelo, resultados, (X_test, y_test, y_pred) = analysis.entrenar_modelo(X, y)
        if modelo is None:
            return

        analysis.analizar_resultados(modelo, X_test, y_test)
        analysis.graficar_perdida(resultados)
        analysis.graficar_importancia(modelo, df)
