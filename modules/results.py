import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log.txt'
)

# Asegurar que statsmodels esté disponible
try:
    import statsmodels.api as sm
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "statsmodels"])
    import statsmodels.api as sm

class analysis:
    @staticmethod
    def load_data():
        """Carga los datos procesados desde session_state."""
        try:
            logging.info("Iniciando carga de datos desde 'data_final_estandarizado.csv'")
            file_path = "data_final_estandarizado.csv"
            df = pd.read_csv(file_path)
            logging.info(f"Datos cargados exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
            return df
        except FileNotFoundError:
            error_msg = "No se encontró el archivo 'data_final_estandarizado.csv'"
            st.error(f"❌ {error_msg}.")
            logging.error(error_msg)
            return None
        except Exception as e:
            error_msg = f"Error al cargar los datos: {str(e)}"
            st.error(f"❌ {error_msg}")
            logging.error(error_msg)
            return None
    
    @staticmethod
    def train_model(X, y):
        """Entrena un modelo XGBoost Regressor."""
        logging.info("Iniciando entrenamiento del modelo XGBoost Regressor")
        
        if X.empty or y.empty:
            error_msg = "No hay suficientes datos para entrenar el modelo"
            st.error(f"⚠️ {error_msg}.")
            logging.error(error_msg)
            return None, None, (None, None, None)

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            modelo = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )
            
            modelo.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            
            y_pred = modelo.predict(X_test)
            
            return modelo, None, (X_test, y_test, y_pred)
        except Exception as e:
            error_msg = f"Error durante el entrenamiento del modelo: {e}"
            st.error(f"❌ {error_msg}")
            logging.error(error_msg)
            return None, None, (None, None, None)

    @staticmethod
    def analyze_results(y_test, y_pred):
        """Calcula métricas de regresión y muestra los resultados."""
        if y_test is None or y_pred is None:
            error_msg = "No se pueden analizar los resultados porque el modelo no se entrenó correctamente"
            st.error(f"❌ {error_msg}.")
            logging.error(error_msg)
            return

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.header("📊 Análisis de Resultados")
        st.write(f"🔹 RMSE: {rmse:.4f}")
        st.write(f"🔹 MAE: {mae:.4f}")
        st.write(f"🔹 MAPE: {mape:.4%}")
        st.write(f"🔹 R² Score: {r2:.4f}")
        
        if r2 > 0.8:
            st.success("🚀 ¡El modelo tiene un rendimiento excelente!")
        elif 0.5 < r2 <= 0.8:
            st.info("✅ El modelo tiene un buen desempeño, pero puede mejorarse.")
        else:
            st.warning("⚠️ El rendimiento del modelo es bajo. Considera ajustar los hiperparámetros o limpiar mejor los datos.")

    @staticmethod
    def plot_predictions(y_test, y_pred):
        """Grafica las predicciones frente a los valores reales."""
        if y_test is None or y_pred is None:
            return
        
        fig = px.scatter(x=y_test, y=y_pred, title="Comparación entre Valores Reales y Predichos",
                         labels={"x": "Valor Real", "y": "Valor Predicho"}, trendline="ols")
        st.plotly_chart(fig)
        
        fig_hist = px.histogram(y_pred, nbins=30, title="Distribución de Predicciones", labels={"value": "Predicciones"})
        st.plotly_chart(fig_hist)
        
        fig_box = px.box(y_pred, title="Diagrama de Caja de Predicciones")
        st.plotly_chart(fig_box)
        
        fig_residuals = px.scatter(x=y_pred, y=y_test - y_pred, title="Gráfico de Residuos",
                                   labels={"x": "Predicción", "y": "Residuo"})
        st.plotly_chart(fig_residuals)

    @staticmethod
    def show_results():
        """Ejecuta todas las funciones de análisis y visualización."""
        df = analysis.load_data()
        if df is None:
            return

        X = df.drop(columns=["Pais Destino"])
        y = df["Pais Destino"]

        modelo, _, (X_test, y_test, y_pred) = analysis.train_model(X, y)
        if modelo is None:
            return
        
        analysis.analyze_results(y_test, y_pred)
        analysis.plot_predictions(y_test, y_pred)


class app:
    def run(self):
        """Función principal para ejecutar la aplicación."""
        st.markdown('<h3 class="custom-h3">Regresión con XGBoost 🌱</h3>', unsafe_allow_html=True)
        st.write("")
        
        df = analysis.load_data()
        if df is None:
            return

        st.dataframe(df.head(6))
        
        if st.button("Ejecutar Análisis y Modelado"):
            analysis.show_results()
