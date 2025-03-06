import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log.txt'
)

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
            error_msg = "No se encontró el archivo 'data_final_estandarizado.csv' en la carpeta del código"
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
        """Entrena un modelo XGBoost con monitoreo de pérdida y manejo de excepciones."""
        logging.info("Iniciando entrenamiento del modelo XGBoost")
        
        if X.empty or y.empty:
            error_msg = "No hay suficientes datos para entrenar el modelo"
            st.error(f"⚠️ {error_msg}.")
            logging.error(error_msg)
            return None, None, (None, None, None)

        try:
            logging.info("Realizando división train-test")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logging.info(f"División completada: X_train: {X_train.shape}, X_test: {X_test.shape}")
            
            logging.info("Inicializando modelo XGBClassifier")
            # Especificar eval_metric en la inicialización en lugar de en fit()
            modelo = xgb.XGBClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                random_state=42,
                eval_metric="logloss"  # Colocar eval_metric aquí, no en fit()
            )

            logging.info("Configurando conjunto de evaluación")
            eval_set = [(X_train, y_train), (X_test, y_test)]
            
            logging.info("Iniciando entrenamiento del modelo")
            # Quitar eval_metric de fit() y usar solo eval_set
            modelo.fit(
                X_train, 
                y_train, 
                eval_set=eval_set, 
                verbose=False
            )
            logging.info("Entrenamiento del modelo completado exitosamente")

            logging.info("Obteniendo resultados de evaluación")
            resultados = modelo.evals_result()
            
            logging.info("Generando predicciones en conjunto de prueba")
            y_pred = modelo.predict(X_test)
            logging.info(f"Predicciones generadas: {len(y_pred)} muestras")

            return modelo, resultados, (X_test, y_test, y_pred)
        except Exception as e:
            error_msg = f"Error durante el entrenamiento del modelo: {e}"
            st.error(f"❌ {error_msg}")
            logging.error(error_msg)
            return None, None, (None, None, None)

    @staticmethod
    def analyze_results(modelo, X_test, y_test):
        """Muestra métricas del modelo con validaciones previas."""
        logging.info("Iniciando análisis de resultados del modelo")
        
        if modelo is None or X_test is None or y_test is None:
            error_msg = "No se pueden analizar los resultados porque el modelo no se entrenó correctamente"
            st.error(f"❌ {error_msg}.")
            logging.error(error_msg)
            return

        try:
            logging.info("Calculando precisión del modelo")
            precision = modelo.score(X_test, y_test)
            logging.info(f"Precisión del modelo: {precision:.4f}")

            st.header("📊 Análisis de Resultados")
            if precision > 0.90:
                st.success(f"🌟 Excelente rendimiento con una precisión de {precision:.2%}.")
                logging.info(f"Rendimiento excelente: precisión {precision:.2%}")
            elif 0.75 <= precision <= 0.90:
                st.info(f"✅ Buen rendimiento con una precisión de {precision:.2%}.")
                logging.info(f"Rendimiento bueno: precisión {precision:.2%}")
            else:
                st.warning(f"⚠️ Precisión baja ({precision:.2%}). Considera revisar los datos y ajustar hiperparámetros.")
                logging.warning(f"Rendimiento bajo: precisión {precision:.2%}")
        except Exception as e:
            error_msg = f"Error al analizar resultados del modelo: {e}"
            st.error(f"❌ {error_msg}")
            logging.error(error_msg)

    @staticmethod
    def plot_loss(resultados):
        """Grafica la evolución de la pérdida del modelo, con validaciones."""
        logging.info("Iniciando generación de gráfica de pérdida")
        
        if not resultados or "validation_0" not in resultados:
            error_msg = "No hay resultados de entrenamiento disponibles para graficar la pérdida"
            st.error(f"❌ {error_msg}.")
            logging.error(error_msg)
            return

        try:
            st.header("📉 Evolución de la Pérdida del Modelo")
            logging.info("Preparando datos para la gráfica de pérdida")
            
            # Plot training and validation loss
            logging.info("Creando figura Plotly para las curvas de pérdida")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(resultados["validation_0"]["logloss"]))),
                y=resultados["validation_0"]["logloss"],
                mode='lines',
                name='Training Loss'
            ))
            fig.add_trace(go.Scatter(
                x=list(range(len(resultados["validation_1"]["logloss"]))),
                y=resultados["validation_1"]["logloss"],
                mode='lines',
                name='Validation Loss'
            ))
            fig.update_layout(
                title="Curva de Pérdida en el Entrenamiento",
                xaxis_title="Iteraciones",
                yaxis_title="Pérdida LogLoss"
            )
            logging.info("Mostrando gráfica de curva de pérdida")
            st.plotly_chart(fig)
            logging.info("Gráfica de pérdida generada y mostrada exitosamente")
        except Exception as e:
            error_msg = f"Error al generar gráfica de pérdida: {e}"
            st.error(f"❌ {error_msg}")
            logging.error(error_msg)

    @staticmethod
    def plot_feature_importance(modelo, X):
        """Muestra la importancia de las características con más detalle."""
        logging.info("Iniciando visualización de importancia de características")
        
        if modelo is None or X is None:
            error_msg = "No se puede calcular la importancia de características"
            st.error(f"❌ {error_msg}.")
            logging.error(error_msg)
            return

        try:
            st.header("🔎 Importancia de las Características")
            logging.info("Extrayendo importancia de características del modelo")
            
            # Get feature importance from model
            importancia = modelo.feature_importances_
            feature_names = X.columns
            logging.info(f"Obtenidas {len(importancia)} características con sus valores de importancia")
            
            # Create DataFrame for easier manipulation
            logging.info("Creando DataFrame de importancias")
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importancia
            }).sort_values('Importance', ascending=False)
            
            # Create bar chart with Plotly
            logging.info("Generando gráfica de barras de importancia de características")
            fig = px.bar(
                importance_df,
                x='Importance', 
                y='Feature',
                orientation='h',
                title="Importancia de Características (ordenadas)",
                color='Importance',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig)
            logging.info("Gráfica de importancia de características mostrada exitosamente")
            
            # Show table with actual values
            logging.info("Mostrando tabla de valores de importancia")
            st.subheader("Valores de Importancia")
            st.dataframe(importance_df)
            logging.info("Tabla de importancia de características mostrada exitosamente")
        except Exception as e:
            error_msg = f"Error al visualizar importancia de características: {e}"
            st.error(f"❌ {error_msg}")
            logging.error(error_msg)

    @staticmethod
    def plot_confusion_matrix(y_test, y_pred, df_original):
        """Muestra la matriz de confusión para evaluar el modelo."""
        logging.info("Iniciando generación de matriz de confusión")
        
        if y_test is None or y_pred is None or df_original is None:
            error_msg = "Datos insuficientes para generar matriz de confusión"
            st.error(f"❌ {error_msg}.")
            logging.error(error_msg)
            return
            
        try:
            st.header("🧩 Matriz de Confusión")
            
            # Get unique classes
            logging.info("Obteniendo clases únicas para matriz de confusión")
            classes = df_original["Pais Destino"].unique()
            logging.info(f"Identificadas {len(classes)} clases únicas")
            
            # Compute confusion matrix
            logging.info("Calculando matriz de confusión")
            cm = confusion_matrix(y_test, y_pred)
            
            # Normalize confusion matrix
            logging.info("Normalizando matriz de confusión")
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Create heatmap
            logging.info("Generando visualización de matriz de confusión")
            fig = px.imshow(
                cm_normalized,
                labels=dict(x="Predicción", y="Real", color="Proporción"),
                x=classes,
                y=classes,
                text_auto=True,
                color_continuous_scale="Blues"
            )
            fig.update_layout(title="Matriz de Confusión Normalizada")
            st.plotly_chart(fig)
            logging.info("Matriz de confusión mostrada exitosamente")
            
            # Show classification report
            logging.info("Generando reporte de clasificación")
            st.subheader("Reporte de Clasificación")
            report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            logging.info("Reporte de clasificación mostrado exitosamente")
        except Exception as e:
            error_msg = f"Error al generar matriz de confusión: {e}"
            st.error(f"❌ {error_msg}")
            logging.error(error_msg)

    @staticmethod
    def plot_predictions_distribution(y_test, y_pred, df_original):
        """Muestra un histograma de predicciones del modelo."""
        logging.info("Iniciando visualización de distribución de predicciones")
        
        if y_test is None or y_pred is None or df_original is None:
            error_msg = "Datos insuficientes para visualizar distribución de predicciones"
            st.error(f"❌ {error_msg}.")
            logging.error(error_msg)
            return
            
        try:
            st.header("📊 Distribución de Predicciones")
            logging.info("Mapeando índices a nombres de países")

            # Map indices to actual country names
            classes = df_original["Pais Destino"].unique()
            y_test_names = [classes[i] for i in y_test]
            y_pred_names = [classes[i] for i in y_pred]
            
            logging.info("Creando DataFrame para visualización de predicciones")
            df_pred = pd.DataFrame({"Real": y_test_names, "Predicción": y_pred_names})
            
            # Distribution of predictions
            logging.info("Generando histograma de distribución de predicciones")
            fig1 = px.histogram(
                df_pred,
                x="Predicción",
                title="Distribución de Predicciones por País Destino",
                color="Predicción",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            st.plotly_chart(fig1)
            logging.info("Histograma de predicciones mostrado exitosamente")
            
            # Accuracy by country
            logging.info("Calculando precisión por país")
            st.subheader("Precisión por País")
            accuracy_by_country = df_pred.groupby("Real").apply(
                lambda x: (x["Real"] == x["Predicción"]).mean()
            ).reset_index(name="Precisión")
            
            logging.info("Generando gráfica de barras de precisión por país")
            fig2 = px.bar(
                accuracy_by_country,
                x="Real",
                y="Precisión",
                title="Precisión del Modelo por País",
                color="Precisión",
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig2)
            logging.info("Gráfica de precisión por país mostrada exitosamente")
        except Exception as e:
            error_msg = f"Error al visualizar distribución de predicciones: {e}"
            st.error(f"❌ {error_msg}")
            logging.error(error_msg)

    @staticmethod
    def feature_contribution(modelo, X_test, feature_names):
        """Visualiza la contribución de características para ejemplos individuales."""
        logging.info("Iniciando análisis de contribución de características")
        
        if modelo is None or X_test is None or feature_names is None:
            error_msg = "Datos insuficientes para analizar contribución de características"
            st.error(f"❌ {error_msg}.")
            logging.error(error_msg)
            return
            
        try:
            st.header("🔍 Contribución de Características (Muestra)")
            
            # Select a random sample to explain
            logging.info("Seleccionando muestras aleatorias para análisis")
            if len(X_test) > 5:
                sample_indices = np.random.choice(len(X_test), 5, replace=False)
                logging.info(f"Seleccionadas 5 muestras aleatorias de {len(X_test)} disponibles")
            else:
                sample_indices = range(len(X_test))
                logging.info(f"Seleccionadas todas las {len(X_test)} muestras disponibles")
            
            for idx in sample_indices:
                logging.info(f"Analizando muestra #{idx+1}")
                sample = X_test.iloc[idx:idx+1]
                
                # Get the feature contributions for this prediction
                logging.info("Calculando contribuciones de características para la predicción")
                contribs = modelo.get_booster().predict(xgb.DMatrix(sample), pred_contribs=True)
                
                # The last column is the bias/base value, so we exclude it
                feature_contribs = contribs[0][:-1]
                
                # Create a DataFrame for visualization
                logging.info("Creando DataFrame para visualización de contribuciones")
                contrib_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Contribution': feature_contribs
                }).sort_values('Contribution', ascending=False)
                
                st.subheader(f"Ejemplo #{idx+1}")
                
                # Display the sample values
                logging.info("Mostrando valores de características de la muestra")
                st.write("Valores de características:")
                sample_display = pd.DataFrame(sample.values, columns=feature_names)
                st.dataframe(sample_display)
                
                # Plot the contributions
                logging.info("Generando gráfica de contribuciones para la muestra")
                fig = px.bar(
                    contrib_df,
                    x='Contribution',
                    y='Feature',
                    orientation='h',
                    title=f"Contribución de Características (Ejemplo #{idx+1})",
                    color='Contribution',
                    color_continuous_scale='RdBu',
                    color_continuous_midpoint=0
                )
                st.plotly_chart(fig)
                logging.info(f"Análisis de contribución para muestra #{idx+1} completado exitosamente")
        except Exception as e:
            error_msg = f"Error al analizar contribución de características: {e}"
            st.error(f"❌ {error_msg}")
            logging.error(error_msg)

    @staticmethod
    def show_results():
        """Ejecuta todas las funciones de análisis y visualización."""
        logging.info("Iniciando proceso completo de análisis y visualización")
        
        try:
            # Apply custom CSS
            logging.info("Aplicando estilos CSS personalizados")
            st.markdown("""
            <style>
                .custom-h3 {
                    color: #1E88E5;
                    font-size: 1.8rem;
                    text-align: center;
                }
                .custom-h4 {
                    color: #26A69A;
                    font-size: 1.4rem;
                }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown('<h3 class="custom-h3">Predicción de Destino de Exportación de Fertilizantes 🌱</h3>', unsafe_allow_html=True)
            logging.info("Iniciando carga de datos para análisis")
            df = analysis.load_data()
            if df is None:
                logging.error("Proceso terminado: no se pudieron cargar los datos")
                return

            logging.info("Preparando datos para entrenamiento del modelo")
            X = df.drop(columns=["Pais Destino"])
            y = df["Pais Destino"]
            logging.info(f"Datos preparados: X: {X.shape}, y: {len(y)}")

            logging.info("Iniciando entrenamiento del modelo")
            modelo, resultados, (X_test, y_test, y_pred) = analysis.train_model(X, y)
            if modelo is None:
                logging.error("Proceso terminado: no se pudo entrenar el modelo")
                return

            logging.info("Ejecutando análisis de resultados")
            analysis.analyze_results(modelo, X_test, y_test)
            
            logging.info("Generando visualización de curva de pérdida")
            analysis.plot_loss(resultados)
            
            logging.info("Analizando importancia de características")
            analysis.plot_feature_importance(modelo, X)
            
            logging.info("Generando matriz de confusión")
            analysis.plot_confusion_matrix(y_test, y_pred, df)
            
            logging.info("Analizando distribución de predicciones")
            analysis.plot_predictions_distribution(y_test, y_pred, df)
            
            logging.info("Analizando contribución de características")
            analysis.feature_contribution(modelo, X_test, X.columns)
            
            logging.info("Proceso completo de análisis y visualización finalizado exitosamente")
        except Exception as e:
            error_msg = f"Error durante el proceso de análisis: {e}"
            st.error(f"❌ {error_msg}")
            logging.error(error_msg)


class app:
    def run(self):
        """Función principal para ejecutar la aplicación sin main()"""
        logging.info("Iniciando aplicación")
        
        try:
            # Apply custom CSS
            logging.info("Aplicando estilos CSS personalizados")
            st.markdown("""
            <style>
                .custom-h3 {
                    color: #1E88E5;
                    font-size: 1.8rem;
                    text-align: center;
                }
                .custom-h4 {
                    color: #26A69A;
                    font-size: 1.4rem;
                }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown('<h3 class="custom-h3">Análisis de Datos de Fertilizantes 🌱</h3>', unsafe_allow_html=True)
            st.write("")
            
            # Use the static method from analysis class to load processed data
            logging.info("Cargando datos procesados")
            df = analysis.load_data()
            if df is None:
                logging.error("Aplicación terminada: no se pudieron cargar los datos")
                return

            # Display information about the processed data
            logging.info("Verificando duplicados en los datos")
            duplicados_existentes = df.duplicated().sum() > 0
            logging.info(f"Duplicados encontrados: {df.duplicated().sum() if duplicados_existentes else 0}")
            
            logging.info("Mostrando muestra de datos")
            st.dataframe(df.head(6))
            st.write("")
            
            # Data summary
            logging.info("Generando resumen de datos")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Filas", df.shape[0])
            with col2:
                st.metric("Columnas", df.shape[1])
            with col3:
                nulos = df.isnull().sum().sum()
                st.metric("Valores nulos", nulos)
                logging.info(f"Resumen de datos: {df.shape[0]} filas, {df.shape[1]} columnas, {nulos} valores nulos")
                
            st.write(f"Duplicados: {'Sí' if duplicados_existentes else 'No'}")
            
            # Backup the processed data
            if st.button("Respaldar datos"):
                logging.info("Iniciando respaldo de datos")
                try:
                    df.to_csv("backup_data_results.csv", index=False)
                    st.success("✅ Datos respaldados en 'backup_data_results.csv'.")
                    logging.info("Datos respaldados exitosamente en 'backup_data_results.csv'")
                except Exception as e:
                    error_msg = f"Error al respaldar datos: {e}"
                    st.error(f"❌ {error_msg}")
                    logging.error(error_msg)
            
            st.write(" *********************** ")
            st.markdown('<h4 class="custom-h4">Limpieza de los datos 🧹📊</h4>', unsafe_allow_html=True)
            
            # Data cleaning options
            logging.info("Configurando opciones de limpieza de datos")
            cleaning_options = st.multiselect(
                "Seleccione opciones de limpieza:",
                ["Eliminar duplicados", "Eliminar valores nulos"],
                default=["Eliminar duplicados", "Eliminar valores nulos"] if duplicados_existentes or df.isnull().sum().sum() > 0 else []
            )
            logging.info(f"Opciones de limpieza seleccionadas: {cleaning_options}")
            
            # Clean data based on selected options
            df_limpio = df.copy()
            if "Eliminar duplicados" in cleaning_options and duplicados_existentes:
                logging.info("Eliminando filas duplicadas")
                filas_antes = df_limpio.shape[0]
                df_limpio = df_limpio.drop_duplicates()
                filas_eliminadas = filas_antes - df_limpio.shape[0]
                st.success(f"✅ Se eliminaron {filas_eliminadas} filas duplicadas.")
                logging.info(f"Se eliminaron {filas_eliminadas} filas duplicadas")
                
            if "Eliminar valores nulos" in cleaning_options and df.isnull().sum().sum() > 0:
                logging.info("Eliminando valores nulos")
                nulos_antes = df_limpio.isnull().sum().sum()
                df_limpio = df_limpio.dropna()
                st.success(f"✅ Se eliminaron {nulos_antes} valores nulos.")
                logging.info(f"Se eliminaron {nulos_antes} valores nulos")
            
            # Display cleaned data
            if len(cleaning_options) > 0:
                logging.info("Mostrando datos después de la limpieza")
                st.subheader("Datos después de la limpieza:")
                st.dataframe(df_limpio.head(6))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Filas originales", df.shape[0])
                with col2:
                    st.metric("Filas después de limpieza", df_limpio.shape[0], 
                            delta=df_limpio.shape[0] - df.shape[0])
                logging.info(f"Datos después de limpieza: {df_limpio.shape[0]} filas, {df_limpio.shape[1]} columnas")
            
            # Save cleaned data to session state
            logging.info("Guardando datos en session_state")
            if len(cleaning_options) > 0:
                st.session_state["df_limpio"] = df_limpio
                logging.info("Datos limpios guardados en session_state")
            else:
                st.session_state["df_limpio"] = df
                logging.info("Datos originales guardados en session_state (sin limpieza)")
            
            # Option to run analysis on the cleaned data
            st.write(" *********************** ")
            st.markdown('<h4 class="custom-h4">Análisis y Modelado 📈🔍</h4>', unsafe_allow_html=True)
            
            if st.button("Ejecutar Análisis y Modelado"):
                logging.info("Iniciando proceso de análisis y modelado")
                try:
                    # We'll use either the cleaned data or the original if no cleaning was done
                    if "df_limpio" in st.session_state:
                        logging.info("Ejecutando análisis con los datos disponibles")
                        analysis.show_results()
                        logging.info("Proceso de análisis y modelado completado")
                    else:
                        error_msg = "Datos no disponibles para análisis"
                        st.error(f"❌ {error_msg}.")
                        logging.error(error_msg)
                except Exception as e:
                    error_msg = f"Error durante el análisis: {str(e)}"
                    st.error(f"❌ {error_msg}")
                    logging.error(error_msg)
        except Exception as e:
            error_msg = f"Error general en la aplicación: {e}"
            st.error(f"❌ {error_msg}")
            logging.error(error_msg)
            logging.error(f"Detalles del error:\n{str(e)}")
