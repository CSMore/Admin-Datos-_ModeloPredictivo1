import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix, 
    classification_report
)
import plotly.express as px
import plotly.graph_objs as go

def cargar_datos():
    """
    Carga los datos desde un archivo local
    
    Returns:
    - DataFrame con los datos de fertilizantes
    """
    try:
        # Asegúrate de que el archivo esté en la misma carpeta que tu script
        df = pd.read_excel('Fertilizantes_CR_En-Feb_2025.xlsx')
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None

def preprocesar_datos(df):
    """
    Preprocesa los datos para el modelo XGBoost
    
    Args:
    - df (DataFrame): Datos originales
    
    Returns:
    - X (array): Variables predictoras
    - y (array): Variable objetivo
    - scaler: Objeto de escalamiento
    """
    # Convertir fecha a características numéricas
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Año'] = df['Fecha'].dt.year
    df['Mes'] = df['Fecha'].dt.month

    # Codificar variables categóricas
    label_encoders = {}
    categoricas = [
        'Tipo', 'Nombre Comercial', 'Unidad', 'Modalidad', 
        'Importador Exportador', 'Pais Origen', 'Pais Destino', 
        'Puerto Ingreso', 'Componente IAGT'
    ]
    
    for columna in categoricas:
        le = LabelEncoder()
        df[columna + '_encoded'] = le.fit_transform(df[columna].astype(str))
        label_encoders[columna] = le

    # Seleccionar características para el modelo
    caracteristicas = [
        'Año', 'Mes', 'Cantidad', 'Peso', 'Valor',
        'Tipo_encoded', 'Nombre Comercial_encoded', 'Unidad_encoded', 
        'Modalidad_encoded', 'Importador Exportador_encoded', 
        'Pais Origen_encoded', 'Pais Destino_encoded', 
        'Puerto Ingreso_encoded', 'Componente IAGT_encoded'
    ]

    # Variable objetivo (Pais Destino)
    X = df[caracteristicas]
    y = df['Pais Destino_encoded']  # Predecir país de destino codificado
    
    # Escalar características
    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X)
    
    return X_escalado, y, scaler, label_encoders, df

def entrenar_modelo(X, y):
    """
    Entrena un modelo XGBoost
    
    Args:
    - X (array): Variables predictoras escaladas
    - y (array): Variable objetivo
    
    Returns:
    - Modelo entrenado
    - Métricas de rendimiento
    """
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configurar modelo XGBoost para clasificación
    modelo = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        random_state=42
    )
    
    # Entrenar modelo
    modelo.fit(X_train, y_train)
    
    # Predicciones
    y_pred = modelo.predict(X_test)
    
    # Calcular métricas
    metricas = {
        'Precisión': modelo.score(X_test, y_test),
        'Exactitud': modelo.score(X_test, y_test)
    }
    
    return modelo, metricas, (X_test, y_test, y_pred)

def mostrar_resultados():
    """
    Módulo principal para mostrar resultados del modelo
    """
    # Paleta de colores personalizada (tonos celestes y verdes)
    color_palette = {
        'matriz_confusion': '#20B2AA',  # Light Sea Green
        'importancia': '#2E8B57',       # Sea Green
        'metricas_fondo': '#E0F2F1',    # Light Teal
        'metricas_texto': '#00695C'     # Dark Teal
    }

    st.title("🌱 Predicción de Destino de Exportación de Fertilizantes")
    
    # Cargar datos
    df = cargar_datos()
    if df is None:
        return
    
    # Preprocesar datos
    X, y, scaler, label_encoders, df_original = preprocesar_datos(df)
    
    # Entrenar modelo
    modelo, metricas, (X_test, y_test, y_pred) = entrenar_modelo(X, y)
    
    # Sección de Métricas con fondo celeste
    st.header("Rendimiento del Modelo")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Precisión", f"{metricas['Precisión']:.2%}")
    with col2:
        st.metric("Exactitud", f"{metricas['Exactitud']:.2%}")
    
    # Decodificar predicciones y valores reales
    paises_destino = label_encoders['Pais Destino'].classes_
    
    # Mostrar Matriz de Confusión con color verde agua
    st.header("Matriz de Confusión")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, 
        labels=dict(x="Predicción", y="Real", color="Frecuencia"),
        x=paises_destino,
        y=paises_destino,
        title="Matriz de Confusión de Países de Destino",
        color_continuous_scale=[[0, color_palette['matriz_confusion']], [1, '#00BFA5']]
    )
    st.plotly_chart(fig_cm)
    
    # Importancia de características con tonos verdes
    importancia = modelo.feature_importances_
    caracteristicas = [
        'Año', 'Mes', 'Cantidad', 'Peso', 'Valor',
        'Tipo', 'Nombre Comercial', 'Unidad', 
        'Modalidad', 'Importador Exportador', 'Pais Origen', 
        'Pais Destino', 'Puerto Ingreso', 'Componente IAGT'
    ]
    
    fig_importancia = px.bar(
        x=caracteristicas, 
        y=importancia, 
        title='Importancia de Características para Predecir Destino',
        color_discrete_sequence=[color_palette['importancia']]
    )
    st.plotly_chart(fig_importancia)
    
    # Informe de Clasificación
    st.header("Informe de Clasificación")
    reporte = classification_report(y_test, y_pred, target_names=paises_destino)
    st.text(reporte)

# Opcional: Si quieres ejecutar directamente este módulo
if __name__ == "__main__":
    mostrar_resultados()
