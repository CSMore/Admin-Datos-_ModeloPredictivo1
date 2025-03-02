# 📌 Análisis Temporal con XGBoost

![XGBoost Time Series](https://via.placeholder.com/1200x300?text=An%C3%A1lisis+Temporal+con+XGBoost)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/) [![XGBoost](https://img.shields.io/badge/XGBoost-1.6%2B-green)](https://xgboost.ai/) [![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)

## 👥 Participantes
- **Kristhel Porras**
- **Carolina Salas**

---

## 🎯 Objetivo
Este proyecto busca evaluar cómo el rendimiento de los cultivos cambia a lo largo del tiempo utilizando el dataset [Crop Yield Prediction Dataset](https://www.kaggle.com/datasets). Se emplea **XGBoost** para modelar datos temporales y se visualizan los resultados en **Streamlit**.

---

## 📁 Estructura del Proyecto
```
📂 data/                   # Datos crudos y preprocesados
📂 notebooks/              # Jupyter Notebooks para exploración
📂 src/                    # Código fuente
│   ├── extraction.py      # Extracción de datos (API, web scraping, etc.)
│   ├── storage.py         # Almacenamiento seguro en la nube
│   ├── preprocessing.py   # Limpieza y transformación de datos
│   ├── model.py           # Implementación del modelo XGBoost
│   ├── visualization.py   # Generación de gráficos y análisis
│   ├── streamlit_app.py   # Aplicación interactiva en Streamlit
📄 requirements.txt        # Dependencias necesarias
📄 README.md               # Este archivo
```

---

## 🚀 Cómo Ejecutar el Proyecto
### 🔧 Instalación
```bash
pip install -r requirements.txt
```

### 📥 Extracción y Procesamiento de Datos
```bash
python src/extraction.py   # Extraer datos desde API / Web scraping
python src/preprocessing.py  # Limpieza y transformación de datos
```

### 🎯 Entrenamiento del Modelo
```bash
python src/model.py
```

### 🌐 Ejecutar la Aplicación Web
```bash
streamlit run src/streamlit_app.py
```

---

## 📖 Tabla de Contenidos
1️⃣ [Arquitectura del Data Pipeline](#)  
2️⃣ [Integración del Modelo de IA](#)  
3️⃣ [Seguridad, Criptografía y Limpieza de Datos](#)

---
