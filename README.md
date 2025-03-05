# 📌 Análisis Temporal de Exportaciones de Fertilizantes con XGBoost
![Fertilizer Export Prediction](time series.png)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/) [![XGBoost](https://img.shields.io/badge/XGBoost-1.6%2B-green)](https://xgboost.ai/) [![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)

## 👥 Participantes
- **Kristhel Porras**
- **Carolina Salas**
---
## 🎯 Objetivo
Predecir los destinos de exportación de fertilizantes de Costa Rica utilizando análisis temporal con **XGBoost**. El proyecto busca identificar patrones y tendencias en las exportaciones de fertilizantes, aprovechando técnicas avanzadas de machine learning para una toma de decisiones más precisa.

---
## 📁 Estructura del Proyecto
```
📂 modules/                # Módulos del proyecto
│   ├── init.py        # Inicialización del paquete
│   ├── login.py           # Autenticación y control de acceso
│   ├── pipeline.py        # Procesamiento de datos y entrenamiento del modelo
│   ├── results.py         # Análisis y visualización de resultados
📂 data/                   # Almacenamiento de datos
│   ├── Fertilizantes_CR_En-Feb_2025.xlsx  # Dataset principal
│   ├── backup_data.csv    # Respaldo de datos
📂 config/                 # Configuraciones
│   └── config.json        # Parámetros de configuración del modelo
📄 app_control.py          # Script principal de control
📄 requirements.txt        # Dependencias del proyecto
📄 README.md               # Documentación del proyecto
```

---

## 🚀 Cómo Ejecutar el Proyecto
### 🔧 Instalación
```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
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
