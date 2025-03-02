# Importacion de librearias
import streamlit as st
import joblib
import pandas as pd
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


#Configuración de login para auditorias
logging.basicConfig(filename="audit_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

# Cargar los roles del json
with open("config.json", "r") as f:
    roles = json.load(f)

# Funcion para verificiar permiso
def check_access(role):
    """Esta funcion sirve para verificar los permisos de los roles """
    if role not in roles:
        logging.warning(f"Acceso denegado: Rol {role} no registrado")
        raise PermissionError("Acceso denegado: Rol no registrado")
    return roles[role]["access_level"]


# Funcion para hacer backups
def backup_data (df, filename="backup_data.csv"):
    df.to_csv(filename, index=False)
    logging.info(f"Backup realizado con exito en : {filename}")


def save_data(df, filename="data_procesada.csv"):
    """Guarda los datos procesados en un archivo CSV."""
    df.to_csv(filename, index=False)
    logging.info(f"Datos guardados con éxito en {filename}")

# Database conexion with Azure "In progress"
'''cnn = st.experimental_connection('data_admin_db', type='sql')
tbl_access = cnn.query('select * from tbl_access')
st.dataframe(tbl_access)'''


def app():
    cont = 1