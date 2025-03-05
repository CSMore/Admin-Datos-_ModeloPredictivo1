# Importacion de librearias
import streamlit as st
import joblib
import pandas as pd
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import csv


#Configuración de login para auditorias
logging.basicConfig(filename="audit_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

# Cargar los roles del json
with open("config.json", "r") as f:
    roles = json.load(f)

class eda:
     def __init__(self):
          self.df = None

     def _check_df(self):
        """Verifica si el DataFrame está cargado."""
        if self.df is None:
            logging.warning("El DataFrame no ha sido cargado. Usa 'read_dataset()' primero.")
            raise ValueError("El DataFrame está vacío.")
        
     def read_dataset(self, path):
            """
            Cargar el dataset e identificar el separador.
            """
            separa = None
            try:
                if path.startswith("http"):
                    self.df = pd.read_csv(path, sep=",", decimal=".")
                else:
                    with open(path, 'r', encoding='utf-8') as file:
                        sample = file.read(500)  # Leer una muestra del archivo
                        st.write("Muestra del archivo:", sample)
                        if not sample.strip():
                            raise ValueError("El archivo está vacío o no contiene suficientes datos para detectar el separador.")

                        dialect = csv.Sniffer().sniff(sample)  
                        separa = dialect.delimiter
                        st.write("Separador detectado:", separa)

                    self.df = pd.read_csv(path, sep=separa, decimal=".",header=0, index_col=None)
            
                logging.info("Datos cargados con éxito. Separador detectado: '%s'" % separa)

            except FileNotFoundError:
                logging.error(f"No se encontró el archivo en la ruta: {path}")
                raise FileNotFoundError(f"No se encontró el archivo en la ruta: {path}")

            except UnicodeDecodeError:
                logging.error("Error de codificación: intenta usar una codificación distinta como 'latin1'.")
                raise UnicodeDecodeError("Error de codificación. Verifica el archivo y la codificación.")   
            except Exception as e:
                logging.error(f"Error al cargar el archivo: {path}: {str(e)}")
                raise

     def show_head(self, n):
        """Muestra las primeras filas del DataFrame."""
        self._check_df()
        st.markdown(f"<u> Estos son los primeros {n} registros del dataset: </u>", unsafe_allow_html=True)
        return self.df.head(n)

     def describe_data(self):
        """Genera estadísticas descriptivas básicas del DataFrame."""
        self._check_df()
        st.markdown("<u> Estadística descriptiva del conjunto de datos: </u>", unsafe_allow_html=True)
        return self.df.describe()            

     def check_missing_values(self):
        """Revisa los valores faltantes en el DataFrame."""
        self._check_df()
        missing_values = self.df.isnull().sum()
        st.markdown("<u> Valores faltantes por columna: </u>", unsafe_allow_html=True)
        if missing_values.sum() == 0:
            st.write("¡Todos los valores están completos! No hay valores faltantes.")
        else:
            st.write(missing_values)

     def check_data_types(self):
        """Muestra los tipos de datos de cada columna del DataFrame."""
        self._check_df()
        st.markdown("<u> Tipos de datos por columna: </u>", unsafe_allow_html=True)
        return self.df.dtypes

     def identify_duplicates(self):
        """Identifica duplicados en el DataFrame."""
        self._check_df()
        duplicated = self.df.duplicated()
        total_duplicates = duplicated.sum()

        st.markdown("<u> Valores duplicados: </u>", unsafe_allow_html=True)
        if total_duplicates > 0:
            st.write(f"Se encontraron {total_duplicates} registros duplicados.")
            st.write("Ejemplos de duplicados:")
            st.write(self.df[duplicated].head())
            return True # hay duplicados
        else:
            st.write("No se encontraron registros duplicados en el dataset.")
            return False # no hay duplicados

     def drop_duplicates(self):
        """Elimina duplicados del DataFrame."""
        self._check_df()
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)
        st.write(f"Duplicados eliminados. Se pasó de {before} a {after} registros.")




     def process_data(self):
            """
            Procesa las columnas no numéricas convirtiéndolas a valores categóricos codificados.
            """
            self._check_df()
            non_numeric_cols = self.df.select_dtypes(exclude=["number"]).columns
            for col in non_numeric_cols:
                if self.df[col].dtype == "object":
                    self.df[col] = pd.Categorical(self.df[col])
                    self.df[col] = self.df[col].cat.codes
            logging.info("Datos procesados con éxito (columnas no numéricas transformadas).")

     def analisisNumerico(self):
        """Filtra y mantiene solo las columnas numéricas del DataFrame."""
        self._check_df()
        return self.df.select_dtypes(include=["number"])

     def analisisCompleto(self):
        """Convierte las columnas categóricas del DataFrame a variables dummy."""
        self._check_df()
        return pd.get_dummies(self.df)

     def plot_distributions(self):
        """Grafica la distribución de las columnas numéricas."""
        self._check_df()
        numeric_cols = self.df.select_dtypes(include=["number"]).columns

        if numeric_cols.empty:
            st.write("No hay columnas numéricas para graficar.")
            return
        
        st.markdown("<u> Distribuciones de variables numéricas:</u>", unsafe_allow_html=True)
        num_cols_per_row = 4 # Agrupacion de los graficos
        col_chunks = [numeric_cols[i:i + num_cols_per_row] for i in range(0, len(numeric_cols), num_cols_per_row)]

        for chunk in col_chunks:
            cols = st.columns(len(chunk))  # Creamos columnas dinámicamente según el tamaño del grupo
            for col, column_name in zip(cols, chunk):
                with col:
                    fig, ax = plt.subplots()
                    sns.histplot(self.df[column_name], kde=True, ax=ax)
                    ax.set_title(f"Distribución de {column_name}")
                    st.pyplot(fig)

     def correlation_matrix(self):
        """Muestra y grafica la matriz de correlación."""
        self._check_df()
        st.write("Grafico 2:")
        corr = self.df.corr()
        st.write("Matriz de correlación:")
        st.write(corr)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Mapa de calor de correlación")
        st.pyplot(fig)

     def outlier_detection(self):
        """Detecta posibles valores atípicos usando diagramas de caja."""
        self._check_df()
        st.write("Grafico 3:")
        numeric_cols = self.df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.boxplot(x=self.df[col], ax=ax)
            ax.set_title(f"Valores atípicos en {col}")
            st.pyplot(fig)
     

    # Funcion para verificiar permiso
     def check_access(role):
        """Esta funcion sirve para verificar los permisos de los roles """
        if role not in roles:
            logging.warning(f"Acceso denegado: Rol {role} no registrado")
            raise PermissionError("Acceso denegado: Rol no registrado")
        return roles[role]["access_level"]



class fileHandler:
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

class app:
    def main(self):
            st.markdown('<h3 class="custom-h3">Análisis de Datos de Cultivos 🌱</h3>', unsafe_allow_html=True)
            st.write("")
            eda_instance = eda()
            backup_instance = fileHandler()
            #Extraccion de datos desde un archivo CSV en GitHub
            path = "https://raw.githubusercontent.com/alitoxSB/data_pipeline/main/predictive_maintenance.csv"

            try:
                eda_instance.read_dataset(path)
                df = eda_instance.df
                st.dataframe(eda_instance.show_head(6))
                st.write("")

                if eda_instance.identify_duplicates():
                    eda_instance.drop_duplicates()
                st.write("")
                fileHandler.backup_data(eda_instance.df, "backup_data.csv")
                st.write("***** Datos respaldados en 'Backup_data.csv'. *****")   
                st.write("")                 
                st.write(eda_instance.describe_data())
                st.write("")
                eda_instance.check_missing_values()
                st.write("")
                st.write(eda_instance.check_data_types())
                ''' Hay que agregar los cambios en los datos antes de los graficos  '''


                eda_instance.plot_distributions()
                eda_instance.correlation_matrix()
                st.write("")
                eda_instance.outlier_detection()
                
                logging.info("Datos extraídos con éxito.")
            except Exception as e:
                st.error("Error al cargar los datos.")
                logging.error(f"Error al cargar los datos: {e}")
            
