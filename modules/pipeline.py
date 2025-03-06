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
from sklearn.preprocessing import StandardScaler
import os
from cryptography.fernet import Fernet


#Configuración de login para auditorias
logging.basicConfig(filename="audit_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

# Cargar los roles del json
with open("config.json", "r") as f:
    roles = json.load(f)

class eda:
     def __init__(self):
          self.df = None
          self.clave = None
          self.cipher = None
          self._load_or_generate_key()

     def _load_or_generate_key(self):
        """Carga la clave de encriptación desde un archivo o genera una nueva."""
        key_path = "encryption_key.key"
        if os.path.exists(key_path):
            with open(key_path, "rb") as file:
                self.clave = file.read()
        else:
            self.clave = Fernet.generate_key()
            with open(key_path, "wb") as file:
                file.write(self.clave)
        self.cipher = Fernet(self.clave)    

     def encrypt_column(self, column_name):
        """
        Encripta una columna del dataset si existe.
        """
        self._check_df()
        if column_name not in self.df.columns:
            raise KeyError(f"La columna '{column_name}' no existe en el dataset.")

        self.df[column_name + "_Encriptado"] = self.df[column_name].apply(
            lambda x: self.cipher.encrypt(str(x).encode()).decode()
        )
        logging.info(f"Columna '{column_name}' encriptada exitosamente.")

     def save_encrypted_data(self, filename="data_procesada_encriptada.csv"):
        """Guarda el dataset con los datos encriptados."""
        self._check_df()
        self.df.to_csv(filename, index=False)
        logging.info(f"Datos encriptados guardados en {filename}.")

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
                    self.df = pd.read_csv(path, sep=";", decimal=".")
                else:
                    with open(path, 'r', encoding='utf-8') as file:
                        sample = file.read(800)  # Leer una muestra del archivo
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

     def check_missing_values(self):
        """Revisa los valores faltantes en el DataFrame."""
        self._check_df()
        missing_values = self.df.isnull().sum()
        st.markdown("<u> Valores faltantes por columna: </u>", unsafe_allow_html=True)
        if missing_values.sum() == 0:
            st.write("¡Todos los valores están completos! No hay valores faltantes.")
        else:
            st.write(missing_values)

     def identify_duplicates(self):
        """Identifica duplicados en el DataFrame."""
        self._check_df()
        duplicated = self.df.duplicated()
        total_duplicates = duplicated.sum()

        st.markdown("<u> Valores duplicados: </u>", unsafe_allow_html=True)
        if total_duplicates > 0:
            st.write(f"Se encontraron {total_duplicates} registros duplicados.")
            st.write("Extracto de los datos duplicados:")
            st.write(self.df[duplicated].head())
            return True # hay duplicados
        else:
            st.write("No se encontraron registros duplicados en el dataset.")
            return False # no hay duplicados

     def analisisNumerico(self):
        """
        Verificar si hay columnas numéricas, mostrar un conteo y el nombre de las columnas.
        """
        self._check_df() 
        st.markdown("<u> Columna(s) numérica(s) </u>", unsafe_allow_html=True)
        try:
            numeric_columns = self.df.select_dtypes(include=["number"]).columns

            if numeric_columns.size > 0:
                st.write(f"El dataset contiene {len(numeric_columns)} columna(s) numérica(s).")
                st.write("Estas son las columnas numéricas:")
                st.write(list(numeric_columns))
            else:
                st.warning("El dataset no contiene columnas numéricas.")


            return list(numeric_columns)

        except Exception as e:
            st.error(f"Ocurrió un error al analizar las columnas numéricas: {e}")
            logging.error(f"Error en analisisNumerico: {e}")
            return []

     def check_data_types(self):
        """Muestra los tipos de datos de cada columna del DataFrame."""
        self._check_df()
        st.markdown("<u> Tipos de datos por columna: </u>", unsafe_allow_html=True)
        return self.df.dtypes

# --------------------  Data cleanning  -------------------- #

     def drop_duplicates(self):
        """Elimina duplicados del DataFrame."""
        self._check_df()
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)
        st.markdown("<u> Registros duplicados: </u> ", unsafe_allow_html=True)
        st.write(f"Duplicados eliminados. Se pasó de {before} a {after} registros.")
 
     def _transform_column(self, column_name, transform_type, all_columns=False):
        """
        Transforma una columna en variables dummy o valores categóricos codificados.
        Si `all_columns` es True, aplica la transformación a todas las columnas categóricas.
        """
        try:
            if all_columns:
                # Transformar todas las columnas categóricas
                categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns
                for col in categorical_cols:
                    if transform_type == "Variables dummy":
                        dummies = pd.get_dummies(self.df[col], prefix=col)
                        self.df = pd.concat([self.df.drop(columns=[col]), dummies], axis=1)
                    elif transform_type == "Códigos categóricos":
                        self.df[col] = pd.Categorical(self.df[col]).codes
                st.success(f"Todas las columnas categóricas fueron transformadas a '{transform_type}'.")
            else:
                # Transformar una columna específica
                if transform_type == "Variables dummy":
                    dummies = pd.get_dummies(self.df[column_name], prefix=column_name)
                    self.df = pd.concat([self.df.drop(columns=[column_name]), dummies], axis=1)
                elif transform_type == "Códigos categóricos":
                    self.df[column_name] = pd.Categorical(self.df[column_name]).codes
                st.success(f"La columna '{column_name}' fue transformada usando '{transform_type}'.")
        except Exception as e:
            st.warning(f"No se pudo transformar la columna '{column_name}' o las columnas categóricas: {e}")

     def type_transform(self):
        """
        Permite al usuario seleccionar cómo transformar las columnas categóricas: 
        variables dummy o códigos categóricos. Aplica la transformación a una columna específica
        o a todas las columnas categóricas.
        """
        self._check_df()
        try:
            st.markdown("<u>Transformar columnas categóricas</u>", unsafe_allow_html=True)
            transform_options = ["Variables dummy", "Códigos categóricos"]
            
            # Escoge entre transformar una columna o todas las columnas
            option = st.radio("¿Deseas transformar una columna específica o todas las categóricas?", 
                            ("Columna específica", "Todas las columnas categóricas"))

            # Selecciona el tipo de transformación (dummy o categórico)
            selected_transform = st.selectbox("Selecciona el tipo de transformación:", transform_options)

            if option == "Columna específica":
                column_name = st.selectbox("Selecciona la columna a transformar:", 
                                        self.df.select_dtypes(include=["object", "category"]).columns)
                if st.button("Transformar columna"):
                    self._transform_column(column_name, selected_transform)

            elif option == "Todas las columnas categóricas":
                if st.button("Transformar todas las columnas"):
                    self._transform_column(None, selected_transform, all_columns=True)

        except Exception as e:
            st.error(f"Error al intentar transformar las columnas categóricas: {e}")

     def describe_data(self):
        """Genera estadísticas descriptivas básicas del DataFrame."""
        self._check_df()
        st.markdown("<u> Estadística descriptiva del conjunto de datos: </u>", unsafe_allow_html=True)
        return self.df.describe()            

     def delete_irrelevant_values(self):
        """
        Elimina columnas del DataFrame que tienen un único valor único (sin variabilidad).
        """
        self._check_df()  
        try:
            columnas_constantes = [col for col in self.df.columns if self.df[col].nunique() <= 1]
            
            if columnas_constantes:
                st.markdown("<u> Datos constantes que serán eliminados. </u>", unsafe_allow_html=True)
                st.write(f"Se encontraron {len(columnas_constantes)} columna(s) constante(s) que serán eliminadas:")
                st.write(columnas_constantes)
                
                self.df = self.df.drop(columns=columnas_constantes)

                st.success("Columnas constantes eliminadas exitosamente.")
            else:
                st.write("No se encontraron columnas constantes en el DataFrame.")

        except Exception as e:
            st.error(f"Un error ocurrió mientras se eliminaban las columnas constantes: {e}")
            logging.error(f"Error al eliminar columnas constantes: {e}")

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
        corr = self.df.corr()
        st.markdown("<u> Matriz de correlación: </u>", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Mapa de calor de correlación")
        st.pyplot(fig)

     def outlier_detection(self):
        """Detecta posibles valores atípicos usando diagramas de caja."""
        self._check_df()
        numeric_cols = self.df.select_dtypes(include=["number"]).columns

        if numeric_cols.empty:
            st.write("No hay columnas numéricas para graficar.")
            return
        
        st.markdown("<u> Diagramas de caja (Boxplots):</u>", unsafe_allow_html=True)
        
        num_cols_per_row = 3
        col_chunks = [numeric_cols[i:i + num_cols_per_row] for i in range(0, len(numeric_cols), num_cols_per_row)]

        for chunk in col_chunks:
            cols = st.columns(len(chunk))
            for col, column_name in zip(cols, chunk):
                with col:
                    fig, ax = plt.subplots()
                    sns.boxplot(x=self.df[column_name], ax=ax)
                    ax.set_title(f"Valores atípicos en {column_name}")
                    st.pyplot(fig)
     
     def advanced_outlier_detection(self):
        """Detecta valores atípicos utilizando el rango intercuartil (IQR)."""
        self._check_df()
        numeric_cols = self.df.select_dtypes(include=["number"]).columns

        st.markdown("<u> Detección avanzada de valores atípicos:</u>", unsafe_allow_html=True)
        for col in numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            st.write(f"Valores atípicos en {col}: {len(outliers)} registros.")
            st.write(outliers)

     def standardize_data(self):
        """
        Estandariza las columnas numéricas del DataFrame.
        Transforma los datos para que tengan media 0 y desviación estándar 1.
        """
        self._check_df()
        numeric_cols = self.df.select_dtypes(include=["number"]).columns

        if numeric_cols.empty:
            st.warning("No hay columnas numéricas para estandarizar.")
            return

        try:
            scaler = StandardScaler()
            self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
            st.success("Las columnas numéricas han sido estandarizadas con éxito.")
            st.write(f"Columnas estandarizadas: {list(numeric_cols)}")
        except Exception as e:
            st.error(f"Error al estandarizar las columnas: {e}")




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
            st.markdown('<h3 class="custom-h3">Análisis de Datos de Fertilizantes 🌱</h3>', unsafe_allow_html=True)
            st.write("")
            eda_instance = eda()
            backup_instance = fileHandler()

            #Extraccion desde GitHub
            path = "https://raw.githubusercontent.com/CSMore/Datasets/refs/heads/main/Fertilizantes_CR_2024.csv"

            try:
                eda_instance.read_dataset(path)
                df = eda_instance.df
                duplicados_existentes = eda_instance.identify_duplicates() #almacenar el resultado

                st.dataframe(eda_instance.show_head(6))
                st.write("")
                eda_instance.check_missing_values()
                st.write("")
                eda_instance.identify_duplicates()
                st.write("")
                eda_instance.analisisNumerico()
                st.write(eda_instance.check_data_types())

                fileHandler.backup_data(eda_instance.df, "backup_data.csv")
                st.write("***** Datos respaldados en 'Backup_data.csv'. *****")   

                st.write(" ********************************************************************* ")
                st.markdown('<h4 class="custom-h4">Limpieza de los datos 🧹📊</h4>', unsafe_allow_html=True)
                if duplicados_existentes:
                    eda_instance.drop_duplicates()
                st.write("")
                eda_instance.type_transform()
                st.write(eda_instance.check_data_types())
                eda_instance.delete_irrelevant_values()
                st.write(eda_instance.describe_data())
                eda_instance.plot_distributions()
                eda_instance.correlation_matrix()
                st.write("")
                eda_instance.outlier_detection()
                eda_instance.advanced_outlier_detection()
                eda_instance.standardize_data()
                
                logging.info("Datos extraídos con éxito.")
            except Exception as e:
                st.error("Error al cargar los datos.")
                logging.error(f"Error al cargar los datos: {e}")
            
