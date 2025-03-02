import streamlit as st
from streamlit_option_menu import option_menu
import modules.pipeline as pipeline  # Importamos Pipeline desde modules
import modules.login as login        # Importamos Login desde modules

st.set_page_config(
    page_title="Testing",
    layout="wide"
)

def app_control():
    # Botón "Cerrar Sesión" justo antes del menú horizontal
    if st.button("Cerrar Sesión", key="logout_button"):
        st.session_state.page = "login"  # Cambiamos de nuevo a la página de login
        st.rerun()         # Forzamos un refresco inmediato para aplicar el cambio

    # Menú horizontal en la parte superior
    selected = option_menu(
        menu_title="",  # Sin título en el menú
        options=["Pipeline", "Resultados"],  # Opciones del menú
        icons=["bar-chart", "rocket-takeoff"],  # Iconos de las opciones
        menu_icon="cast",  # Icono principal del menú (opcional)
        default_index=0,  # Página inicial
        orientation="horizontal",  # Menú horizontal
        styles={
            "container": {"padding": "0", "margin": "0"},
            "icon": {"font-size": "16px"},
            "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "--hover-color": "#d9d9d9"},
            "nav-link-selected": {"background-color": "#384B70", "color": "white"},
        }
    )

    # Lógica de las opciones del menú
    if selected == "Pipeline":
        pipeline.app()  # Llamamos al Pipeline
    elif selected == "Resultados":
        st.write("Aquí se muestran los resultados 🚀")  # Página de resultados

def main():
    # Inicializar el estado de la página si no está definido
    if "page" not in st.session_state:
        st.session_state.page = "login"  # Página inicial predeterminada

    # Navegar entre login y el menú general
    if st.session_state.page == "login":
        login.access()  # Llamamos a la función de login
    elif st.session_state.page == "app_control":
        app_control()   # Mostramos el menú principal


if __name__ == "__main__":
    main()  # Ejecutar la lógica principal

