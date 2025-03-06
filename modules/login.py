import streamlit as st
import json
from pathlib import Path

CONFIG_PATH = Path("config.json")

# --- Cargar estilos desde el archivo CSS ---
def load_css(file_name):
    try:
        with open(file_name, "r") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("⚠️ El archivo CSS no se encontró.")

def load_users():
    """Carga usuarios desde el archivo config.json"""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
            return data.get("users", [])
    return []

def save_user(username, password):
    """Guarda un nuevo usuario en el archivo config.json"""
    users = load_users()
    users.append({"username": username, "password": password})
    with open(CONFIG_PATH, "w") as f:
        json.dump({"users": users}, f, indent=4)

def authenticate(username, password):
    """Autentica si un usuario y contraseña son correctos"""
    users = load_users()
    for user in users:
        if user["username"] == username and user["password"] == password:
            return True
    return False

def access():
    load_css("styles.css")
    st.markdown("<h1 style='text-align: center;'>🌱 Predicción de exportación de fertilizantes</h1>", unsafe_allow_html=True)

    choice = st.selectbox('Login/Sigup', ['Login', 'Sign Up'])
    if choice == 'Login':
        username = st.text_input('Usuario')
        password = st.text_input('Contraseña', type="password")
        if st.button('Login'):
            if authenticate(username, password):
            #if username and password:
                st.success(f'Bienvenido de nuevo, {username}!')
                st.session_state.page ="app_control"
                st.rerun()
                
            else:
                st.error("⚠️ Por favor, completa todos los campos.")

    else: #Sign Up
        username = st.text_input('Nuevo Usuario')
        password = st.text_input('Nueva Contraseña', type="password")
        if st.button('Crear Cuenta'):
            if username and password:
                save_user(username, password)
                st.success("¡Cuenta creada exitosamente! Ahora puedes iniciar sesión.")
                st.session_state.page = "login"
                st.rerun()
            else:
                st.error("⚠️ Por favor completa todos los campos.")
'''
        email = st.text_input('Email')
        new_password = st.text_input('contraseña', type="password")
        username = st.text_input('Cree un usuario que no exista')

        if st.button('Cree una cuenta'):
            if email and new_password and username:
                st.success(f'Cuenta creada exitosamente para {username}!')
                st.session_state.page = "login"  # Volver al login
                st.rerun()
            else:
                st.error("⚠️ Por favor, completa todos los campos.")

'''
