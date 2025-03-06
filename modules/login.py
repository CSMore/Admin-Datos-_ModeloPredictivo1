import streamlit as st

# --- Cargar estilos desde el archivo CSS ---
def load_css(file_name):
    try:
        with open(file_name, "r") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("⚠️ El archivo CSS no se encontró.")

def access():
    load_css("styles.css")
    st.markdown("<h1 style='text-align: center;'>🌱 Predicción de exportación de fertilizantes</h1>", unsafe_allow_html=True)

    choice = st.selectbox('Login/Sigup', ['Login', 'Sign Up'])
    if choice == 'Login':
        username = st.text_input('Usuario')
        password = st.text_input('Contraseña', type="password")
        if st.button('Login'):
            if username and password:
                st.success(f'Bienvenido de nuevo, {username}!')
                st.session_state.page ="app_control"
                st.rerun()
                
            else:
                st.error("⚠️ Por favor, completa todos los campos.")

    else: #Sign Up
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


