import streamlit as st

st.title("Asistente de Estudio Inteligente")

uploaded_file = st.file_uploader("Sube tu archivo de estudio (PDF, TXT)", type=["pdf", "txt"])
if uploaded_file is not None:
    # Aquí lees el contenido del archivo
    st.write("Archivo cargado con éxito!")
    # Si es TXT:
    text_content = uploaded_file.read().decode("utf-8")
    st.text_area("Contenido del archivo:", text_content, height=300)
    # Si es PDF, necesitarás una librería como PyPDF2 o pypdf para extraer el texto.
    # (pip install pypdf)

else:
    text_input = st.text_area("O pega tu texto de estudio aquí:", height=300)

# Continúa aquí con la lógica de resumen y preguntas