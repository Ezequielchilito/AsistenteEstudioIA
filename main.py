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

from transformers import pipeline

# Inicializa el pipeline de resumen una sola vez para que no se recargue
# cada vez que Streamlit renderiza (usa st.cache_resource para esto)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn") # Un buen modelo para empezar
    # O "mrm8488/bert-spanish-cased-finetuned-summarization" para español si encuentras uno adecuado

summarizer = load_summarizer()

if st.button("Generar Resumen"):
    if uploaded_file is not None:
        # Usa el contenido del archivo
        text_to_summarize = text_content
    elif text_input:
        # Usa el texto pegado
        text_to_summarize = text_input
    else:
        st.warning("Por favor, carga un archivo o pega texto.")
        text_to_summarize = None

    if text_to_summarize:
        with st.spinner("Generando resumen..."):
            # Los modelos pueden tener límites de token, si el texto es muy largo
            # podrías necesitar dividirlo o usar un modelo más grande.
            summary_text = summarizer(text_to_summarize, max_length=150, min_length=50, do_sample=False)
            st.subheader("Resumen:")
            st.write(summary_text[0]['summary_text'])