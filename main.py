import streamlit as st
from transformers import pipeline
import pypdf # Importamos la librería para PDF
import io    # Para trabajar con los bytes del archivo cargado

# --- Configuración de la página (opcional, pero mejora la apariencia) ---
st.set_page_config(layout="centered", page_title="Asistente de Estudio IA")

# --- Título y descripción de la aplicación ---
st.title("📚 Asistente de Estudio Inteligente")
st.markdown("Carga tus documentos de estudio (TXT o PDF) para obtener resúmenes y preguntas de autoevaluación.")

# --- Sección para la carga de archivos o entrada de texto ---
uploaded_file = st.file_uploader(
    "1. Sube tu archivo de estudio (.txt o .pdf)",
    type=["txt", "pdf"],
    help="Soporta archivos de texto plano (.txt) y documentos PDF (.pdf)."
)

text_content = "" # Variable para almacenar el contenido de texto extraído

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension == "txt":
        st.info("Archivo TXT detectado. Intentando decodificar...")
        try:
            # Vuelve al inicio del archivo cargado para asegurar una lectura completa
            uploaded_file.seek(0)
            text_content = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            st.warning("El archivo TXT no está en UTF-8. Intentando con Latin-1...")
            uploaded_file.seek(0) # Vuelve a posicionar el puntero para reintentar
            try:
                text_content = uploaded_file.read().decode("latin-1")
            except UnicodeDecodeError:
                st.error("❌ No se pudo decodificar el archivo TXT ni con UTF-8 ni con Latin-1. Por favor, verifica la codificación del archivo.")
                text_content = "" # Asegura que la variable esté vacía si falla

    elif file_extension == "pdf":
        st.info("Archivo PDF detectado. Extrayendo texto...")
        try:
            # PyPDF2 requiere un objeto de archivo tipo bytes, io.BytesIO lo provee
            reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
            pdf_text_pages = []
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                extracted_page_text = page.extract_text()
                if extracted_page_text: # Solo añade si hay texto
                    pdf_text_pages.append(extracted_page_text)
            text_content = "\n".join(pdf_text_pages)
            if not text_content.strip(): # Si el texto extraído está vacío después de limpiar espacios
                st.warning("⚠️ No se pudo extraer texto del PDF. Podría ser un PDF escaneado (imagen) o protegido.")
        except Exception as e:
            st.error(f"❌ Error al leer el archivo PDF: {e}. Asegúrate de que sea un PDF válido y no protegido.")
            text_content = "" # Asegura que la variable esté vacía si falla

    else: # Esto debería ser manejado por 'type=["txt", "pdf"]' en file_uploader, pero es buena práctica
        st.warning("Tipo de archivo no soportado. Por favor, sube un archivo .txt o .pdf")
        text_content = ""

# Área de texto para pegar contenido si no se sube un archivo
# Se mostrará si no hay un archivo cargado o si el cargador de archivos es None (al inicio)
if not text_content: # Si el texto no viene de un archivo, permite pegarlo
    text_input_area = st.text_area(
        "2. O pega tu texto de estudio aquí:",
        height=300,
        help="El texto pegado será utilizado para el resumen si no se sube un archivo."
    )
    if text_input_area: # Si el usuario pega texto, actualiza text_content
        text_content = text_input_area


# --- Carga del modelo de resumen (optimizado con st.cache_resource) ---
@st.cache_resource
def load_summarizer_model():
    st.spinner("Cargando modelo de resumen (esto puede tardar la primera vez)...")
    try:
        # Puedes cambiar este modelo por uno en español si encuentras uno mejor:
        # Por ejemplo: "mrm8488/bert-spanish-cased-finetuned-summarization" (verifica su compatibilidad y rendimiento)
        # O "facebook/bart-large-cnn" sigue siendo bueno para inglés general y muchos idiomas si el texto es limpio.
        return pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo de resumen: {e}. Asegúrate de tener PyTorch o TensorFlow instalado correctamente.")
        return None

summarizer = load_summarizer_model()


# --- Sección para el botón de resumen y mostrar resultados ---
st.markdown("---") # Separador visual

if summarizer is not None: # Solo muestra el botón si el modelo se cargó correctamente
    if st.button("✨ Generar Resumen"):
        if not text_content.strip(): # .strip() quita espacios en blanco para ver si hay texto real
            st.warning("Por favor, sube un archivo o pega texto para generar el resumen.")
        else:
            with st.spinner("Generando resumen... Por favor, espera."):
                try:
                    # Los modelos tienen límites de token. Si el texto es muy largo, podría fallar o resumir solo el inicio.
                    # Para textos muy largos, necesitarías dividir el texto en chunks y resumir cada chunk.
                    # max_length y min_length controlan la longitud del resumen.
                    summary_output = summarizer(text_content, max_length=200, min_length=50, do_sample=False)
                    st.subheader("📝 Resumen Generado:")
                    st.write(summary_output[0]['summary_text'])
                except Exception as e:
                    st.error(f"❌ Error al generar el resumen: {e}. El texto puede ser demasiado largo para el modelo o hubo un problema interno.")
                    st.info("Intenta con un texto más corto si el problema persiste.")
else:
    st.warning("El modelo de resumen no está disponible. No se puede generar resúmenes.")

# --- Sección para la generación de preguntas (futuro) ---
st.markdown("---")
st.subheader("❓ Generación de Preguntas (Próximamente)")
st.info("Esta funcionalidad se implementará en futuras versiones para ayudarte con la autoevaluación.")

st.markdown("---")
st.caption("Asistente de Estudio Inteligente v0.1")