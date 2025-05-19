import streamlit as st
from transformers import pipeline
import pypdf
import io
import nltk # Importamos NLTK para la división de texto
from nltk.tokenize import sent_tokenize # Para dividir en oraciones

# Descarga los datos de tokenización de NLTK (solo la primera vez)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- Configuración de la página ---
st.set_page_config(layout="centered", page_title="Asistente de Estudio IA")

# --- Título y descripción ---
st.title("📚 Asistente de Estudio Inteligente")
st.markdown("Carga tus documentos de estudio (TXT o PDF) para obtener resúmenes y preguntas de autoevaluación.")

# --- Sección para la carga de archivos o entrada de texto ---
uploaded_file = st.file_uploader(
    "1. Sube tu archivo de estudio (.txt o .pdf)",
    type=["txt", "pdf"],
    help="Soporta archivos de texto plano (.txt) y documentos PDF (.pdf)."
)

text_content = ""

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension == "txt":
        st.info("Archivo TXT detectado. Intentando decodificar...")
        try:
            uploaded_file.seek(0)
            text_content = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            st.warning("El archivo TXT no está en UTF-8. Intentando con Latin-1...")
            uploaded_file.seek(0)
            try:
                text_content = uploaded_file.read().decode("latin-1")
            except UnicodeDecodeError:
                st.error("❌ No se pudo decodificar el archivo TXT.")
                text_content = ""

    elif file_extension == "pdf":
        st.info("Archivo PDF detectado. Extrayendo texto...")
        try:
            reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
            pdf_text_pages = []
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                extracted_page_text = page.extract_text()
                if extracted_page_text:
                    pdf_text_pages.append(extracted_page_text)
            text_content = "\n".join(pdf_text_pages)
            if not text_content.strip():
                st.warning("⚠️ No se pudo extraer texto del PDF. Podría ser un PDF escaneado (imagen) o protegido.")
        except Exception as e:
            st.error(f"❌ Error al leer el archivo PDF: {e}. Asegúrate de que sea un PDF válido y no protegido.")
            text_content = ""

    else:
        st.warning("Tipo de archivo no soportado. Por favor, sube un archivo .txt o .pdf")
        text_content = ""

if not text_content: # Si el texto no viene de un archivo o falló la lectura, permite pegarlo
    text_input_area = st.text_area(
        "2. O pega tu texto de estudio aquí:",
        height=300,
        help="El texto pegado será utilizado para el resumen si no se sube un archivo."
    )
    if text_input_area:
        text_content = text_input_area

# --- Función para dividir texto en chunks de forma inteligente ---
def split_text_into_chunks(text, max_chunk_length=1000): # max_chunk_length en caracteres
    # Usamos sent_tokenize para dividir en oraciones, manteniendo la coherencia
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Añade un espacio para separar oraciones si no es la primera
        if current_chunk and not current_chunk.endswith((' ', '\n')):
            temp_chunk = current_chunk + " " + sentence
        else:
            temp_chunk = current_chunk + sentence

        # Si el chunk temporal excede el límite, guarda el chunk actual y empieza uno nuevo
        if len(temp_chunk) > max_chunk_length and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence # La oración actual inicia el nuevo chunk
        else:
            current_chunk = temp_chunk # Continúa construyendo el chunk

    # Añade el último chunk si no está vacío
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# --- Carga del modelo de resumen ---
@st.cache_resource
def load_summarizer_model():
    with st.spinner("Cargando modelo de resumen (esto puede tardar la primera vez)..."):
        try:
            # Modelo BART es bueno para inglés. Puedes cambiarlo si encuentras uno mejor para español.
            return pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e:
            st.error(f"❌ Error al cargar el modelo de resumen: {e}. Asegúrate de tener PyTorch o TensorFlow instalado correctamente.")
            return None

summarizer = load_summarizer_model()

# --- Sección para el botón de resumen y mostrar resultados ---
st.markdown("---")

if summarizer is not None:
    if st.button("✨ Generar Resumen"):
        if not text_content.strip():
            st.warning("Por favor, sube un archivo o pega texto para generar el resumen.")
        else:
            with st.spinner("Generando resumen... Por favor, espera."):
                final_summary_parts = []
                try:
                    # Umbral de longitud del texto para decidir si hacer chunking (aproximado)
                    # 2000 caracteres es solo un punto de partida, ajústalo según pruebas
                    if len(text_content) > 2000:
                        st.info("Texto largo detectado. Procesando en secciones para un resumen completo...")
                        chunks = split_text_into_chunks(text_content, max_chunk_length=1000) # Ajusta este valor
                        
                        # Resumir cada chunk
                        for i, chunk in enumerate(chunks):
                            if chunk.strip(): # Asegúrate de que el chunk no esté vacío
                                st.text(f"Procesando sección {i+1}/{len(chunks)}...")
                                # Ajusta max_length y min_length para los resúmenes de los chunks
                                # No los hagas demasiado largos, ya que podrías resumir los resúmenes.
                                chunk_summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
                                final_summary_parts.append(chunk_summary[0]['summary_text'])
                        
                        # Opcional: Resumir los resúmenes de los chunks para una mayor cohesión
                        if final_summary_parts:
                            combined_summaries = " ".join(final_summary_parts)
                            st.info("Combinando y refinando resúmenes de secciones...")
                            # Resumir el texto combinado de los mini-resúmenes
                            # Aquí el max_length puede ser mayor para el resumen final
                            final_summary_output = summarizer(combined_summaries, max_length=300, min_length=100, do_sample=False)
                            final_summary = final_summary_output[0]['summary_text']
                        else:
                            final_summary = "No se pudo generar un resumen completo."

                    else: # Para textos cortos o medianos
                        st.info("Texto de longitud estándar. Generando resumen directo...")
                        # Ajusta max_length y min_length aquí para el resumen directo
                        summary_output = summarizer(text_content, max_length=300, min_length=100, do_sample=False)
                        final_summary = summary_output[0]['summary_text']
                    
                    st.subheader("📝 Resumen Generado:")
                    st.write(final_summary)

                except Exception as e:
                    st.error(f"❌ Error al generar el resumen: {e}. El texto puede ser muy complejo o hubo un problema con el modelo.")
                    st.info("Intenta con un texto más corto o diferente si el problema persiste.")
else:
    st.warning("El modelo de resumen no está disponible. No se puede generar resúmenes.")

# --- Sección para la generación de preguntas (futuro) ---
st.markdown("---")
st.subheader("❓ Generación de Preguntas (Próximamente)")
st.info("Esta funcionalidad se implementará en futuras versiones para ayudarte con la autoevaluación.")

st.markdown("---")
st.caption("Asistente de Estudio Inteligente v0.1")