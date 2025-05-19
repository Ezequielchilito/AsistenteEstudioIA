import streamlit as st
from transformers import pipeline
import pypdf
import io
import nltk
from nltk.tokenize import sent_tokenize

# --- Descarga los datos de tokenización de NLTK (solo se ejecuta una vez si no están presentes) ---
# Esto es crucial para que sent_tokenize funcione.
# NLTK detecta si 'punkt' ya está descargado y lo ignora si es así.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError: # LookupError es la excepción correcta para 'find' cuando no encuentra el recurso
    nltk.download('punkt', quiet=True) # 'quiet=True' para no imprimir tantos mensajes en la terminal

# --- Configuración de la página ---
st.set_page_config(layout="centered", page_title="Asistente de Estudio IA")

# --- Título y descripción de la aplicación ---
st.title("📚 Asistente de Estudio Inteligente")
st.markdown("Carga tus documentos de estudio (TXT o PDF) o pega texto para obtener resúmenes automáticos.")

# --- Sección para la carga de archivos o entrada de texto ---
uploaded_file = st.file_uploader(
    "1. Sube tu archivo de estudio (.txt o .pdf)",
    type=["txt", "pdf"],
    help="Soporta archivos de texto plano (.txt) y documentos PDF (.pdf)."
)

text_content = "" # Variable para almacenar el contenido de texto extraído del archivo

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension == "txt":
        st.info("Archivo TXT detectado. Intentando decodificar...")
        try:
            uploaded_file.seek(0) # Vuelve al inicio del archivo cargado para asegurar una lectura completa
            text_content = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            st.warning("El archivo TXT no está en UTF-8. Intentando con Latin-1...")
            uploaded_file.seek(0) # Vuelve a posicionar el puntero para reintentar
            try:
                text_content = uploaded_file.read().decode("latin-1")
            except UnicodeDecodeError:
                st.error("❌ Error: No se pudo decodificar el archivo TXT ni con UTF-8 ni con Latin-1. Por favor, verifica la codificación del archivo.")
                text_content = "" # Asegura que la variable esté vacía si falla la decodificación

    elif file_extension == "pdf":
        st.info("Archivo PDF detectado. Extrayendo texto...")
        try:
            # PyPDF requiere un objeto de archivo tipo bytes, io.BytesIO lo provee
            reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
            pdf_text_pages = []
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                extracted_page_text = page.extract_text()
                if extracted_page_text: # Solo añade texto si la página no está vacía
                    pdf_text_pages.append(extracted_page_text)
            text_content = "\n".join(pdf_text_pages)
            if not text_content.strip(): # Si el texto extraído está vacío después de limpiar espacios
                st.warning("⚠️ Advertencia: No se pudo extraer texto del PDF. Podría ser un PDF escaneado (imagen) o protegido con contraseña.")
        except Exception as e:
            st.error(f"❌ Error al leer el archivo PDF: {e}. Asegúrate de que sea un PDF válido y no protegido.")
            text_content = "" # Asegura que la variable esté vacía si falla la extracción

    else: # Esto es un fallback, file_uploader con 'type' ya debería filtrar
        st.warning("Tipo de archivo no soportado. Por favor, sube un archivo .txt o .pdf")
        text_content = ""

# Área de texto para pegar contenido si no se sube un archivo o si la carga falló
# Solo muestra esta área si 'text_content' está vacío (no hay texto de un archivo)
if not text_content.strip(): # .strip() para considerar cadenas solo con espacios como vacías
    text_input_area = st.text_area(
        "2. O pega tu texto de estudio aquí:",
        height=300,
        help="El texto pegado será utilizado para el resumen si no se sube un archivo o el archivo no pudo ser procesado."
    )
    if text_input_area: # Si el usuario pega texto, actualiza text_content
        text_content = text_input_area

# --- Función para dividir texto en chunks de forma inteligente (para modelos con límite de tokens) ---
def split_text_into_chunks(text, max_chunk_length=800): # max_chunk_length en caracteres
    # Usamos sent_tokenize para dividir en oraciones, manteniendo la coherencia
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Añade un espacio para separar oraciones si no es la primera del chunk
        if current_chunk and not current_chunk.endswith((' ', '\n')):
            temp_chunk = current_chunk + " " + sentence
        else:
            temp_chunk = current_chunk + sentence

        # Si el chunk temporal excede el límite (o está a punto de), guarda el chunk actual
        # y la oración actual comienza un nuevo chunk.
        # Ajustar 0.9 para dejar un margen para la tokenización del modelo
        if len(temp_chunk) > max_chunk_length * 0.9 and current_chunk: # Considera un margen
            chunks.append(current_chunk.strip())
            current_chunk = sentence # La oración actual inicia el nuevo chunk
        else:
            current_chunk = temp_chunk # Continúa construyendo el chunk

    # Añade el último chunk si no está vacío
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# --- Carga del modelo de resumen (optimizado con st.cache_resource) ---
@st.cache_resource
def load_summarizer_model():
    with st.spinner("Cargando modelo de resumen (esto puede tardar la primera vez)..."):
        try:
            # Puedes cambiar este modelo por uno en español si encuentras uno mejor:
            # Por ejemplo: "mrm8488/bert-spanish-cased-finetuned-summarization" (verifica su compatibilidad y rendimiento)
            # O "facebook/bart-large-cnn" es generalista y funciona aceptablemente con español si el texto es limpio.
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
            st.write("DEBUG: Contenido del texto antes de resumir (primeros 500 caracteres):")
            st.code(text_content[:500]) # Muestra el inicio del texto
            st.write(f"DEBUG: Longitud total del texto: {len(text_content)} caracteres")
            with st.spinner("Generando resumen... Por favor, espera."):
                final_summary = ""
                try:
                    # Umbral de longitud del texto para decidir si hacer chunking (en caracteres)
                    # El límite de tokens de BART-large-CNN es 1024. 2000-3000 caracteres es un buen umbral inicial.
                    if len(text_content) > 3000: # Ajusta este umbral según tus pruebas
                        st.info("Texto largo detectado. Procesando en secciones para un resumen completo...")
                        # max_chunk_length ajustado para tener margen para el modelo (aprox. 1024 tokens)
                        chunks = split_text_into_chunks(text_content, max_chunk_length=2000) 
                        
                        all_summaries_from_chunks = []
                        for i, chunk in enumerate(chunks):
                            if chunk.strip():
                                st.text(f"Resumiendo sección {i+1}/{len(chunks)}...")
                                # Los resúmenes de los chunks pueden ser más cortos, luego se resumirán ellos mismos
                                chunk_summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
                                all_summaries_from_chunks.append(chunk_summary[0]['summary_text'])
                        
                        # Fase de re-resumen: combinar los resúmenes parciales y resumirlos de nuevo
                        if all_summaries_from_chunks:
                            combined_summaries_text = " ".join(all_summaries_from_chunks)
                            st.info("Combinando y refinando resúmenes de secciones...")
                            # Ahora resumimos el conjunto de los resúmenes anteriores. Aquí el max_length puede ser más alto.
                            final_summary_output = summarizer(combined_summaries_text, max_length=500, min_length=150, do_sample=False)
                            final_summary = final_summary_output[0]['summary_text']
                        else:
                            final_summary = "No se pudo generar un resumen completo de las secciones."

                    else: # Para textos cortos o medianos (directo al modelo)
                        st.info("Texto de longitud estándar. Generando resumen directo...")
                        # Ajusta max_length y min_length aquí para el resumen directo
                        # Estos valores controlan la longitud del resumen final.
                        summary_output = summarizer(text_content, max_length=500, min_length=100, do_sample=False)
                        final_summary = summary_output[0]['summary_text']
                    
                    st.subheader("📝 Resumen Generado:")
                    st.write(final_summary)

                except Exception as e:
                    st.error(f"❌ Error al generar el resumen: {e}. El texto puede ser muy largo, el modelo tuvo un problema o el texto es demasiado complejo.")
                    st.info("Intenta con un texto más corto o diferente si el problema persiste. Verifica la terminal para más detalles.")
else:
    st.warning("El modelo de resumen no está disponible. No se puede generar resúmenes.")

# --- Sección para la generación de preguntas (futuro) ---
st.markdown("---")
st.subheader("❓ Generación de Preguntas (Próximamente)")
st.info("Esta funcionalidad se implementará en futuras versiones para ayudarte con la autoevaluación.")

st.markdown("---")
st.caption("Asistente de Estudio Inteligente v0.1")