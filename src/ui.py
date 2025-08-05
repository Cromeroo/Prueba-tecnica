import os
import streamlit as st
import shutil
from rag_pipeline import load_and_process_documents, create_vector_store, build_rag_chain

# Personaliza el estilo con CSS (opcional)
st.markdown(
    """
    <style>
    body {
        background-color: #F2F2F2;
    }
    .header {
        text-align: center;
        color: #4B0082;
    }
    .instructions {
        background-color: #4B0082;  /* Fondo oscuro */
        color: white;             /* Texto blanco */
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
#No saleee!!!!
# Título principal con emoji
st.markdown("<h1 class='header'>🤖 Chatbot RAG con LangGraph</h1>", unsafe_allow_html=True)

# Instrucciones al usuario
st.markdown(
    """
    <div class="instructions">
    <h3>📚 Instrucciones</h3>
    <ul>
      <li>Sube uno o varios archivos PDF. ¡Sí, funciona con múltiples PDFs para un análisis conjunto!</li>
      <li>Presiona el botón <b>"Entrenar modelo"</b> para procesar y almacenar los documentos.</li>
      <li>Una vez entrenado, ingresa tu pregunta en el campo correspondiente y presiona <b>"Consultar"</b> para obtener la respuesta.</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# Definir la ruta donde se guardarán los documentos
docs_path = "data/documents/"

# Esto debe elliminar la carpeta y su contenido si existe 
if os.path.exists(docs_path):
    shutil.rmtree(docs_path)
os.makedirs(docs_path, exist_ok=True)

# Subida de archivos PDF con mensaje
uploaded_files = st.file_uploader(
    "📄 Sube tus archivos PDF",
    accept_multiple_files=True,
    type=["pdf"],
    key="unique_pdf_uploader"
)
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(docs_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
    st.success("✅ Archivos subidos correctamente.")

# Botón para entrenar el modelo
if st.button("Entrenar modelo", key="train_button"):
    st.info("🔄 Iniciando entrenamiento...")
    try:
        with st.spinner("🚀 Procesando documentos y entrenando el modelo..."):
            texts = load_and_process_documents()
            st.write("DEBUG: Número de fragmentos extraídos:", len(texts))
            
            if len(texts) == 0:
                st.error("❌ No se extrajeron fragmentos de texto. Verifica los PDFs.")
                st.stop()
            
            vector_db = create_vector_store(texts)
            st.write("DEBUG: Base vectorial:", vector_db)
            
            if vector_db is None:
                st.error("❌ Falló la creación de la base vectorial.")
                st.stop()
            
            pipeline, _ = build_rag_chain(vector_db)
            st.session_state["rag_chain"] = pipeline
            st.write("DEBUG: Pipeline asignado:", pipeline)
        st.success("🎉 Modelo entrenado con éxito. Ahora puedes hacer preguntas.")
    except Exception as e:
        st.error(f"❌ Error durante el entrenamiento: {e}")

# Campo de entrada para la consulta
query = st.text_input("❓ Haz una pregunta sobre los documentos:", key="query_input")

# Botón para consultar
if st.button("Consultar", key="consult_button"):
    if "rag_chain" in st.session_state and query.strip():
        # AGREGAR DEBUG AQUÍ
        if st.session_state["rag_chain"] is None:
            st.error("❌ RAG Chain es None. El entrenamiento falló.")
        else:
            st.info(f"✅ RAG Chain disponible: {type(st.session_state['rag_chain'])}")
            with st.spinner("🔍 Consultando..."):
                result_state = st.session_state["rag_chain"]({"query": query})
                if result_state and isinstance(result_state, dict) and "response" in result_state:
                    st.markdown("**Respuesta:**")
                    st.write(result_state["response"])
                else:
                    st.warning(f"⚠️ El objeto `result_state` no es válido: {result_state}")
    else:
        st.warning("⚠️ No hay cadena RAG entrenada o la pregunta está vacía.")
        # AGREGAR MÁS DEBUG mas masssssssssssssss no sé que hacer acá quedo par amañana
        if "rag_chain" not in st.session_state:
            st.error("❌ 'rag_chain' no existe en session_state")
        elif st.session_state["rag_chain"] is None:
            st.error("❌ 'rag_chain' es None en session_state")
        if not query.strip():
            st.error("❌ La pregunta está vacía")
