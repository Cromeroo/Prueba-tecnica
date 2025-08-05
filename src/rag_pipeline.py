import os
import shutil
import streamlit as st
from typing import TypedDict
import time
import logging
import vertexai

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\danil\OneDrive\Escritorio\RAG\vertex2-stately-moon.json"

logger = logging.getLogger("LLMLogger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("llm_latency_logs.txt", mode="a", encoding="utf-8")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAI

docs_path = "data/documents/"
vector_db_path = "data/vector_store/"
os.makedirs(docs_path, exist_ok=True)
os.makedirs(vector_db_path, exist_ok=True)

class RAGState(TypedDict):
    query: str
    docs: list
    response: str

def load_and_process_documents():
    texts = []
    for file in os.listdir(docs_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(docs_path, file)
            print(f"📄 Procesando archivo: {file}")
            try:
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                if documents and any(doc.page_content.strip() for doc in documents):
                    print(f"✅ Texto extraído de {file}.")
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                    text_chunks = text_splitter.split_documents(documents)
                    for i, chunk in enumerate(text_chunks[:3]):
                        print(f"📌 Fragmento {i+1}: {chunk.page_content[:200]}...")
                    texts.extend(text_chunks)
                else:
                    print(f"⚠️ No se extrajo texto de {file}.")
            except Exception as e:
                print(f"❌ Error procesando {file}: {e}")
    if not texts:
        print("⚠️ No se extrajo texto de ningún documento.")
    return texts

def create_vector_store(texts):
    try:
        vertexai.init(
            project="stately-moon-451804-a9", 
            location="us-central1"
        )
        
        embeddings = VertexAIEmbeddings(model_name="textembedding-gecko")
        if os.path.exists(vector_db_path):
            print("🔄 Reiniciando la base de datos vectorial...")
            vector_db = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
            vector_db.delete_collection()
            del vector_db
            shutil.rmtree(vector_db_path, ignore_errors=True)
            print("✅ Base vectorial eliminada.")
        vector_db = Chroma.from_documents(texts, embeddings, persist_directory=vector_db_path)
        print("✅ Base vectorial creada correctamente.")
        return vector_db
    except Exception as e:
        print(f"❌ Error detallado en create_vector_store(): {e}")
        import traceback
        traceback.print_exc()
        return None

def build_rag_chain(vector_db):
    if vector_db is None:
        print(" Error: La base de datos vectorial es None. No se puede construir la cadena RAG.")
        
        def error_pipeline(input_state: dict) -> dict:
            return {
                "query": input_state.get("query", ""),
                "docs": [],
                "response": " Error: Base de datos vectorial no disponible. Por favor, sube documentos primero."
            }
        
        return error_pipeline, "Error: Base de datos vectorial no disponible"
    
    vertexai.init(
        project="stately-moon-451804-a9",
        location="us-central1"
    )
    
    llm = VertexAI(model_name="gemini-2.5-flash") 
    def retrieve(state: dict) -> dict:
        query = state["query"]
        try:
            docs = vector_db.similarity_search(query)
            print("\n🔎 Documentos recuperados:")
            for doc in docs:
                print(f"- {doc.page_content[:300]}...")
            state["docs"] = docs
            return state
        except Exception as e:
            print(f"❌ Error en retrieve: {e}")
            state["docs"] = []
            return state

    def generate(state: dict) -> dict:
        query = state["query"]
        docs = state.get("docs", [])
        if not docs:
            print("⚠️ No se recuperaron documentos.")
            state["response"] = "⚠️ No se encontraron documentos relevantes para responder."
            return state

        print("\n🔎 Documentos recuperados (generate):")
        formatted_docs = "\n\n".join(doc.page_content for doc in docs)
        formatted_prompt = f"""
        Human:
        Usa los siguientes documentos para responder la pregunta.

        **Documentos:**
        {formatted_docs}

        **Pregunta:** {query}
        """
        print(f"\n📨 Prompt enviado al modelo:\n{formatted_prompt}\n")
        try:
            start_time = time.perf_counter()
            raw_response = llm.invoke(formatted_prompt)
            end_time = time.perf_counter()
            latency = end_time - start_time
            logger.info(f"Tiempo de latencia del LLM: {latency:.2f} segundos")
            for handler in logger.handlers:
                handler.flush()
            print(f"💬 Respuesta obtenida: {raw_response}")

            if isinstance(raw_response, str):
                state["response"] = raw_response
            elif hasattr(raw_response, "content"):
                state["response"] = raw_response.content
            else:
                state["response"] = "⚠️ No se generó una respuesta válida."
            return state
        except Exception as e:
            end_time = time.perf_counter()
            latency = end_time - start_time
            logger.error(f"Error al invocar VertexAI después de {latency:.2f} segundos: {e}")
            for handler in logger.handlers:
                handler.flush()
            state["response"] = "⚠️ Error al generar respuesta con el modelo."
            return state

    def pipeline(input_state: dict) -> dict:
        after_retrieve = retrieve(input_state)
        final_state = generate(after_retrieve)
        return final_state

    return pipeline, None
