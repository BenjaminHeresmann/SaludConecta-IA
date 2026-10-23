"""Configuración central: rutas del proyecto, modelo LLM (Groq) y embeddings (Gemini)."""
import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")

DATA = RAIZ / "data"
DOCS_RAG = DATA / "docs_rag"
RUNTIME = DATA / "runtime"          # estado generado al ejecutar (agenda, memoria, logs)
FAISS_DIR = RAIZ / "faiss_index"

MODELO_LLM = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
MODELO_RESPALDO = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.3-70b-versatile")
MODELO_EMBEDDINGS = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")


def crear_llm(modelo: str | None = None, temperatura: float = 0.1):
    """Crea el modelo de chat de Groq. En Qwen se oculta el razonamiento <think>."""
    from langchain_groq import ChatGroq

    modelo = modelo or MODELO_LLM
    extra = {"reasoning_format": "hidden"} if "qwen" in modelo else {}
    return ChatGroq(
        model=modelo,
        temperature=temperatura,
        max_tokens=1200,
        api_key=os.getenv("GROQ_API_KEY"),
        **extra,
    )


def crear_embeddings():
    """Embeddings de Gemini (768 dimensiones), los mismos con que se construyó el índice FAISS."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=MODELO_EMBEDDINGS,
        output_dimensionality=768,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
