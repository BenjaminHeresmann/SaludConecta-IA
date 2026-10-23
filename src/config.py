"""Configuración central: rutas del proyecto, modelo LLM (Groq) y embeddings (Gemini)."""
import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")

DATA = RAIZ / "data"
DOCS_RAG = DATA / "docs_rag"
RUNTIME = DATA / "runtime"          # estado generado al ejecutar (agenda, memoria, logs)
MODELO_LLM = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
MODELO_RESPALDO = os.getenv("GROQ_MODEL_FALLBACK", "openai/gpt-oss-120b")
MODELO_EMBEDDINGS = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")

# Embeddings: "local" (FastEmbed, sin API ni clave; por defecto) o "gemini" (API de Google, como en la EP1).
# Cada proveedor tiene su propio índice FAISS porque los vectores no son compatibles entre sí.
PROVEEDOR_EMBEDDINGS = os.getenv("EMBEDDINGS_PROVIDER", "local").lower()
MODELO_EMBEDDINGS_LOCAL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FAISS_DIR = RAIZ / ("faiss_index_local" if PROVEEDOR_EMBEDDINGS == "local" else "faiss_index")


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
    """Gemini (768 dimensiones, índice de la EP1) o modelo local multilingüe (384 dimensiones)."""
    if PROVEEDOR_EMBEDDINGS == "local":
        from langchain_community.embeddings import FastEmbedEmbeddings

        # Caché en una ruta corta (raíz de la unidad): Windows falla con rutas de más de 260 caracteres.
        cache = os.getenv("FASTEMBED_CACHE", str(Path(RAIZ.anchor) / "modelos-ia"))
        return FastEmbedEmbeddings(model_name=MODELO_EMBEDDINGS_LOCAL, cache_dir=cache)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=MODELO_EMBEDDINGS,
        output_dimensionality=768,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
