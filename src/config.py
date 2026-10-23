"""Configuración central: rutas del proyecto, modelo LLM (Groq) y embeddings (Gemini)."""
import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")

DATA = RAIZ / "data"
DOCS_RAG = DATA / "docs_rag"
RUNTIME = DATA / "runtime"          # estado generado al ejecutar (agenda, memoria, logs)
# Plan gratuito de Groq: cada modelo tiene su propia cuota por minuto. qwen/qwen3.8-27b permite solo
# 1.000 tokens de salida por minuto, por eso el principal es gpt-oss-120b y los demás son respaldo.
MODELO_LLM = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MODELOS_RESPALDO = [m.strip() for m in os.getenv("GROQ_MODEL_FALLBACK", "openai/gpt-oss-20b,qwen/qwen3.8-27b").split(",")
                    if m.strip()]
MODELO_EMBEDDINGS = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")

# Embeddings: "local" (FastEmbed, sin API ni clave; por defecto) o "gemini" (API de Google, como en la EP1).
# Cada proveedor tiene su propio índice FAISS porque los vectores no son compatibles entre sí.
PROVEEDOR_EMBEDDINGS = os.getenv("EMBEDDINGS_PROVIDER", "local").lower()
MODELO_EMBEDDINGS_LOCAL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FAISS_DIR = RAIZ / ("faiss_index_local" if PROVEEDOR_EMBEDDINGS == "local" else "faiss_index")


def crear_llm(modelo: str | None = None, temperatura: float = 0.1, max_tokens: int = 700, reintentos: int = 0):
    """Crea un modelo de chat de Groq. Sin reintentos: si hay límite de uso (429) se pasa de inmediato
    al siguiente modelo de la cascada en vez de esperar. El razonamiento interno no se muestra."""
    from langchain_groq import ChatGroq

    modelo = modelo or MODELO_LLM
    if "qwen" in modelo:
        extra = {"reasoning_format": "hidden"}
    elif "gpt-oss" in modelo:
        extra = {"reasoning_effort": "low"}
    else:
        extra = {}
    return ChatGroq(model=modelo, temperature=temperatura, max_tokens=max_tokens, max_retries=reintentos,
                    api_key=os.getenv("GROQ_API_KEY"), **extra)


def crear_cascada(temperatura: float = 0.1, max_tokens: int = 700) -> list:
    """Modelo principal + respaldos. El último reintenta (espera la cuota) para no quedar sin respuesta."""
    nombres = [MODELO_LLM] + MODELOS_RESPALDO
    return [crear_llm(m, temperatura, max_tokens, reintentos=2 if i == len(nombres) - 1 else 0)
            for i, m in enumerate(nombres)]


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
