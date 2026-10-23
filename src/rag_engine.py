"""Motor RAG: ingesta de protocolos (PDF y Markdown) y búsqueda semántica en FAISS.

Construir el índice:  python -m src.rag_engine
"""
from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DOCS_RAG, FAISS_DIR, PROVEEDOR_EMBEDDINGS, crear_embeddings

INDICE = FAISS_DIR / "indice_ep2.bin"


def build_vectorstore():
    print("Iniciando el proceso del Motor RAG...")
    documents = PyPDFDirectoryLoader(str(DOCS_RAG)).load()
    documents += DirectoryLoader(str(DOCS_RAG), glob="*.md", loader_cls=TextLoader,
                                 loader_kwargs={"encoding": "utf-8"}).load()
    if not documents:
        print("No se encontraron documentos en data/docs_rag.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    print(f"Generando embeddings ({PROVEEDOR_EMBEDDINGS}) para {len(chunks)} fragmentos...")
    vectorstore = FAISS.from_documents(chunks, crear_embeddings())
    # Se guarda como bytes desde Python: el save_local de FAISS (C++) falla en Windows
    # cuando la ruta tiene tildes (ej. "Evaluación").
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    INDICE.write_bytes(vectorstore.serialize_to_bytes())
    print(f"Base de datos vectorial creada y guardada en {INDICE.relative_to(FAISS_DIR.parent)}.")
    return vectorstore


_vectorstore = None


def cargar_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = FAISS.deserialize_from_bytes(INDICE.read_bytes(), crear_embeddings(),
                                                    allow_dangerous_deserialization=True)
    return _vectorstore


def buscar(consulta: str, k: int = 3) -> list[dict]:
    """Devuelve los k fragmentos más similares con su distancia L2 (menor = más parecido) y su fuente."""
    resultados = cargar_vectorstore().similarity_search_with_score(consulta, k=k)
    return [{"texto": doc.page_content, "fuente": doc.metadata.get("source", "").split("\\")[-1].split("/")[-1],
             "distancia": round(float(score), 4)} for doc, score in resultados]


if __name__ == "__main__":
    build_vectorstore()
