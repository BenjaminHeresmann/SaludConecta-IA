import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def build_vectorstore():
    print("🚀 Iniciando el proceso del Motor RAG...")
    
    docs_path = "data/docs_rag"
    print(f"📖 Cargando archivos PDF desde '{docs_path}'...")
    loader = PyPDFDirectoryLoader(docs_path)
    documents = loader.load()
    
    if not documents:
        print("⚠️ No se encontraron documentos PDF en data/docs_rag.")
        return None

    print(f"✅ Se cargaron {len(documents)} página(s) de documentos.")

    print("✂️ Dividiendo el texto en fragmentos (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Texto dividido en {len(chunks)} fragmento(s).")

    print("🧠 Generando embeddings con Google (gemini-embedding-001)...")

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise RuntimeError(
            "Falta GOOGLE_API_KEY. Configúrala en el archivo .env antes de ejecutar el motor RAG."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        google_api_key=google_api_key
    )

    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    except Exception as error:
        if "403" in str(error) and "project has been denied access" in str(error):
            raise RuntimeError(
                "Google denegó el acceso al proyecto asociado a GOOGLE_API_KEY. "
                "Crea o selecciona un proyecto habilitado, activa la Generative Language API, "
                "genera una clave nueva y actualiza GOOGLE_API_KEY en .env."
            ) from error
        raise
    
    output_folder = "faiss_index"
    vectorstore.save_local(output_folder)
    print(f"🎉 ¡Base de datos vectorial creada y guardada con éxito en '{output_folder}'!")
    
    return vectorstore

if __name__ == "__main__":
    build_vectorstore()