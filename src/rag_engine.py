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

    print("🧠 Generando embeddings con LangChain y Google...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    output_folder = "faiss_index"
    vectorstore.save_local(output_folder)
    print(f"🎉 ¡Base de datos vectorial creada y guardada con éxito en '{output_folder}'!")
    
    return vectorstore

if __name__ == "__main__":
    build_vectorstore()