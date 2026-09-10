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
    loader = PyPDFDirectoryLoader(docs_path)
    documents = loader.load()
    
    if not documents:
        print("⚠️ No se encontraron documentos PDF en data/docs_rag.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    print("🧠 Generando embeddings con Google Gemini...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        output_dimensionality=768
    )
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("🎉 ¡Base de datos vectorial creada y guardada con éxito!")

if __name__ == "__main__":
    build_vectorstore()