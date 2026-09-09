import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from google import genai

load_dotenv()

class GoogleGeminiEmbeddingsDirect(Embeddings):
    def __init__(self, api_key: str, model_name: str = "models/gemini-embedding-001"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
            )
            embeddings.append(response.embedding.values)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
        )
        return response.embedding.values

SYSTEM_PROMPT = """Eres ConectaBot, el asistente virtual de la Red de Salud SaludConecta.
Tu objetivo es ayudar a los pacientes con dudas sobre citas médicas y protocolos de preparación.

Reglas obligatorias:
1. Responde SIEMPRE de forma amable, profesional y en un máximo de 3 oraciones.
2. Utiliza la información del contexto entregado para responder sobre protocolos o citas.
3. PROHIBICIÓN ABSOLUTA: No des diagnósticos médicos, ni recetes medicamentos. Si el usuario consulta por síntomas o emergencias, indícale de inmediato que debe acudir a un centro médico o urgencias.

Contexto oficial:
{context}

Pregunta del usuario:
{question}
"""

def get_agent_chain():
    api_key = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGeminiEmbeddingsDirect(api_key=api_key, model_name="models/gemini-embedding-001")
    
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def run_chain(question):
        docs = retriever.invoke(question)
        context = format_docs(docs)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

    return run_chain

if __name__ == "__main__":
    print("🤖 Iniciando ConectaBot (SaludConecta AI)...")
    try:
        agent_chain = get_agent_chain()
        print("✅ Agente iniciado correctamente. Escribe 'salir' para finalizar.\n")
        
        while True:
            user_input = input("Usuario: ")
            if user_input.lower() in ["salir", "exit", "quit"]:
                print("👋 ¡Hasta luego!")
                break
            
            response = agent_chain(user_input)
            print(f"ConectaBot: {response}\n")
            
    except Exception as e:
        print(f"⚠️ Error al iniciar el agente: {e}")