import os
import re
import traceback
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Prompt ajustado para forzar salida directa sin etiquetas de razonamiento
SYSTEM_PROMPT = """Eres ConectaBot, el asistente virtual oficial de la Red de Salud SaludConecta.
Tu objetivo es responder las consultas de los pacientes utilizando exclusivamente el contexto proporcionado.

REGLAS OBLIGATORIAS:
1. Responde SIEMPRE en español de forma amable, clara y profesional.
2. Utiliza la información del contexto adjunto para responder sobre cancelaciones, emergencias, preparaciones o citas.
3. PROHIBICIÓN DE RAZONAMIENTO: No incluyas cadenas de pensamiento, reflexiones ni etiquetas <think>. Escribe DIRECTAMENTE la respuesta final para el usuario.
4. Si el contexto no contiene la respuesta, indica amablemente que no dispones de esa información en el sistema.
5. PROHIBICIÓN ABSOLUTA: No des diagnósticos médicos ni recetes medicamentos. Ante síntomas graves o emergencias, indica acudir de inmediato a urgencias.
"""

def get_agent_chain():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        output_dimensionality=768,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
    )

    def run_chain(question):
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        prompt_completo = f"{SYSTEM_PROMPT}\n\nContexto extraído de los documentos:\n{context}"

        response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[
                {"role": "system", "content": prompt_completo},
                {"role": "user", "content": question}
            ],
            temperature=0.1,
            max_tokens=450  # Límite ampliado para permitir la respuesta tras el contexto
        )
        
        raw_content = response.choices[0].message.content or ""
        
        # Limpieza de cualquier etiqueta <think> que el modelo genere
        if "</think>" in raw_content:
            clean_content = raw_content.split("</think>")[-1].strip()
        else:
            clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL)
            clean_content = re.sub(r'<think>.*', '', clean_content, flags=re.DOTALL).strip()
            
        if not clean_content:
            # Si el modelo consumió los tokens antes de cerrar el pensamiento, mostramos la salida cruda filtrada
            clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
            if not clean_content:
                clean_content = "Hola. No dispongo de información específica sobre esos requisitos en el sistema. Te sugiero contactar directamente a la recepción de SaludConecta para más detalles."

        return clean_content

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
            print(f"\nConectaBot: {response}\n")
            
    except Exception as e:
        print(f"⚠️ Error en el agente: {e}")
        traceback.print_exc()