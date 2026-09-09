#  SaludConecta - Sistema RAG & Agente de Gestión de Citas con IA

Sistema de Inteligencia Artificial para la gestión autónoma de citas médicas, prevención de ausentismo (*no-show*) y atención a pacientes de la **Red de Salud SaludConecta**. Integra un pipeline de Generación Aumentada por Recuperación (RAG) para consulta de protocolos clínicos y un Agente Conversacional basado en LLM.

---

## 🚀 Requisitos Previos

Asegúrate de contar con lo siguiente instalado en tu sistema:
* **Python 3.10+**
* Acceso a Internet (para consumo de APIs de Groq y Google Gemini)
* Git

---

## 🛠️ Configuración del Entorno de Desarrollo

### 1. Clonar el repositorio
```bash
git clone [https://github.com/TU_USUARIO/ISY0101-SaludConecta-IA.git](https://github.com/TU_USUARIO/ISY0101-SaludConecta-IA.git)
cd ISY0101-SaludConecta-IA


2. Crear y activar entorno virtual (Recomendado)

python -m venv venv
.\venv\Scripts\activate



3. Instalar dependencias

pip install -r requirements.txt


4. Configurar Variables de Entorno
Copia el archivo .env.example y renómbralo a .env:

cp .env.example .env



Abre el archivo .env y añade tus claves de API correspondientes:

GROQ_API_KEY=tu_api_key_de_groq
GOOGLE_API_KEY=tu_api_key_de_google_gemini
GOOGLE_EMBEDDING_MODEL=models/gemini-embedding-001
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=tu_api_key_de_langsmith

La clave de Google debe pertenecer a un proyecto habilitado para la Generative
Language API. Si el motor devuelve `403 PERMISSION_DENIED` con el mensaje
`Your project has been denied access`, crea o selecciona otro proyecto en Google
AI Studio/Google Cloud, habilita la API, genera una clave nueva y reemplaza
`GOOGLE_API_KEY` en `.env`. Ese error es una restricción del proyecto o de la
clave, no un problema del PDF ni de FAISS.




 Estructura del Proyecto
Plaintext
ISY0101-SaludConecta-IA/
├── data/
│   └── docs_rag/           # Documentos PDF institucionales para el motor RAG
├── docs/                   # Documentación técnica, diagramas y evidencias
│   ├── arquitectura_sistema.png
│   ├── boceto_interfaz_whatsapp.png
│   └── evidencia_pruebas.png
├── src/
│   ├── agent.py            # Orquestador del Agente Conversacional y System Prompt
│   └── rag_engine.py       # Pipeline RAG (PyPDF, Gemini Embeddings, FAISS)
├── tests/
│   └── test_agent.py       # Pruebas unitarias del sistema (Pytest)
├── .env.example            # Plantilla de variables de entorno
├── README.md               # Instrucciones de uso
└── requirements.txt        # Dependencias de Python





Ejecución de Pruebas Automáticas
Para validar la coherencia del System Prompt, guardrails médicos y reglas del agente, ejecuta la suite de pruebas unitarias con pytest:

Bash
python -m pytest tests/





Ejecución del Sistema
1. Inicializar la Base Vectorial (RAG Engine)
Para procesar la documentación almacenada en data/docs_rag/ e indizarla en FAISS:

Bash
python src/rag_engine.py





Interacción con el Agente Conversacional
Para iniciar la interacción con ConectaBot:

Bash
python src/agent.py




Arquitectura del Sistema
El flujo general integra carga de PDFs, vectorización semántica con Gemini Embeddings, almacenamiento en FAISS e inferencia mediante Groq (Llama 3.3 70B):

Autores y Asignatura
Proyecto: SaludConecta - Solución con IA

Asignatura: ISY0101 - Ingeniería de Soluciones con Inteligencia Artificial

Integrantes: Diego Hernandez, Benjamin Heresmann