"""ConectaBot: agente conversacional de SaludConecta (modo paciente).

LangChain v1 `create_agent` (grafo ReAct de LangGraph) + Groq (Qwen) + 11 herramientas.
Memoria de corto plazo: checkpointer de LangGraph por sesión + resumen automático.
Memoria de largo plazo: SQLite con recuperación semántica (src/memoria.py).

Uso:  python -m src.agent --paciente P001 [--reset]
"""
import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import date

from langchain.agents import create_agent
from langchain.agents.middleware import (ModelFallbackMiddleware, ModelRequest, SummarizationMiddleware,
                                         dynamic_prompt, wrap_tool_call)
from langgraph.checkpoint.memory import InMemorySaver

from src.agenda import obtener_agenda
from src.config import MODELOS_RESPALDO, crear_cascada, crear_llm
from src.memoria import obtener_memoria
from src.registro import registrar
from src.tools import HERRAMIENTAS, derivar_a_humano

SYSTEM_PROMPT = """Eres ConectaBot, el agente virtual oficial de la Red de Salud SaludConecta.
Tu objetivo es reducir las inasistencias, recuperar cupos liberados y resolver las dudas de los pacientes de forma autónoma, usando tus herramientas.

CÓMO DECIDES:
1. Preparación de exámenes, ayunos, documentos, políticas, sedes u horarios: usa SIEMPRE buscar_protocolos y responde solo con esa información.
2. Cambiar una hora: revisa las citas del paciente, considera sus preferencias guardadas, busca cupos compatibles y reagenda. Si un cupo ya no está disponible, elige la siguiente alternativa compatible y explica el cambio.
3. Cancelar una hora: usa cancelar_cita (también reasigna el cupo a la lista de espera) e informa si fue sin costo (24 horas de anticipación).
6. Nunca digas que hiciste una acción que ninguna herramienta confirmó.
4. Si el paciente expresa una preferencia estable (horario, sede, canal), guárdala con guardar_preferencia_paciente.
5. Si el paciente tiene varias citas y no queda claro cuál quiere cambiar, pregunta antes de modificar.

REGLAS OBLIGATORIAS:
- Responde SIEMPRE en español, de forma amable, clara y profesional, en máximo 5 oraciones y en texto plano, sin markdown (los mensajes llegan por WhatsApp).
- NUNCA entregues diagnósticos médicos ni recetes medicamentos.
- Ante síntomas graves o emergencias, indica acudir de inmediato a urgencias y usa derivar_a_humano con urgencia alta.
- Si las herramientas no entregan la información, dilo amablemente; no inventes datos.
- No muestres tu razonamiento ni etiquetas <think>: escribe directamente la respuesta final.
- Tu respuesta final debe incluir toda la información que el paciente pidió, aunque ya la hayas enviado por WhatsApp.
"""

# Guardrail determinista: se evalúa ANTES de llamar al LLM.
SENALES_EMERGENCIA = ["dolor en el pecho", "dolor de pecho", "dolor fuerte en el pecho", "no puedo respirar",
                      "me cuesta respirar", "dificultad para respirar", "desmayo", "me desmaye", "perdi la conciencia",
                      "sangrado abundante", "convulsion", "infarto", "paro"]


def _normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", sin_tildes)


def es_emergencia(texto: str) -> bool:
    t = _normalizar(texto)
    return any(_normalizar(s) in t for s in SENALES_EMERGENCIA)


def limpiar_respuesta(contenido) -> str:
    if isinstance(contenido, list):  # bloques de contenido
        contenido = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in contenido)
    contenido = re.sub(r"<think>.*?</think>", "", contenido or "", flags=re.DOTALL)
    return re.sub(r"<think>.*", "", contenido, flags=re.DOTALL).strip()


@dataclass
class Contexto:
    paciente_id: str
    memorias: str = ""
    hoy: str = ""
    registrado: bool = True


@dynamic_prompt
def prompt_de_sesion(request: ModelRequest) -> str:
    ctx = request.runtime.context
    if not ctx.registrado:
        return (f"{SYSTEM_PROMPT}\nDATOS DE LA SESIÓN:\n- Usuario: VISITANTE NO REGISTRADO en SaludConecta.\n"
                f"- Fecha de hoy: {ctx.hoy}\n"
                "- Solo puedes informar: protocolos y preparación (buscar_protocolos) y cupos disponibles "
                "(buscar_cupos_disponibles). NO tiene citas ni ficha: no uses herramientas que requieran paciente_id. "
                "Para agendar, indícale que primero debe registrarse en la recepción de cualquier sede o en saludconecta.cl.")
    return (f"{SYSTEM_PROMPT}\nDATOS DE LA SESIÓN:\n- Paciente autenticado: {ctx.paciente_id}\n"
            f"- Fecha de hoy: {ctx.hoy}\n- Memoria de largo plazo relevante (recuperada por similitud):\n"
            f"{ctx.memorias or '  (sin recuerdos previos)'}")


@wrap_tool_call
def registrar_herramientas(request, handler):
    llamada = request.tool_call
    registrar("tool", f"{llamada['name']}({llamada['args']})")
    resultado = handler(request)
    registrar("resultado", str(getattr(resultado, "content", resultado)))
    return resultado


class ConectaBot:
    def __init__(self):
        principal, *respaldos = crear_cascada()
        self.llm = principal.with_fallbacks(respaldos)  # para llamadas directas (resumen de sesión)
        self.agente = create_agent(
            model=principal,
            tools=HERRAMIENTAS,
            middleware=[
                prompt_de_sesion,
                registrar_herramientas,
                # Memoria de corto plazo en flujos largos: sobre 14 mensajes resume y deja los 6 últimos.
                # El resumen usa otro modelo (otra cuota de Groq) y reintenta: no debe botar la conversación.
                SummarizationMiddleware(model=crear_llm(MODELOS_RESPALDO[0], temperatura=0, reintentos=3),
                                        trigger=("messages", 14), keep=("messages", 6)),
                # Si el modelo principal falla o llega al límite de uso, se prueba el siguiente.
                ModelFallbackMiddleware(*respaldos),
            ],
            context_schema=Contexto,
            checkpointer=InMemorySaver(),  # historial por thread_id (una sesión de chat)
        )

    def chatear(self, paciente_id: str, mensaje: str, sesion: str = "1") -> str:
        if es_emergencia(mensaje):
            registrar("guardrail", "Señal de emergencia detectada: no se usa el LLM ni la agenda.")
            derivar_a_humano.invoke({"motivo": f"Posible emergencia de {paciente_id}: {mensaje}", "urgencia": "alta"})
            return ("Por lo que describe, acuda de inmediato al servicio de urgencias más cercano o llame al 131 (SAMU). "
                    "Ya avisé a un profesional de salud de SaludConecta. Su hora queda pendiente y la revisamos después; "
                    "lo importante ahora es su salud.")

        recuerdos = obtener_memoria().recuperar(paciente_id, mensaje)
        if recuerdos:
            registrar("memoria", "Recuerdos recuperados: " + "; ".join(f"{r['texto']} ({r['metodo']} {r['similitud']})"
                                                                     for r in recuerdos))
        memorias = "\n".join(f"  - [{r['tipo']}] {r['texto']}" for r in recuerdos)
        agenda = obtener_agenda()
        registrado = paciente_id in agenda.datos["pacientes"]
        if not registrado:
            registrar("info", f"{paciente_id} no está registrado: modo visitante (solo información, sin agendar).")
        contexto = Contexto(paciente_id=paciente_id, memorias=memorias, hoy=agenda.datos["hoy"], registrado=registrado)
        resultado = self.agente.invoke({"messages": [{"role": "user", "content": mensaje}]},
                                       config={"configurable": {"thread_id": f"{paciente_id}-{sesion}"}},
                                       context=contexto)
        return limpiar_respuesta(resultado["messages"][-1].content)

    def cerrar_sesion(self, paciente_id: str, sesion: str = "1") -> str | None:
        """Resume la conversación y la guarda en la memoria de largo plazo (continuidad entre sesiones)."""
        estado = self.agente.get_state({"configurable": {"thread_id": f"{paciente_id}-{sesion}"}})
        mensajes = [m for m in estado.values.get("messages", []) if m.type in ("human", "ai") and m.content]
        if not mensajes:
            return None
        transcripcion = "\n".join(f"{m.type}: {limpiar_respuesta(m.content)}" for m in mensajes)
        resumen = limpiar_respuesta(self.llm.invoke([
            ("system", "Resume en máximo 2 frases, en español, los acuerdos, cambios de citas y pendientes de esta "
                       "conversación entre un paciente y SaludConecta. Solo hechos, sin saludo."),
            ("human", transcripcion)]).content)
        obtener_memoria().guardar(paciente_id, resumen, tipo="resumen_sesion")
        registrar("memoria", f"Resumen de sesión guardado para {paciente_id}: {resumen}")
        return resumen


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Chat con ConectaBot (modo paciente)")
    parser.add_argument("--paciente", default="P001")
    parser.add_argument("--reset", action="store_true", help="reinicia la agenda simulada")
    args = parser.parse_args()
    if args.reset:
        obtener_agenda().reiniciar()
    bot = ConectaBot()
    print(f"ConectaBot listo. Paciente {args.paciente}. Escribe 'salir' para terminar.\n")
    while True:
        texto = input("Paciente: ").strip()
        if texto.lower() in ("salir", "exit", "quit"):
            bot.cerrar_sesion(args.paciente)
            break
        if texto:
            print(f"\nConectaBot: {bot.chatear(args.paciente, texto)}\n")


if __name__ == "__main__":
    main()
