"""Herramientas del agente ConectaBot (LangChain @tool).

Consulta:     buscar_protocolos, consultar_citas_paciente, buscar_cupos_disponibles, recordar_paciente
Escritura:    reagendar_cita, cancelar_cita, reasignar_cupo_liberado, enviar_mensaje_paciente,
              guardar_preferencia_paciente, derivar_a_humano
Razonamiento: evaluar_riesgo_inasistencia

Cada herramienta devuelve JSON. Si una regla de negocio falla, devuelve
{"ok": false, "error": ...} para que el agente decida otra alternativa en vez de caerse.
"""
import json
from functools import wraps

from langchain_core.tools import tool

from src.agenda import ErrorAgenda, obtener_agenda
from src.memoria import obtener_memoria


def _json(funcion):
    @wraps(funcion)
    def envoltura(*args, **kwargs):
        try:
            resultado = funcion(*args, **kwargs)
        except ErrorAgenda as e:
            resultado = {"ok": False, "error": str(e)}
        except Exception as e:  # falla externa (API de embeddings, archivo, etc.)
            resultado = {"ok": False, "error": f"Falla técnica: {type(e).__name__}: {e}"}
        return json.dumps(resultado, ensure_ascii=False)
    return envoltura


# ---------------- consulta ----------------
@tool
@_json
def buscar_protocolos(consulta: str) -> str:
    """Busca en los protocolos oficiales de SaludConecta (RAG semántico sobre FAISS).
    Úsala para preguntas sobre preparación de exámenes, ayunos, documentos, políticas de
    cancelación, lista de espera, sedes, horarios o urgencias."""
    from src.rag_engine import buscar
    return {"ok": True, "fragmentos": buscar(consulta, k=3)}


@tool
@_json
def consultar_citas_paciente(paciente_id: str) -> str:
    """Lista las citas vigentes del paciente con fecha, hora, sede, especialidad y tipo."""
    return {"ok": True, "citas": obtener_agenda().citas_de_paciente(paciente_id)}


@tool
@_json
def buscar_cupos_disponibles(especialidad: str, sede: str | None = None, franja: str | None = None) -> str:
    """Busca cupos libres de una especialidad (Laboratorio, Dermatología, Cardiología, Pediatría,
    Imagenología, Medicina General). sede es opcional. franja opcional: 'mañana' o 'tarde'.
    Devuelve los cupos ordenados del más próximo al más lejano."""
    cupos = obtener_agenda().buscar_cupos(especialidad, sede, franja)
    return {"ok": True, "cupos": cupos[:5], "total": len(cupos)}


@tool
@_json
def recordar_paciente(paciente_id: str, tema: str) -> str:
    """Recupera de la memoria de largo plazo lo que se sabe del paciente sobre un tema
    (preferencias de horario o sede, acuerdos de sesiones anteriores). Búsqueda semántica."""
    return {"ok": True, "recuerdos": obtener_memoria().recuperar(paciente_id, tema)}


# ---------------- escritura ----------------
@tool
@_json
def reagendar_cita(cita_id: str, nuevo_cupo_id: str) -> str:
    """Mueve una cita a un cupo libre de la misma especialidad y la deja confirmada.
    Si el cupo ya no está disponible devuelve ok=false: en ese caso busca otro cupo."""
    return obtener_agenda().reagendar(cita_id, nuevo_cupo_id)


@tool
@_json
def reasignar_cupo_liberado(cupo_id: str) -> str:
    """Asigna un cupo liberado al paciente de mayor prioridad en la lista de espera
    (urgencia clínica, sede preferida, días en espera) y le avisa por WhatsApp."""
    agenda = obtener_agenda()
    resultado = agenda.reasignar_cupo(cupo_id)
    cita = resultado["nueva_cita"]
    agenda.enviar_mensaje(cita["paciente_id"],
                          f"SaludConecta: se liberó una hora de {cita['especialidad']} para usted el "
                          f"{cita['fecha']} a las {cita['hora']} en sede {cita['sede']}. Responda SÍ para confirmar.")
    return resultado


@tool
@_json
def cancelar_cita(cita_id: str, motivo: str) -> str:
    """Cancela una cita, informa si fue con 24 horas de anticipación (sin costo) y, en la misma
    operación, reasigna el cupo liberado al paciente de mayor prioridad de la lista de espera.
    Informa al paciente solo lo que diga el resultado (campo cupo_reasignado)."""
    agenda = obtener_agenda()
    resultado = agenda.cancelar(cita_id, motivo)
    resultado.pop("siguiente_paso", None)
    if resultado["candidatos_lista_espera"]:
        # Cancelar y reasignar van juntos: un cupo liberado no debe depender de que el LLM recuerde reasignarlo.
        reasignado = json.loads(reasignar_cupo_liberado.func(resultado["cupo_liberado"]))
        resultado["cupo_reasignado"] = reasignado.get("ok", False)
        resultado["reasignado_a"] = reasignado.get("paciente_asignado", {}).get("paciente_id")
    else:
        resultado["cupo_reasignado"] = False
    return resultado


@tool
@_json
def enviar_mensaje_paciente(paciente_id: str, mensaje: str) -> str:
    """Envía un mensaje al paciente por WhatsApp (simulado: queda en mensajes_enviados.jsonl).
    Úsala para recordatorios, confirmaciones e instrucciones de preparación."""
    return obtener_agenda().enviar_mensaje(paciente_id, mensaje)


@tool
@_json
def guardar_preferencia_paciente(paciente_id: str, preferencia: str) -> str:
    """Guarda en la memoria de largo plazo una preferencia o dato útil del paciente
    (ej. 'prefiere horarios de tarde', 'prefiere sede Viña del Mar'). Persiste entre sesiones."""
    return obtener_memoria().guardar(paciente_id, preferencia, tipo="preferencia")


@tool
@_json
def derivar_a_humano(motivo: str, urgencia: str = "normal") -> str:
    """Deriva el caso a una persona del equipo (recepción o profesional de salud).
    urgencia: 'normal' o 'alta'. Úsala ante consultas clínicas, reclamos o casos que no puedes resolver."""
    destino = "profesional de salud de turno" if urgencia == "alta" else "recepción de la sede"
    return {"ok": True, "derivado_a": destino, "motivo": motivo, "urgencia": urgencia}


# ---------------- razonamiento ----------------
@tool
@_json
def evaluar_riesgo_inasistencia(cita_id: str) -> str:
    """Calcula el riesgo de inasistencia (no-show) de una cita: puntaje 0-1, nivel
    (bajo, medio, alto) y los factores que lo explican."""
    return obtener_agenda().riesgo_inasistencia(cita_id)


HERRAMIENTAS = [
    buscar_protocolos, consultar_citas_paciente, buscar_cupos_disponibles, recordar_paciente,
    reagendar_cita, cancelar_cita, reasignar_cupo_liberado, enviar_mensaje_paciente,
    guardar_preferencia_paciente, derivar_a_humano, evaluar_riesgo_inasistencia,
]
