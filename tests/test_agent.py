"""Pruebas sin conexión (no llaman a Groq ni a Gemini): herramientas, prioridades,
replanificación, memoria y guardrails. Ejecutar:  python -m pytest -v"""
import json
from datetime import datetime

import pytest

from src.agenda import ErrorAgenda
from src.agent import SYSTEM_PROMPT, es_emergencia, limpiar_respuesta
from src.planner import construir_estado, ejecutar, evento_cancelacion, planificar_reglas
from src.tools import HERRAMIENTAS, cancelar_cita, reagendar_cita


# ---------- prompt y guardrails ----------
def test_prompt_mantiene_reglas_de_la_ep1():
    assert "ConectaBot" in SYSTEM_PROMPT
    assert "NUNCA entregues diagnósticos" in SYSTEM_PROMPT
    assert "urgencias" in SYSTEM_PROMPT
    assert "<think>" in SYSTEM_PROMPT


@pytest.mark.parametrize("texto", ["Tengo un dolor fuerte en el pecho", "me cuesta respirar",
                                   "Ayer me desmayé en la casa"])
def test_detecta_emergencias(texto):
    assert es_emergencia(texto)


def test_no_confunde_consultas_normales_con_emergencias():
    assert not es_emergencia("¿Puedo tomar café antes del examen de sangre?")


def test_limpia_etiquetas_think():
    assert limpiar_respuesta("<think>razono</think>Hola, su hora es mañana.") == "Hola, su hora es mañana."


def test_agente_tiene_herramientas_de_consulta_escritura_y_razonamiento():
    nombres = {h.name for h in HERRAMIENTAS}
    assert {"buscar_protocolos", "consultar_citas_paciente"} <= nombres          # consulta
    assert {"reagendar_cita", "cancelar_cita", "enviar_mensaje_paciente"} <= nombres  # escritura
    assert "evaluar_riesgo_inasistencia" in nombres                               # razonamiento


# ---------- herramientas de agenda ----------
def test_riesgo_alto_para_paciente_con_historial_de_inasistencias(agenda):
    r = agenda.riesgo_inasistencia("C002")  # P002: 3 de 6 inasistencias, 24 años, reservó con 30 días
    assert r["nivel"] == "alto" and r["puntaje"] == 0.75


def test_riesgo_bajo_para_paciente_cumplidor(agenda):
    assert agenda.riesgo_inasistencia("C003")["nivel"] == "bajo"


def test_buscar_cupos_filtra_por_franja(agenda):
    cupos = agenda.buscar_cupos("Laboratorio", franja="tarde")
    assert [c["cupo_id"] for c in cupos] == ["K002", "K003", "K004"]


def test_cancelar_libera_cupo_y_prioriza_lista_de_espera(agenda):
    r = agenda.cancelar("C007", "viaje", ahora=datetime(2026, 10, 23, 18, 0))
    assert r["cupo_liberado"] == "K021" and r["sin_costo"]
    assert r["candidatos_lista_espera"][0]["paciente_id"] == "P009"  # urgencia 1 antes que 60 días de espera


def test_reasignar_cupo_al_paciente_mas_prioritario(agenda):
    agenda.cancelar("C007", "viaje")
    r = agenda.reasignar_cupo("K021")
    assert r["paciente_asignado"]["paciente_id"] == "P009"
    assert agenda.datos["cupos"]["K021"]["estado"] == "ocupado"


def test_cupo_tomado_por_otro_canal_devuelve_error_controlado(agenda):
    agenda.cupos_en_disputa.add("K002")
    resultado = json.loads(reagendar_cita.invoke({"cita_id": "C001", "nuevo_cupo_id": "K002"}))
    assert resultado["ok"] is False and "otro canal" in resultado["error"]
    ok = json.loads(reagendar_cita.invoke({"cita_id": "C001", "nuevo_cupo_id": "K003"}))
    assert ok["ok"] and ok["cita"]["hora"] == "16:00"


def test_no_se_cancela_dos_veces(agenda):
    json.loads(cancelar_cita.invoke({"cita_id": "C002", "motivo": "trabajo"}))
    with pytest.raises(ErrorAgenda):
        agenda.cancelar("C002", "otra vez")


def test_mensajes_quedan_registrados(agenda):
    agenda.enviar_mensaje("P001", "Recordatorio")
    lineas = agenda.ruta_outbox.read_text(encoding="utf-8").splitlines()
    assert json.loads(lineas[-1])["paciente_id"] == "P001"


# ---------- planificación ----------
def test_plan_por_reglas_ordena_por_prioridad(agenda):
    plan = planificar_reglas(construir_estado(agenda, []))
    reforzados = {p.objetivo for p in plan.pasos if p.accion == "recordatorio_reforzado"}
    assert reforzados == {"C002", "C006"}                          # riesgo alto
    escalados = [p for p in plan.pasos if p.accion == "escalar_call_center"]
    assert all(p.depende_de for p in escalados)                    # después del recordatorio
    assert {p.objetivo for p in plan.pasos if p.accion == "enviar_preparacion"} == {"C001", "C005"}


def test_replanifica_ante_cancelacion_y_reasigna_primero(agenda):
    resultado = ejecutar(eventos={2: evento_cancelacion("C007", "viaje")}, llm=None, agenda=agenda)
    assert resultado["replanificaciones"] == 1
    hechos = resultado["hechos"]
    assert hechos[2] == "reasignar_cupo K021"                       # prioridad 1 apenas se libera
    assert "recordatorio_estandar C007" not in hechos              # la cita cancelada sale del plan


# ---------- memoria de largo plazo ----------
def test_memoria_persiste_entre_instancias(memoria, tmp_path):
    from src.memoria import MemoriaLargoPlazo
    memoria.guardar("P001", "Prefiere horarios de tarde")
    otra = MemoriaLargoPlazo(tmp_path / "memoria.db", embeddings=None)  # simula reinicio del agente
    recuerdos = otra.recuperar("P001", "¿qué horario de tarde le acomoda?")
    assert recuerdos and recuerdos[0]["texto"] == "Prefiere horarios de tarde"


def test_memoria_separa_pacientes(memoria):
    memoria.guardar("P001", "Prefiere sede Viña del Mar")
    assert memoria.recuperar("P002", "sede") == []


class EmbeddingsFalsos:
    """Vectores simples para probar la similitud coseno sin llamar a Gemini."""
    def embed_query(self, texto):
        t = texto.lower()
        return [1.0 if "tarde" in t else 0.0, 1.0 if "sede" in t else 0.0, 0.1]


def test_recuperacion_semantica_por_coseno(tmp_path):
    from src.memoria import MemoriaLargoPlazo
    m = MemoriaLargoPlazo(tmp_path / "m.db", embeddings=EmbeddingsFalsos())
    m.guardar("P001", "Prefiere la sede de Viña")
    m.guardar("P001", "Prefiere horarios de tarde")
    top = m.recuperar("P001", "¿a qué hora de la tarde?")[0]
    assert top["texto"] == "Prefiere horarios de tarde" and top["metodo"] == "coseno"


# ---------- ciclo completo del agente con un LLM simulado ----------
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402


class LLMGuionado(BaseChatModel):
    """Devuelve respuestas fijas en orden; registra lo que recibe. Sin red."""
    respuestas: list = []
    recibido: list = []
    i: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.recibido.append(messages)
        respuesta = self.respuestas[min(self.i, len(self.respuestas) - 1)]
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=respuesta)])

    @property
    def _llm_type(self):
        return "guionado"

    def bind_tools(self, tools, **kwargs):
        return self


def test_agente_se_recupera_de_cupo_tomado_y_usa_memoria(agenda, memoria, monkeypatch):
    import src.agent as agente
    memoria.guardar("P001", "Prefiere horarios de tarde")
    agenda.cupos_en_disputa.add("K002")
    llm = LLMGuionado(respuestas=[
        AIMessage(content="", tool_calls=[{"name": "reagendar_cita", "id": "1",
                                           "args": {"cita_id": "C001", "nuevo_cupo_id": "K002"}}]),
        AIMessage(content="", tool_calls=[{"name": "reagendar_cita", "id": "2",
                                           "args": {"cita_id": "C001", "nuevo_cupo_id": "K003"}}]),
        AIMessage(content="<think>...</think>Listo, quedó a las 16:00."),
    ])
    monkeypatch.setattr(agente, "crear_cascada", lambda *a, **k: [llm, llm])
    bot = agente.ConectaBot()
    respuesta = bot.chatear("P001", "Cámbiame el examen a la tarde")
    assert respuesta == "Listo, quedó a las 16:00."
    assert agenda.detalle_cita("C001")["hora"] == "16:00"
    prompt_sistema = [m for m in llm.recibido[0] if m.type == "system"][0].content
    assert "P001" in prompt_sistema and "Prefiere horarios de tarde" in prompt_sistema


def test_guardrail_no_llama_al_llm(agenda, memoria, monkeypatch):
    import src.agent as agente
    llm = LLMGuionado(respuestas=[AIMessage(content="no debería usarse")])
    monkeypatch.setattr(agente, "crear_cascada", lambda *a, **k: [llm, llm])
    respuesta = agente.ConectaBot().chatear("P003", "Tengo dolor en el pecho")
    assert "urgencias" in respuesta and llm.i == 0


def test_normaliza_prioridades_y_completa_recordatorios_del_plan_llm(agenda):
    from src.planner import Paso, Plan, normalizar
    estado = construir_estado(agenda, [])
    plan_llm = Plan(pasos=[Paso(id=1, accion="recordatorio_estandar", objetivo="C003", prioridad=4, razon="x")])
    plan, ajustes = normalizar(plan_llm, estado)
    assert plan.pasos[0].prioridad == 5                       # C003 es riesgo bajo
    objetivos = {p.objetivo for p in plan.pasos if p.accion.startswith("recordatorio")}
    assert {"C001", "C002", "C005", "C006"} <= objetivos      # se agregaron los que faltaban
    assert any("prioridad 4 -> 5" in a for a in ajustes)


def test_cancelar_cita_reasigna_el_cupo_en_la_misma_operacion(agenda):
    r = json.loads(cancelar_cita.invoke({"cita_id": "C007", "motivo": "viaje"}))
    assert r["cupo_reasignado"] is True and r["reasignado_a"] == "P009"
    assert agenda.datos["cupos"]["K021"]["estado"] == "ocupado"
