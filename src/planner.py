"""Modo coordinador: planificación por prioridades con replanificación (Plan-and-Execute adaptativo).

1. Construye el estado de las citas de las próximas 48 h (con su riesgo de inasistencia).
2. El LLM genera un plan JSON (Pydantic) con acción, objetivo, prioridad 1-5 y dependencias.
   Si el plan no es válido, se usa el planificador por reglas (mismo esquema de prioridades).
3. El ejecutor toma siempre el paso listo de mayor prioridad (dependencias cumplidas).
4. Si un paso falla o llega un evento externo (ej. una cancelación), se replanifica lo pendiente.

Esquema de prioridades:
  1 reasignar_cupo          cupo liberado: capacidad médica que se pierde si no se usa pronto
  2 recordatorio_reforzado  riesgo de inasistencia alto
  2 escalar_call_center     riesgo alto y 3 o más inasistencias previas (después del recordatorio)
  3 enviar_preparacion      exámenes con preparación (evita anulaciones en recepción)
  4 recordatorio_estandar   riesgo medio
  5 recordatorio_estandar   riesgo bajo
"""
import json
from typing import Callable, Literal

from pydantic import BaseModel, Field

from src.agenda import Agenda, ErrorAgenda, obtener_agenda
from src.registro import registrar

Accion = Literal["reasignar_cupo", "recordatorio_reforzado", "escalar_call_center",
                 "enviar_preparacion", "recordatorio_estandar"]
EXAMENES = ("examen de sangre", "ecografía abdominal")


class Paso(BaseModel):
    id: int
    accion: Accion
    objetivo: str = Field(description="cita_id (C###) o, solo para reasignar_cupo, cupo_id (K###)")
    prioridad: int = Field(ge=1, le=5, description="1 = más urgente, 5 = menos urgente")
    depende_de: list[int] = Field(default_factory=list, description="ids de pasos que deben ir antes")
    razon: str = Field(description="por qué este paso y esta prioridad, en una frase")


class Plan(BaseModel):
    pasos: list[Paso]


# ---------------- estado ----------------
def construir_estado(agenda: Agenda, cupos_liberados: list[str]) -> dict:
    citas = []
    for c in agenda.citas_proximas(dias=2):
        riesgo = agenda.riesgo_inasistencia(c["cita_id"])
        paciente = agenda.datos["pacientes"][c["paciente_id"]]
        citas.append({"cita_id": c["cita_id"], "paciente_id": c["paciente_id"], "fecha": c["fecha"],
                      "hora": c["hora"], "especialidad": c["especialidad"], "tipo": c["tipo"],
                      "confirmada": c["confirmada"], "riesgo": riesgo["nivel"], "puntaje": riesgo["puntaje"],
                      "inasistencias_previas": paciente["inasistencias_previas"]})
    libres = [{"cupo_id": k, **agenda.datos["cupos"][k],
               "candidatos_lista_espera": len(agenda.candidatos_lista_espera(k))}
              for k in cupos_liberados if agenda.datos["cupos"][k]["estado"] == "libre"]
    return {"citas": sorted(citas, key=lambda c: (c["fecha"], c["hora"])), "cupos_liberados": libres}


# ---------------- planificadores ----------------
def planificar_reglas(estado: dict) -> Plan:
    pasos, n = [], 0
    for cupo in estado["cupos_liberados"]:
        n += 1
        pasos.append(Paso(id=n, accion="reasignar_cupo", objetivo=cupo["cupo_id"], prioridad=1,
                          razon="Cupo liberado: se reasigna a la lista de espera antes de perder la hora."))
    for c in estado["citas"]:
        if c["confirmada"]:
            continue
        n += 1
        if c["riesgo"] == "alto":
            pasos.append(Paso(id=n, accion="recordatorio_reforzado", objetivo=c["cita_id"], prioridad=2,
                              razon=f"Riesgo alto ({c['puntaje']}): pedir confirmación explícita."))
            if c["inasistencias_previas"] >= 3:
                n += 1
                pasos.append(Paso(id=n, accion="escalar_call_center", objetivo=c["cita_id"], prioridad=2,
                                  depende_de=[n - 1], razon="Historial de 3 o más inasistencias: llamada humana."))
        else:
            prioridad = 4 if c["riesgo"] == "medio" else 5
            pasos.append(Paso(id=n, accion="recordatorio_estandar", objetivo=c["cita_id"], prioridad=prioridad,
                              razon=f"Riesgo {c['riesgo']} ({c['puntaje']}): recordatorio normal."))
        if c["tipo"] in EXAMENES:
            n += 1
            pasos.append(Paso(id=n, accion="enviar_preparacion", objetivo=c["cita_id"], prioridad=3,
                              razon=f"{c['tipo']}: enviar preparación desde los protocolos."))
    return Plan(pasos=pasos)


PROMPT_PLANIFICADOR = """Eres el planificador del agente coordinador de agenda de SaludConecta.
Objetivo: {objetivo}
Genera un plan de pasos para las citas y cupos del ESTADO, usando solo estas acciones:
- reasignar_cupo (objetivo = cupo_id K###): para cada cupo liberado. Prioridad 1.
- recordatorio_reforzado (objetivo = cita_id): citas NO confirmadas con riesgo alto. Prioridad 2.
- escalar_call_center (objetivo = cita_id): riesgo alto y 3 o más inasistencias previas; depende del recordatorio reforzado de esa cita. Prioridad 2.
- enviar_preparacion (objetivo = cita_id): citas de tipo examen de sangre o ecografía, sin dependencias. Prioridad 3.
- recordatorio_estandar (objetivo = cita_id): citas NO confirmadas con riesgo medio (prioridad 4) o bajo (prioridad 5).
No incluyas pasos sobre citas confirmadas. Ids de paso correlativos desde 1. Usa solo ids que aparecen en el ESTADO.
{ya_hecho}
ESTADO:
{estado}"""


def validar(plan: Plan, estado: dict) -> list[str]:
    citas = {c["cita_id"] for c in estado["citas"]}
    cupos = {c["cupo_id"] for c in estado["cupos_liberados"]}
    errores = []
    for p in plan.pasos:
        validos = cupos if p.accion == "reasignar_cupo" else citas
        if p.objetivo not in validos:
            errores.append(f"paso {p.id}: objetivo {p.objetivo} no válido para {p.accion}")
    faltan = cupos - {p.objetivo for p in plan.pasos if p.accion == "reasignar_cupo"}
    if faltan:
        errores.append(f"faltan cupos liberados por reasignar: {sorted(faltan)}")
    return errores


def prioridad_esperada(paso: Paso, riesgos: dict[str, str]) -> int:
    fija = {"reasignar_cupo": 1, "recordatorio_reforzado": 2, "escalar_call_center": 2, "enviar_preparacion": 3}
    if paso.accion in fija:
        return fija[paso.accion]
    return 4 if riesgos.get(paso.objetivo) == "medio" else 5


def normalizar(plan: Plan, estado: dict, ya_hecho: list[str] | None = None) -> tuple[Plan, list[str]]:
    """Guardrail del plan del LLM: corrige prioridades fuera del esquema y agrega recordatorios faltantes."""
    riesgos = {c["cita_id"]: c["riesgo"] for c in estado["citas"]}
    ajustes = []
    for p in plan.pasos:
        esperada = prioridad_esperada(p, riesgos)
        if p.prioridad != esperada:
            ajustes.append(f"{p.accion} {p.objetivo}: prioridad {p.prioridad} -> {esperada}")
            p.prioridad = esperada
    con_recordatorio = {p.objetivo for p in plan.pasos if p.accion.startswith("recordatorio")}
    con_recordatorio |= {h.split()[1] for h in (ya_hecho or []) if h.startswith("recordatorio")}
    siguiente = max((p.id for p in plan.pasos), default=0) + 1
    for c in estado["citas"]:
        if not c["confirmada"] and c["cita_id"] not in con_recordatorio:
            accion = "recordatorio_reforzado" if c["riesgo"] == "alto" else "recordatorio_estandar"
            nuevo = Paso(id=siguiente, accion=accion, objetivo=c["cita_id"], prioridad=1,
                         razon=f"Agregado: faltaba recordatorio (riesgo {c['riesgo']}).")
            nuevo.prioridad = prioridad_esperada(nuevo, riesgos)
            plan.pasos.append(nuevo)
            ajustes.append(f"agregado {accion} {c['cita_id']}")
            siguiente += 1
    return plan, ajustes


def planificar(objetivo: str, estado: dict, llm=None, ya_hecho: list[str] | None = None) -> tuple[Plan, str]:
    if llm is not None:
        try:
            nota = ("Pasos YA EJECUTADOS (no repetir): " + "; ".join(ya_hecho)) if ya_hecho else ""
            modelos = llm if isinstance(llm, list) else [llm]
            cadena = modelos[0].with_structured_output(Plan).with_fallbacks(
                [m.with_structured_output(Plan) for m in modelos[1:]])
            plan = cadena.invoke(PROMPT_PLANIFICADOR.format(
                objetivo=objetivo, ya_hecho=nota, estado=json.dumps(estado, ensure_ascii=False, indent=1)))
            errores = validar(plan, estado)
            if not errores:
                if ya_hecho:  # no repetir pasos ya ejecutados
                    plan.pasos = [p for p in plan.pasos if f"{p.accion} {p.objetivo}" not in ya_hecho]
                plan, ajustes = normalizar(plan, estado, ya_hecho)
                if ajustes:
                    registrar("plan", "Ajustes al plan del LLM según el esquema de prioridades: " + "; ".join(ajustes))
                return plan, "LLM"
            registrar("fallo", "Plan del LLM inválido, uso planificador por reglas: " + "; ".join(errores))
        except Exception as e:
            registrar("fallo", f"El LLM no entregó un plan ({type(e).__name__}); uso planificador por reglas.")
    return planificar_reglas(estado), "reglas"


# ---------------- ejecución ----------------
def _texto_cita(c: dict) -> str:
    return f"{c['especialidad']} ({c['tipo']}) el {c['fecha']} a las {c['hora']} en sede {c['sede']}"


def ejecutar_accion(agenda: Agenda, accion: str, objetivo: str) -> str:
    if accion == "reasignar_cupo":
        r = agenda.reasignar_cupo(objetivo)
        c = r["nueva_cita"]
        agenda.enviar_mensaje(c["paciente_id"], f"SaludConecta: se liberó una hora de {_texto_cita(c)}. "
                                                f"Responda SÍ para confirmar.")
        return f"Cupo {objetivo} asignado a {c['paciente_id']} (urgencia {r['paciente_asignado']['urgencia']})."
    c = agenda.detalle_cita(objetivo)
    if c["estado"] == "cancelada":
        raise ErrorAgenda(f"La cita {objetivo} fue cancelada; el paso ya no aplica.")
    if accion == "recordatorio_reforzado":
        agenda.enviar_mensaje(c["paciente_id"], f"SaludConecta: tiene hora de {_texto_cita(c)}. Es importante que "
                              "responda SÍ para confirmar o NO para liberarla a otro paciente que la espera.")
        return f"Recordatorio reforzado a {c['paciente_id']}."
    if accion == "recordatorio_estandar":
        agenda.enviar_mensaje(c["paciente_id"], f"SaludConecta: le recordamos su hora de {_texto_cita(c)}.")
        return f"Recordatorio estándar a {c['paciente_id']}."
    if accion == "escalar_call_center":
        return f"Cita {objetivo} derivada al call center para llamada de confirmación."
    if accion == "enviar_preparacion":
        try:
            from src.rag_engine import buscar
            indicacion = buscar(f"preparación para {c['tipo']}", k=1)[0]["texto"]
            fuente = "RAG"
        except Exception:
            indicacion = "Revise las indicaciones de preparación en saludconecta.cl o llame a su sede."
            fuente = "texto de respaldo (RAG no disponible)"
        agenda.enviar_mensaje(c["paciente_id"], f"SaludConecta, preparación para su {c['tipo']}: {indicacion}")
        return f"Preparación enviada a {c['paciente_id']} ({fuente})."
    raise ErrorAgenda(f"Acción desconocida: {accion}")


def mostrar_plan(plan: Plan, origen: str):
    registrar("plan", f"Plan generado por {origen} con {len(plan.pasos)} pasos (prioridad 1 = más urgente):",
              datos=plan.model_dump())
    for p in sorted(plan.pasos, key=lambda p: (p.prioridad, p.id)):
        deps = f" tras {p.depende_de}" if p.depende_de else ""
        print(f"      P{p.prioridad}  #{p.id:<2} {p.accion:<23} {p.objetivo}{deps}  - {p.razon}")


def ejecutar(objetivo: str = "Gestionar las citas de las próximas 48 horas para reducir inasistencias",
             eventos: dict[int, Callable[[Agenda], str]] | None = None, llm=None,
             agenda: Agenda | None = None, max_replan: int = 3) -> dict:
    """Ejecuta el plan. `eventos` simula cambios del entorno: {n° de paso: función(agenda) -> cupo liberado}."""
    agenda = agenda or obtener_agenda()
    eventos = dict(eventos or {})
    cupos_liberados: list[str] = []
    plan, origen = planificar(objetivo, construir_estado(agenda, cupos_liberados), llm)
    mostrar_plan(plan, origen)
    pendientes, hechos, completados, replans, n = list(plan.pasos), [], set(), 0, 0

    while pendientes:
        ids_pendientes = {p.id for p in pendientes}
        listos = [p for p in pendientes if not (set(p.depende_de) & ids_pendientes)] or pendientes
        paso = min(listos, key=lambda p: (p.prioridad, p.id))
        pendientes.remove(paso)
        n += 1
        motivo_replan = None
        try:
            resultado = ejecutar_accion(agenda, paso.accion, paso.objetivo)
            registrar("paso", f"{n}. P{paso.prioridad} {paso.accion} {paso.objetivo}: {resultado}")
            hechos.append(f"{paso.accion} {paso.objetivo}")
            completados.add(paso.id)
        except ErrorAgenda as e:
            registrar("fallo", f"{n}. {paso.accion} {paso.objetivo}: {e}")
            motivo_replan = str(e)
        if n in eventos:
            descripcion, cupo = eventos.pop(n)(agenda)
            registrar("evento", f"Cambio en el entorno: {descripcion}")
            cupos_liberados.append(cupo)
            motivo_replan = descripcion
        if motivo_replan and replans < max_replan:
            replans += 1
            estado = construir_estado(agenda, cupos_liberados)
            nuevo, origen = planificar(objetivo, estado, llm, ya_hecho=hechos)
            nuevo.pasos = [p for p in nuevo.pasos if f"{p.accion} {p.objetivo}" not in hechos]
            registrar("replan", f"Replanificación {replans} por: {motivo_replan}")
            mostrar_plan(nuevo, origen)
            pendientes = list(nuevo.pasos)
    registrar("info", f"Fin: {len(hechos)} pasos ejecutados, {replans} replanificaciones.")
    return {"hechos": hechos, "replanificaciones": replans}


def evento_cancelacion(cita_id: str, motivo: str):
    """Fabrica un evento externo: el paciente cancela por otro canal mientras corre el plan."""
    def aplicar(agenda: Agenda):
        r = agenda.cancelar(cita_id, motivo)
        paciente = agenda.datos["citas"][cita_id]["paciente_id"]
        return f"{paciente} canceló {cita_id} ({motivo}); se liberó {r['cupo_liberado']}.", r["cupo_liberado"]
    return aplicar
