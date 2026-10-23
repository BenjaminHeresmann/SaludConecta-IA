"""Sistema de agenda simulado (reemplaza al HIS/ERP de SaludConecta).

El estado vive en data/runtime/agenda_estado.json y se reinicia desde
data/agenda_seed.json. Todas las operaciones validan reglas de negocio y
lanzan ErrorAgenda con un mensaje claro cuando no se pueden cumplir.
"""
import json
from datetime import date, datetime, timedelta
from pathlib import Path

from src.config import DATA, RUNTIME

SEED = DATA / "agenda_seed.json"
ESTADO = RUNTIME / "agenda_estado.json"
OUTBOX = RUNTIME / "mensajes_enviados.jsonl"


class ErrorAgenda(Exception):
    """Regla de negocio incumplida (cupo tomado, cita inexistente, etc.)."""


class Agenda:
    def __init__(self, ruta_estado: Path = ESTADO, ruta_outbox: Path = OUTBOX):
        self.ruta_estado = Path(ruta_estado)
        self.ruta_outbox = Path(ruta_outbox)
        # Cupos que otro canal (call center, web) toma justo antes que el agente.
        # Simula la condición cambiante del escenario 3 de la demo.
        self.cupos_en_disputa: set[str] = set()
        if not self.ruta_estado.exists():
            self.reiniciar()
        self.datos = json.loads(self.ruta_estado.read_text(encoding="utf-8"))

    # ---------- persistencia ----------
    def reiniciar(self, hoy: date | None = None):
        """Carga los datos semilla y convierte el desfase 'dia' en fechas reales."""
        hoy = hoy or date.today()
        datos = json.loads(SEED.read_text(encoding="utf-8"))
        datos.pop("_nota", None)
        for cupo in datos["cupos"].values():
            cupo["fecha"] = (hoy + timedelta(days=cupo.pop("dia"))).isoformat()
        datos["hoy"] = hoy.isoformat()
        self.ruta_estado.parent.mkdir(parents=True, exist_ok=True)
        self.ruta_estado.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.ruta_outbox.exists():
            self.ruta_outbox.unlink()
        self.datos = datos
        self.cupos_en_disputa = set()

    def _guardar(self):
        self.ruta_estado.write_text(json.dumps(self.datos, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- consultas ----------
    def _cita(self, cita_id: str) -> dict:
        cita = self.datos["citas"].get(cita_id)
        if not cita:
            raise ErrorAgenda(f"La cita {cita_id} no existe.")
        return cita

    def _paciente(self, paciente_id: str) -> dict:
        paciente = self.datos["pacientes"].get(paciente_id)
        if not paciente:
            raise ErrorAgenda(f"El paciente {paciente_id} no existe.")
        return paciente

    def detalle_cita(self, cita_id: str) -> dict:
        cita = self._cita(cita_id)
        cupo = self.datos["cupos"][cita["cupo_id"]]
        return {"cita_id": cita_id, **cita, "sede": cupo["sede"], "especialidad": cupo["especialidad"],
                "fecha": cupo["fecha"], "hora": cupo["hora"], "medico": cupo["medico"]}

    def citas_de_paciente(self, paciente_id: str) -> list[dict]:
        self._paciente(paciente_id)
        return [self.detalle_cita(cid) for cid, c in self.datos["citas"].items()
                if c["paciente_id"] == paciente_id and c["estado"] != "cancelada"]

    def citas_proximas(self, dias: int = 2) -> list[dict]:
        """Citas vigentes entre mañana y los próximos `dias` días."""
        hoy = date.fromisoformat(self.datos["hoy"])
        limite = hoy + timedelta(days=dias)
        return [self.detalle_cita(cid) for cid, c in self.datos["citas"].items()
                if c["estado"] != "cancelada"
                and hoy < date.fromisoformat(self.datos["cupos"][c["cupo_id"]]["fecha"]) <= limite]

    def buscar_cupos(self, especialidad: str, sede: str | None = None, franja: str | None = None) -> list[dict]:
        """Cupos libres ordenados por fecha y hora. franja: 'mañana' (<13:00) o 'tarde' (>=13:00)."""
        resultado = []
        for cid, cupo in self.datos["cupos"].items():
            if cupo["estado"] != "libre" or cupo["especialidad"].lower() != especialidad.lower():
                continue
            if sede and cupo["sede"].lower() != sede.lower():
                continue
            hora = int(cupo["hora"][:2])
            if franja == "mañana" and hora >= 13 or franja == "tarde" and hora < 13:
                continue
            resultado.append({"cupo_id": cid, **cupo})
        return sorted(resultado, key=lambda c: (c["fecha"], c["hora"]))

    # ---------- razonamiento: riesgo de inasistencia ----------
    def riesgo_inasistencia(self, cita_id: str) -> dict:
        """Puntaje 0-1 por reglas (proxy del clasificador de no-show propuesto en la EP1).

        0.50 x tasa histórica de inasistencia
        + 0.20 si reservó con más de 14 días de anticipación
        + 0.15 si tiene entre 18 y 30 años
        + 0.15 si la cita aún no está confirmada
        """
        cita = self._cita(cita_id)
        paciente = self._paciente(cita["paciente_id"])
        tasa = paciente["inasistencias_previas"] / max(paciente["citas_previas"], 1)
        factores = {
            "tasa_historica": round(0.5 * tasa, 3),
            "anticipacion_mayor_14_dias": 0.2 if cita["dias_anticipacion_reserva"] > 14 else 0.0,
            "edad_18_30": 0.15 if 18 <= paciente["edad"] <= 30 else 0.0,
            "sin_confirmar": 0.15 if not cita["confirmada"] else 0.0,
        }
        puntaje = round(sum(factores.values()), 3)
        nivel = "alto" if puntaje >= 0.5 else "medio" if puntaje >= 0.3 else "bajo"
        return {"cita_id": cita_id, "paciente_id": cita["paciente_id"], "puntaje": puntaje,
                "nivel": nivel, "factores": factores}

    # ---------- escritura ----------
    def reagendar(self, cita_id: str, nuevo_cupo_id: str) -> dict:
        cita = self._cita(cita_id)
        if cita["estado"] == "cancelada":
            raise ErrorAgenda(f"La cita {cita_id} está cancelada; no se puede reagendar.")
        nuevo = self.datos["cupos"].get(nuevo_cupo_id)
        if not nuevo:
            raise ErrorAgenda(f"El cupo {nuevo_cupo_id} no existe.")
        if nuevo_cupo_id in self.cupos_en_disputa:
            self.cupos_en_disputa.discard(nuevo_cupo_id)
            nuevo["estado"] = "ocupado"
            self._guardar()
            raise ErrorAgenda(f"El cupo {nuevo_cupo_id} fue tomado por otro canal hace instantes. Busca otra alternativa.")
        if nuevo["estado"] != "libre":
            raise ErrorAgenda(f"El cupo {nuevo_cupo_id} ya no está disponible.")
        anterior = self.datos["cupos"][cita["cupo_id"]]
        if nuevo["especialidad"] != anterior["especialidad"]:
            raise ErrorAgenda("El nuevo cupo es de otra especialidad.")
        anterior["estado"] = "libre"
        nuevo["estado"] = "ocupado"
        cupo_anterior = cita["cupo_id"]
        cita["cupo_id"] = nuevo_cupo_id
        cita["confirmada"] = True
        self._guardar()
        return {"ok": True, "cita": self.detalle_cita(cita_id), "cupo_liberado": cupo_anterior}

    def cancelar(self, cita_id: str, motivo: str, ahora: datetime | None = None) -> dict:
        cita = self._cita(cita_id)
        if cita["estado"] == "cancelada":
            raise ErrorAgenda(f"La cita {cita_id} ya estaba cancelada.")
        cupo = self.datos["cupos"][cita["cupo_id"]]
        inicio = datetime.fromisoformat(f"{cupo['fecha']}T{cupo['hora']}")
        horas = (inicio - (ahora or datetime.now())).total_seconds() / 3600
        cita["estado"] = "cancelada"
        cita["motivo_cancelacion"] = motivo
        cupo["estado"] = "libre"
        self._guardar()
        return {"ok": True, "cita_id": cita_id, "cupo_liberado": cita["cupo_id"],
                "horas_de_anticipacion": round(horas, 1), "sin_costo": horas >= 24,
                "candidatos_lista_espera": self.candidatos_lista_espera(cita["cupo_id"])}

    def candidatos_lista_espera(self, cupo_id: str) -> list[dict]:
        """Lista de espera de la especialidad del cupo, ordenada por prioridad:
        1) urgencia clínica (1 = alta), 2) misma sede preferida, 3) más días esperando."""
        cupo = self.datos["cupos"][cupo_id]
        candidatos = [e for e in self.datos["lista_espera"] if e["especialidad"] == cupo["especialidad"]]
        return sorted(candidatos, key=lambda e: (e["urgencia"], e["sede_preferida"] != cupo["sede"], -e["dias_en_espera"]))

    def reasignar_cupo(self, cupo_id: str) -> dict:
        """Asigna un cupo libre al primer paciente de la lista de espera priorizada."""
        cupo = self.datos["cupos"].get(cupo_id)
        if not cupo or cupo["estado"] != "libre":
            raise ErrorAgenda(f"El cupo {cupo_id} no está libre.")
        candidatos = self.candidatos_lista_espera(cupo_id)
        if not candidatos:
            raise ErrorAgenda(f"No hay pacientes en lista de espera para {cupo['especialidad']}.")
        elegido = candidatos[0]
        self.datos["lista_espera"].remove(elegido)
        nueva_id = f"C{len(self.datos['citas']) + 1:03d}"
        self.datos["citas"][nueva_id] = {"paciente_id": elegido["paciente_id"], "cupo_id": cupo_id,
                                         "tipo": "consulta", "estado": "agendada", "confirmada": False,
                                         "dias_anticipacion_reserva": 0, "origen": "lista_espera"}
        cupo["estado"] = "ocupado"
        self._guardar()
        return {"ok": True, "nueva_cita": self.detalle_cita(nueva_id), "paciente_asignado": elegido,
                "criterio": "urgencia clínica, luego sede preferida, luego días en espera"}

    def confirmar(self, cita_id: str) -> dict:
        cita = self._cita(cita_id)
        cita["confirmada"] = True
        self._guardar()
        return {"ok": True, "cita": self.detalle_cita(cita_id)}

    def enviar_mensaje(self, paciente_id: str, mensaje: str, canal: str = "WhatsApp") -> dict:
        """Simula el envío por WhatsApp Business: queda registrado en mensajes_enviados.jsonl."""
        self._paciente(paciente_id)
        registro = {"fecha": datetime.now().isoformat(timespec="seconds"), "paciente_id": paciente_id,
                    "canal": canal, "mensaje": mensaje}
        self.ruta_outbox.parent.mkdir(parents=True, exist_ok=True)
        with self.ruta_outbox.open("a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        return {"ok": True, **registro}


_agenda: Agenda | None = None


def obtener_agenda() -> Agenda:
    global _agenda
    if _agenda is None:
        _agenda = Agenda()
    return _agenda


def usar_agenda(agenda: Agenda):
    """Permite inyectar otra agenda (tests)."""
    global _agenda
    _agenda = agenda
