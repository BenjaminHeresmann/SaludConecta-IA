"""Memoria de largo plazo del agente (persistente entre sesiones).

Guarda en SQLite hechos del paciente (preferencias, acuerdos, resúmenes de sesión)
junto con su embedding. La recuperación es semántica: se compara la consulta con
cada recuerdo por similitud coseno. Si no hay embeddings disponibles, se usa
coincidencia de palabras como respaldo.
"""
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from src.config import RUNTIME

RUTA_DB = RUNTIME / "memoria_largo_plazo.db"


class MemoriaLargoPlazo:
    def __init__(self, ruta: Path = RUTA_DB, embeddings=None):
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings
        with self._conectar() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS memorias (
                id INTEGER PRIMARY KEY AUTOINCREMENT, paciente_id TEXT, tipo TEXT,
                texto TEXT, creado TEXT, vector TEXT)""")

    def _conectar(self):
        return sqlite3.connect(self.ruta)

    def _vector(self, texto: str):
        if self.embeddings is None:
            return None
        try:
            return self.embeddings.embed_query(texto)
        except Exception:
            return None  # sin red o sin clave: se guarda igual y se busca por palabras

    def guardar(self, paciente_id: str, texto: str, tipo: str = "preferencia") -> dict:
        vector = self._vector(texto)
        creado = datetime.now().isoformat(timespec="seconds")
        with self._conectar() as con:
            con.execute("INSERT INTO memorias (paciente_id, tipo, texto, creado, vector) VALUES (?, ?, ?, ?, ?)",
                        (paciente_id, tipo, texto, creado, json.dumps(vector) if vector else None))
        return {"ok": True, "paciente_id": paciente_id, "tipo": tipo, "texto": texto, "creado": creado}

    def recuperar(self, paciente_id: str, consulta: str, k: int = 3) -> list[dict]:
        with self._conectar() as con:
            filas = con.execute("SELECT tipo, texto, creado, vector FROM memorias WHERE paciente_id = ?",
                                (paciente_id,)).fetchall()
        if not filas:
            return []
        q = self._vector(consulta)
        resultados = []
        for tipo, texto, creado, vector in filas:
            if q is not None and vector:
                a, b = np.array(q), np.array(json.loads(vector))
                score = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
                metodo = "coseno"
            else:
                palabras = set(re.findall(r"\w+", consulta.lower()))
                score = len(palabras & set(re.findall(r"\w+", texto.lower()))) / max(len(palabras), 1)
                metodo = "palabras"
            resultados.append({"tipo": tipo, "texto": texto, "creado": creado,
                               "similitud": round(score, 3), "metodo": metodo})
        return sorted(resultados, key=lambda r: r["similitud"], reverse=True)[:k]

    def todas(self, paciente_id: str) -> list[dict]:
        with self._conectar() as con:
            filas = con.execute("SELECT tipo, texto, creado FROM memorias WHERE paciente_id = ? ORDER BY id",
                                (paciente_id,)).fetchall()
        return [{"tipo": t, "texto": x, "creado": c} for t, x, c in filas]

    def borrar_todo(self):
        with self._conectar() as con:
            con.execute("DELETE FROM memorias")


_memoria: MemoriaLargoPlazo | None = None


def obtener_memoria() -> MemoriaLargoPlazo:
    global _memoria
    if _memoria is None:
        from src.config import crear_embeddings
        try:
            embeddings = crear_embeddings()
        except Exception:
            embeddings = None
        _memoria = MemoriaLargoPlazo(embeddings=embeddings)
    return _memoria


def usar_memoria(memoria: MemoriaLargoPlazo):
    """Permite inyectar otra memoria (tests)."""
    global _memoria
    _memoria = memoria
