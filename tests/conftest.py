import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import agenda as modulo_agenda  # noqa: E402
from src import memoria as modulo_memoria  # noqa: E402

HOY = date(2026, 10, 23)


@pytest.fixture(autouse=True)
def sin_apis(monkeypatch):
    """Los tests no llaman a Gemini: el RAG falla a propósito y el agente usa su texto de respaldo."""
    import src.rag_engine

    def sin_red(*args, **kwargs):
        raise RuntimeError("RAG deshabilitado en tests")
    monkeypatch.setattr(src.rag_engine, "buscar", sin_red)


@pytest.fixture
def agenda(tmp_path):
    """Agenda simulada aislada en una carpeta temporal, con fecha fija."""
    a = modulo_agenda.Agenda(tmp_path / "agenda.json", tmp_path / "outbox.jsonl")
    a.reiniciar(hoy=HOY)
    modulo_agenda.usar_agenda(a)
    return a


@pytest.fixture
def memoria(tmp_path):
    """Memoria de largo plazo sin embeddings (usa búsqueda por palabras, no llama APIs)."""
    m = modulo_memoria.MemoriaLargoPlazo(tmp_path / "memoria.db", embeddings=None)
    modulo_memoria.usar_memoria(m)
    return m
