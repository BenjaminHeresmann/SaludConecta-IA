"""Registro de decisiones del agente: se muestra en consola (para la demo) y se guarda
en data/runtime/decisiones.jsonl como evidencia de pruebas."""
import json
from datetime import datetime

from colorama import Fore, Style, just_fix_windows_console

from src.config import RUNTIME

just_fix_windows_console()
LOG = RUNTIME / "decisiones.jsonl"
COLORES = {
    "tool": Fore.CYAN, "resultado": Style.DIM, "memoria": Fore.MAGENTA, "guardrail": Fore.RED,
    "plan": Fore.YELLOW, "replan": Fore.YELLOW, "paso": Fore.GREEN, "evento": Fore.RED,
    "fallo": Fore.RED, "info": Fore.BLUE,
}


def registrar(tipo: str, texto: str, datos=None, max_chars: int = 260):
    corto = texto if len(texto) <= max_chars else texto[:max_chars] + "..."
    print(f"{COLORES.get(tipo, '')}  [{tipo.upper()}] {corto}{Style.RESET_ALL}")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"fecha": datetime.now().isoformat(timespec="seconds"), "tipo": tipo,
                            "texto": texto, "datos": datos}, ensure_ascii=False, default=str) + "\n")
