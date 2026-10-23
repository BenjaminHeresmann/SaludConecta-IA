"""Modo coordinador sobre la agenda ACTUAL (sin reiniciar): muestra lo que hicieron los pacientes
desde el chat y luego corre el plan por prioridades con ese estado.

Uso:  python -m src.coordinador [--sin-llm]   |   python -m src.coordinador --reiniciar
"""
import argparse
import json
import sys
from datetime import date

from colorama import Fore, Style

from src.agenda import obtener_agenda


def cupos_liberados_pendientes(agenda) -> list[str]:
    """Cupos de citas canceladas que siguen libres (nadie de la lista de espera los tomó)."""
    hoy = date.fromisoformat(agenda.datos["hoy"])
    cupos = []
    for c in agenda.datos["citas"].values():
        cupo = agenda.datos["cupos"][c["cupo_id"]]
        if (c["estado"] == "cancelada" and cupo["estado"] == "libre" and date.fromisoformat(cupo["fecha"]) > hoy
                and agenda.candidatos_lista_espera(c["cupo_id"])):  # sin candidatos no hay a quién reasignar
            cupos.append(c["cupo_id"])
    return sorted(set(cupos))


def mostrar_estado(agenda):
    print(f"\n{Style.BRIGHT}ESTADO ACTUAL DE LA AGENDA (hoy {agenda.datos['hoy']}){Style.RESET_ALL}")
    for cid, c in sorted(agenda.datos["citas"].items()):
        d = agenda.detalle_cita(cid)
        if c["estado"] == "cancelada":
            marca = f"{Fore.RED}CANCELADA ({c.get('motivo_cancelacion', '')}){Style.RESET_ALL}"
        elif c.get("origen") == "lista_espera":
            marca = f"{Fore.GREEN}NUEVA desde lista de espera{Style.RESET_ALL}"
        else:
            marca = "confirmada" if c["confirmada"] else "sin confirmar"
        print(f"  {cid}  {c['paciente_id']}  {d['fecha']} {d['hora']}  {d['especialidad']:<16} {d['sede']:<15} {marca}")
    espera = ", ".join(f"{e['paciente_id']} ({e['especialidad']}, urg. {e['urgencia']})"
                       for e in agenda.datos["lista_espera"]) or "vacía"
    print(f"  Lista de espera: {espera}")
    if agenda.ruta_outbox.exists():
        mensajes = [json.loads(l) for l in agenda.ruta_outbox.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"  Mensajes WhatsApp enviados: {len(mensajes)}")
        for m in mensajes[-3:]:
            print(f"    -> {m['paciente_id']}: {m['mensaje'][:100]}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--sin-llm", action="store_true", help="usa el planificador por reglas")
    parser.add_argument("--reiniciar", action="store_true", help="solo reinicia la agenda simulada")
    args = parser.parse_args()
    if args.reiniciar:
        obtener_agenda().reiniciar()
        print("Agenda reiniciada: 8 citas, 4 pacientes en lista de espera. La memoria de los pacientes se mantiene.")
        return
    from src.planner import ejecutar

    agenda = obtener_agenda()
    mostrar_estado(agenda)
    liberados = cupos_liberados_pendientes(agenda)
    if liberados:
        print(f"  Cupos liberados sin reasignar: {', '.join(liberados)}")
    print(f"\n{Style.BRIGHT}COORDINADOR: plan de las próximas 48 horas{Style.RESET_ALL}")
    llm = None
    if not args.sin_llm:
        from src.config import crear_cascada
        llm = crear_cascada(temperatura=0, max_tokens=2000)
    ejecutar(llm=llm, agenda=agenda, cupos_liberados=liberados)
    mostrar_estado(agenda)


if __name__ == "__main__":
    main()
