"""Demo de la presentación: 6 escenarios que muestran decisiones adaptativas del agente.

  python -m src.demo                # todos los escenarios
  python -m src.demo --escenario 3  # uno solo
  python -m src.demo --pausa        # espera Enter entre escenarios (presentación en vivo)
  python -m src.demo --sin-llm      # solo el escenario 6 con el planificador por reglas (sin API)
"""
import argparse
import sys
import time

from colorama import Fore, Style

from src.agenda import obtener_agenda
from src.memoria import obtener_memoria

LATENCIAS: list[float] = []


def titulo(n: int, texto: str, condicion: str):
    print(f"\n{Fore.WHITE}{Style.BRIGHT}{'=' * 78}\nESCENARIO {n}. {texto}\n"
          f"{Style.NORMAL}Condición del entorno: {condicion}\n{'=' * 78}{Style.RESET_ALL}")


def turno(bot, paciente: str, mensaje: str, sesion: str = "1"):
    print(f"\n{Fore.GREEN}{Style.BRIGHT}{paciente}:{Style.RESET_ALL} {mensaje}")
    inicio = time.perf_counter()
    respuesta = bot.chatear(paciente, mensaje, sesion)
    segundos = time.perf_counter() - inicio
    LATENCIAS.append(segundos)
    print(f"{Fore.CYAN}{Style.BRIGHT}ConectaBot:{Style.RESET_ALL} {respuesta}")
    print(f"{Style.DIM}  (latencia: {segundos:.1f} s){Style.RESET_ALL}")


def escenario_1(bot):
    titulo(1, "Consulta de preparación (RAG semántico)", "pregunta sin las palabras exactas del protocolo")
    turno(bot, "P001", "Hola, para mi examen de sangre, ¿puedo tomar desayuno o un café antes?")


def escenario_2(bot):
    titulo(2, "Cancelación y recuperación del cupo", "paciente cancela; hay lista de espera de la especialidad")
    turno(bot, "P002", "No voy a poder ir a mi hora de dermatología, se me complicó el trabajo. Cancélala por favor.")


def escenario_3(bot):
    titulo(3, "Reagendamiento con cupo tomado por otro canal", "el primer cupo de la tarde lo toma el call center")
    obtener_agenda().cupos_en_disputa.add("K002")
    turno(bot, "P001", "Necesito cambiar mi examen de sangre para la tarde, yo siempre prefiero las tardes.")


def escenario_4(bot):
    titulo(4, "Emergencia (guardrail)", "síntomas graves: no corresponde gestionar agenda")
    turno(bot, "P003", "Tengo un dolor fuerte en el pecho y me cuesta respirar, ¿qué hago con mi hora de cardiología?")


def escenario_5(bot):
    from src.agent import ConectaBot
    titulo(5, "Memoria entre sesiones", "se cierra la sesión y se reinicia el agente (sin historial de chat)")
    bot.cerrar_sesion("P001")
    nuevo = ConectaBot()  # nuevo checkpointer: el historial de corto plazo se perdió
    print("  (Agente reiniciado: solo queda la memoria de largo plazo en SQLite)")
    turno(nuevo, "P001", "Hola de nuevo. ¿Me buscas una hora de medicina general? Ya sabes qué horario me acomoda.",
          sesion="2")


def escenario_6(llm=None):
    from src.planner import ejecutar, evento_cancelacion
    titulo(6, "Modo coordinador: plan por prioridades y replanificación",
           "mientras corre el plan, P007 cancela su cardiología por teléfono")
    obtener_agenda().reiniciar()
    inicio = time.perf_counter()
    ejecutar(eventos={2: evento_cancelacion("C007", "aviso telefónico: viaje de trabajo")}, llm=llm)
    print(f"{Style.DIM}  (duración del modo coordinador: {time.perf_counter() - inicio:.1f} s){Style.RESET_ALL}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--escenario", type=int, choices=range(1, 7))
    parser.add_argument("--pausa", action="store_true")
    parser.add_argument("--sin-llm", action="store_true")
    args = parser.parse_args()

    if args.sin_llm:
        escenario_6(llm=None)
        return

    from src.agent import ConectaBot
    from src.config import crear_llm
    if args.escenario in (None, 1):
        obtener_agenda().reiniciar()
        obtener_memoria().borrar_todo()
    bot = ConectaBot()
    escenarios = {1: escenario_1, 2: escenario_2, 3: escenario_3, 4: escenario_4, 5: escenario_5,
                  6: lambda _bot: escenario_6(llm=crear_llm(temperatura=0))}
    for n in ([args.escenario] if args.escenario else range(1, 7)):
        escenarios[n](bot)
        if args.pausa and n < 6:
            input(f"\n{Style.DIM}[Enter para el siguiente escenario]{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
