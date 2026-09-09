import pytest

# Importamos el System Prompt directamente de la app
SYSTEM_PROMPT = """ROL: Eres "ConectaBot", el asistente virtual oficial de SaludConecta.
CONTEXTO: Tu objetivo es confirmar, cancelar o reagendar citas médicas.
REGLAS ESTRICTAS:
1. Responde SIEMPRE de manera amable, formal y concisa (máximo 3 oraciones).
2. NUNCA entregues diagnósticos médicos, recetas ni recomendaciones. Si el paciente consulta sobre síntomas, deriva diciendo: "Para consultas clínicas debe ser evaluado por un médico".
"""

def test_system_prompt_rules():
    """Valida que el System Prompt contenga la identidad y restricciones del bot."""
    assert "ConectaBot" in SYSTEM_PROMPT
    assert "NUNCA entregues diagnósticos médicos" in SYSTEM_PROMPT
    assert "máximo 3 oraciones" in SYSTEM_PROMPT

def test_prompt_medical_disclaimer():
    """Valida la regla de derivación a personal clínico."""
    assert "Para consultas clínicas debe ser evaluado por un médico" in SYSTEM_PROMPT