import pytest
from src.agent import SYSTEM_PROMPT

def test_system_prompt_rules():
    """Valida que el System Prompt contenga las restricciones críticas de negocio."""
    assert "ConectaBot" in SYSTEM_PROMPT
    assert "NUNCA entregues diagnósticos médicos" in SYSTEM_PROMPT
    assert "máximo 3 oraciones" in SYSTEM_PROMPT

def test_prompt_medical_disclaimer():
    """Valida la presencia del mensaje de derivación médica."""
    assert "Para consultas clínicas debe ser evaluado por un médico" in SYSTEM_PROMPT


