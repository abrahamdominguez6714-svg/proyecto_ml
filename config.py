"""Configuración del proyecto: rutas, variables y semillas."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_DATOS = RAIZ / "data" / "german-credit.csv"

VARIABLE_OBJETIVO = "credit_risk"

COLUMNAS_NUMERICAS = [
    "duration", "amount", "installment_rate",
    "present_residence", "age", "number_credits", "people_liable",
]
COLUMNAS_CATEGORICAS = [
    "status", "credit_history", "purpose", "savings", "employment_duration",
    "personal_status_sex", "other_debtors", "property", "other_installment_plans",
    "housing", "job", "telephone", "foreign_worker",
]

SEMILLA = 42
TAMANO_TEST = 0.20       # 20% del total para prueba
TAMANO_VALID = 0.25      # 25% del desarrollo (= 20% del total) para validación

METRICA_SELECCION = "F1"  # métrica para elegir el mejor modelo en validación
