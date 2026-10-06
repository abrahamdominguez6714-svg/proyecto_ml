"""Definición de los modelos (cada uno dentro de su propio Pipeline)."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import SEMILLA
from src.preprocessing import crear_preprocesador


def crear_modelos():
    """Regresa un diccionario {nombre: pipeline}."""
    return {
        "Árbol de decisión": Pipeline([
            ("preprocessor", crear_preprocesador()),
            ("model", DecisionTreeClassifier(max_depth=6, random_state=SEMILLA)),
        ]),
        "Regresión logística": Pipeline([
            ("preprocessor", crear_preprocesador()),
            ("model", LogisticRegression(max_iter=1000, random_state=SEMILLA)),
        ]),
    }
