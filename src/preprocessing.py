"""Preprocesamiento de los datos."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import COLUMNAS_CATEGORICAS, COLUMNAS_NUMERICAS


def crear_preprocesador():
    """Escala las numéricas y codifica las categóricas.

    Se integra al modelo mediante un Pipeline, así solo se ajusta
    con los datos de entrenamiento.
    """
    return ColumnTransformer([
        ("numericas", StandardScaler(), COLUMNAS_NUMERICAS),
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), COLUMNAS_CATEGORICAS),
    ])
