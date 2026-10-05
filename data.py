"""Carga de datos y separación de predictores y objetivo."""
import pandas as pd

from src.config import RUTA_DATOS, VARIABLE_OBJETIVO


def cargar_datos(ruta=RUTA_DATOS):
    """Lee el CSV y regresa el DataFrame completo."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Coloca german-credit.csv en la carpeta data/."
        )
    return pd.read_csv(ruta)


def separar_X_y(datos):
    """Separa los predictores (X) de la variable objetivo (y)."""
    X = datos.drop(columns=VARIABLE_OBJETIVO)
    y = datos[VARIABLE_OBJETIVO]
    return X, y
