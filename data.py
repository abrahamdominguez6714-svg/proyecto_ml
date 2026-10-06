"""Carga de datos, validación y separación de predictores y objetivo."""

import warnings

import pandas as pd

from src.config import (
    COLUMNAS_CATEGORICAS,
    COLUMNAS_NUMERICAS,
    RUTA_DATOS,
    VARIABLE_OBJETIVO,
)

CLASES_VALIDAS = {0, 1}


def validar_datos(datos):
    """Revisa que el DataFrame cumpla lo que el resto del proyecto asume.

    Lanza ValueError si hay un problema que haría fallar (o falsear) el
    entrenamiento. Solo avisa (warning) si hay columnas de más.
    """
    if datos.empty:
        raise ValueError("El archivo de datos está vacío.")

    # 1) Columnas esperadas
    esperadas = COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS + [VARIABLE_OBJETIVO]
    faltantes = [c for c in esperadas if c not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el CSV: {faltantes}")

    extras = [c for c in datos.columns if c not in esperadas]
    if extras:
        warnings.warn(
            f"Columnas no declaradas en config.py (se ignorarán en el "
            f"preprocesamiento): {extras}"
        )

    # 2) Valores nulos en las columnas que se usan
    nulos = datos[esperadas].isna().sum()
    nulos = nulos[nulos > 0]
    if not nulos.empty:
        raise ValueError(
            f"Hay valores faltantes y el pipeline no los imputa:\n{nulos}"
        )

    # 3) Las columnas numéricas deben ser realmente numéricas
    no_numericas = [
        c for c in COLUMNAS_NUMERICAS
        if not pd.api.types.is_numeric_dtype(datos[c])
    ]
    if no_numericas:
        raise ValueError(
            f"Columnas declaradas como numéricas pero con otro tipo: {no_numericas}"
        )

    # 4) Variable objetivo: solo 0 y 1, y con ambas clases presentes
    clases = set(datos[VARIABLE_OBJETIVO].unique())
    if not clases <= CLASES_VALIDAS:
        raise ValueError(
            f"'{VARIABLE_OBJETIVO}' debe contener solo {sorted(CLASES_VALIDAS)}; "
            f"se encontró: {sorted(clases)}"
        )
    if len(clases) < 2:
        raise ValueError(
            f"'{VARIABLE_OBJETIVO}' solo tiene una clase ({clases}); "
            "no se puede entrenar un clasificador."
        )


def cargar_datos(ruta=RUTA_DATOS):
    """Lee el CSV, lo valida y regresa el DataFrame completo."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Coloca german-credit.csv en la carpeta data/."
        )
    datos = pd.read_csv(ruta)
    validar_datos(datos)
    return datos


def separar_X_y(datos):
    """Separa los predictores (X) de la variable objetivo (y)."""
    X = datos.drop(columns=VARIABLE_OBJETIVO)
    y = datos[VARIABLE_OBJETIVO]
    return X, y
