"""Archivo principal: coordina todo el flujo de ML.

Uso:
    python train.py
"""
import pandas as pd

from src.config import METRICA_SELECCION
from src.data import cargar_datos, separar_X_y
from src.evaluation import comparar_modelos, evaluar
from src.models import crear_modelos
from src.split import dividir_datos


def main():
    datos = cargar_datos()
    X, y = separar_X_y(datos)
    X_train, X_valid, X_test, y_train, y_valid, y_test = dividir_datos(X, y)
    print(f"Train: {len(X_train)} | Validación: {len(X_valid)} | Prueba: {len(X_test)}\n")

    modelos = crear_modelos()

    # 1) Comparar modelos en validación
    resultados_valid = comparar_modelos(modelos, X_train, y_train, X_valid, y_valid)
    print("=== Resultados en validación ===")
    print(resultados_valid, "\n")

    # 2) Elegir el mejor según validación
    mejor = resultados_valid[METRICA_SELECCION].idxmax()
    print(f"Modelo seleccionado ({METRICA_SELECCION} en validación): {mejor}\n")

    # 3) Evaluar solo el mejor en prueba
    resultados_test = pd.DataFrame(
        {mejor: evaluar(modelos[mejor], X_test, y_test)}
    ).T.round(3)
    print("=== Resultados en prueba ===")
    print(resultados_test)


if __name__ == "__main__":
    main()
