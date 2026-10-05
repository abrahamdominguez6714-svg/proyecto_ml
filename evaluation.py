"""Entrenamiento y evaluación compartidos por todos los modelos."""
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluar(modelo, X, y):
    """Calcula las métricas de clasificación de un modelo ya entrenado."""
    pred = modelo.predict(X)
    return {
        "Exactitud": accuracy_score(y, pred),
        "Precisión": precision_score(y, pred),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
    }


def comparar_modelos(modelos, X_train, y_train, X_valid, y_valid):
    """Entrena cada modelo y regresa una tabla de métricas en validación."""
    filas = {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        filas[nombre] = evaluar(modelo, X_valid, y_valid)
    return pd.DataFrame(filas).T.round(3)
