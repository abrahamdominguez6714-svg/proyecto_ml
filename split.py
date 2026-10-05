"""División en entrenamiento, validación y prueba."""
from sklearn.model_selection import train_test_split

from src.config import SEMILLA, TAMANO_TEST, TAMANO_VALID


def dividir_datos(X, y):
    """Regresa (X_train, X_valid, X_test, y_train, y_valid, y_test)."""
    X_desarrollo, X_test, y_desarrollo, y_test = train_test_split(
        X, y, test_size=TAMANO_TEST, random_state=SEMILLA
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_desarrollo, y_desarrollo, test_size=TAMANO_VALID, random_state=SEMILLA
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test
