import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
    )
    return (
        mo, pd, clone, train_test_split, ColumnTransformer, StandardScaler,
        OneHotEncoder, Pipeline, DecisionTreeClassifier, LogisticRegression,
        accuracy_score, precision_score, recall_score, f1_score,
    )


@app.cell
def _(mo):
    mo.md("# Prueba del proyecto (German Credit)")
    return


@app.cell
def _(mo):
    archivo = mo.ui.file(filetypes=[".csv"], label="Sube aquí tu german-credit.csv")
    archivo
    return (archivo,)


@app.cell
def _(mo, pd, archivo):
    mo.stop(not archivo.value, mo.md("⏸ Sube el archivo CSV para continuar."))

    import io as io_builtin
    datos = pd.read_csv(io_builtin.BytesIO(archivo.contents()))
    X = datos.drop(columns="credit_risk")
    y = datos["credit_risk"]
    datos.head()
    return X, y


@app.cell
def _(X, y, train_test_split):
    X_desarrollo, X_test, y_desarrollo, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_desarrollo, y_desarrollo, test_size=0.25, random_state=42
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


@app.cell
def _(ColumnTransformer, StandardScaler, OneHotEncoder):
    columnas_numericas = [
        "duration", "amount", "installment_rate",
        "present_residence", "age", "number_credits", "people_liable",
    ]
    columnas_categoricas = [
        "status", "credit_history", "purpose", "savings", "employment_duration",
        "personal_status_sex", "other_debtors", "property", "other_installment_plans",
        "housing", "job", "telephone", "foreign_worker",
    ]

    preprocesador = ColumnTransformer([
        ("numericas", StandardScaler(), columnas_numericas),
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
    ])
    return (preprocesador,)


@app.cell
def _(
    Pipeline, clone, preprocesador,
    DecisionTreeClassifier, LogisticRegression,
):
    # Cada modelo tiene su propio pipeline (con una copia del preprocesador),
    # así el preprocesamiento se ajusta solo con los datos de entrenamiento.
    modelos = {
        "Árbol de decisión": Pipeline([
            ("preprocessor", clone(preprocesador)),
            ("model", DecisionTreeClassifier(max_depth=6, random_state=42)),
        ]),
        "Regresión logística": Pipeline([
            ("preprocessor", clone(preprocesador)),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
    }
    return (modelos,)


@app.cell
def _(accuracy_score, precision_score, recall_score, f1_score):
    # Función de evaluación compartida por todos los modelos
    def evaluar(modelo, X, y):
        pred = modelo.predict(X)
        return {
            "Exactitud": accuracy_score(y, pred),
            "Precisión": precision_score(y, pred),
            "Recall": recall_score(y, pred),
            "F1": f1_score(y, pred),
        }
    return (evaluar,)


@app.cell
def _(modelos, evaluar, X_train, y_train, X_valid, y_valid, pd, mo):
    # Entrenar y comparar en VALIDACIÓN
    _filas = {}
    for _nombre, _modelo in modelos.items():
        _modelo.fit(X_train, y_train)
        _filas[_nombre] = evaluar(_modelo, X_valid, y_valid)

    resultados_valid = pd.DataFrame(_filas).T.round(3)
    mejor_nombre = resultados_valid["F1"].idxmax()

    mo.vstack([
        mo.md("## Comparación en validación"),
        mo.ui.table(resultados_valid.reset_index(names="Modelo"), selection=None),
        mo.md(f"**Modelo seleccionado (mejor F1 en validación):** {mejor_nombre}"),
    ])
    return (mejor_nombre,)


@app.cell
def _(modelos, mejor_nombre, evaluar, X_test, y_test, pd, mo):
    # Evaluar SOLO el modelo seleccionado en PRUEBA
    resultados_test = pd.DataFrame(
        {mejor_nombre: evaluar(modelos[mejor_nombre], X_test, y_test)}
    ).T.round(3)

    mo.vstack([
        mo.md(f"## Evaluación final en prueba: {mejor_nombre}"),
        mo.ui.table(resultados_test.reset_index(names="Modelo"), selection=None),
    ])
    return


if __name__ == "__main__":
    app.run()
