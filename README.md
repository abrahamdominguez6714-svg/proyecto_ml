# Proyecto Modular de ML 
# Clasificación de riesgo crediticio (German Credit)

## 1. Descripción del problema y del conjunto de datos

El objetivo es predecir si un solicitante de crédito representa un buen o mal riesgo
(`credit_risk`) a partir de sus características. Es un problema de **clasificación binaria**.

El conjunto de datos *German Credit* tiene **1,000 registros** y **20 variables predictoras**
(7 numéricas como `duration`, `amount` y `age`, y 13 categóricas como `status`, `purpose`
y `housing`). La variable objetivo `credit_risk` es `1` (700 casos) o `0` (300 casos),
por lo que las clases están desbalanceadas. No hay valores faltantes.

## 2. Integrantes

| Nombre | Usuario de GitHub |
|---|---|
| Abraham Dominguez Valdez | @abrahamdominguez6714-svg |
| Cristopher Altamirano Hernandez | @crist-o |
| Angel Said Vazquez Cisneros | @angelsaid06 |
| Adrián Carrera Ahumada | @datadri-code |

## 3. Organización de los archivos

```
.
├── data/
│   └── german-credit.csv       # conjunto de datos
├── notebooks/
│   └── original_marimo.py      # notebook original de referencia (clase)
├── src/
│   ├── __init__.py
│   ├── config.py               # rutas, variables predictoras, objetivo y semillas
│   ├── data.py                 # carga de datos y separación X / y
│   ├── split.py                # división en entrenamiento, validación y prueba
│   ├── preprocessing.py        # ColumnTransformer (escalado + one-hot)
│   ├── models.py               # definición de los modelos (pipelines)
│   └── evaluation.py           # métricas, entrenamiento y comparación
├── train.py                    # archivo principal que coordina el flujo
├── pyproject.toml              # dependencias (uv)
└── README.md
```

## 4. Instalación de dependencias y datos

Se necesita [uv](https://docs.astral.sh/uv/) instalado.

```bash
git clone <URL-DEL-REPOSITORIO>
cd <carpeta-del-repositorio>
uv sync
```

`uv sync` crea el entorno virtual e instala las dependencias de `pyproject.toml`.
El archivo `german-credit.csv` ya viene en la carpeta `data/`; si no estuviera,
colócalo ahí.

## 5. Ejecución

```bash
uv run python train.py
```

El script divide los datos (60% entrenamiento, 20% validación, 20% prueba), entrena los
modelos, los compara en validación, selecciona el mejor y lo evalúa en prueba.

## 6. Modelos, resultados e interpretación

Ambos modelos usan un `Pipeline` con el mismo preprocesamiento, que se ajusta **solo** con
los datos de entrenamiento: escalado (`StandardScaler`) para las numéricas y
`OneHotEncoder` para las categóricas.

- **Árbol de decisión** (`max_depth=6`)
- **Regresión logística** (`max_iter=1000`)

Resultados en validación (semilla 42):

| Modelo | Exactitud | Precisión | Recall | F1 |
|---|---|---|---|---|
| Árbol de decisión | 0.685 | 0.745 | 0.826 | 0.784 |
| Regresión logística | 0.725 | 0.752 | 0.899 | 0.818 |

Se seleccionó la **regresión logística** por tener el mejor F1 en validación.
Resultado final en prueba:

| Modelo | Exactitud | Precisión | Recall | F1 |
|---|---|---|---|---|
| Regresión logística | 0.775 | 0.812 | 0.887 | 0.847 |

**Interpretación breve:** la regresión logística superó al árbol en todas las métricas de
validación. Un árbol de profundidad 6 tiende a sobreajustarse con solo 600 registros de
entrenamiento, mientras que el modelo lineal generaliza mejor aquí. El recall alto indica
que el modelo identifica bien a la clase `1`, la mayoritaria. El resultado en prueba es
similar al de validación, lo que sugiere que la selección no estuvo sobreajustada a los
datos de validación.

## 7. Limitaciones y problemas conocidos

- El dataset es pequeño (1,000 registros), así que las métricas varían según la semilla
  y el tamaño de cada partición.
- Las clases están desbalanceadas (70% / 30%), no se aplicó ningún balanceo y la
  división de los datos no está estratificada.
- Se usa una sola división de los datos; no se hizo validación cruzada.
- No se hizo búsqueda de hiperparámetros; los valores son fijos.

## 8. Tabla de contribuciones

| Quién | Archivos | Rama | Lo revisa | Cuándo |
|---|---|---|---|---|
| *Abraham* | config.py, data.py, split.py | feature/config-data-split | Cristo | Primero |
| *Cristo* | preprocessing.py, models.py | feature/preprocessing-models | Angel | Después del merge de Abraham |
| *Angel* | evaluation.py, train.py | feature/evaluation-train | Abraham | Después del merge de Cristo |
| *Adrián* | README.md | feature/docs-readme | Angel / Supervisión general | Después del merge de Angel |
