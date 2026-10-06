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

Asimismo, data.py valida al cargar que existan las columnas esperadas, que no haya nulos y que credit_risk contenga solo 0 y 1.

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


## 9. Reflexión final

### Abraham Dominguez Valdez
* **¿Qué implementaste y qué decisión técnica tomaste para realizarla?**  
  Implementé los módulos `config.py`, `data.py` y `split.py`. Centralicé todas las constantes, rutas dinámicas mediante `pathlib.Path` y listas de columnas en `config.py`. En `split.py`, tomé la decisión técnica de estructurar un split en dos etapas (60% entrenamiento, 20% validación y 20% prueba) usando `train_test_split`.
* **¿Cómo verificaste tu aportación?**  
  Imprimí las dimensiones (`shape` y `len()`) de cada subconjunto (`X_train`, `X_valid`, `X_test`, etc.), comprobando que sumaran los 1,000 registros del dataset original y verificando la lectura correcta del CSV.
* **¿Qué observaste o aprendiste al revisar el trabajo de otra persona?** *(Revisión del PR de Angel)*  
  Observé cómo se integran todos los módulos dentro del orquestador `train.py` y aprendí la importancia de automatizar la selección del mejor modelo con base en la métrica `F1` en validación antes de evaluar aisladamente en prueba.
* **¿Qué mejorarías en la siguiente versión?**  
  Agregaría el parámetro `stratify=y` en la división de datos para preservar exactamente la proporción original del desbalance de clases (70% / 30%) en las tres particiones.

---

### Cristopher Altamirano Hernandez
* **¿Qué implementaste y qué decisión técnica tomaste para realizarla?**  
  Implementé los módulos `preprocessing.py` y `models.py`. Tomé la decisión de encapsular el `ColumnTransformer` (`StandardScaler` para numéricas y `OneHotEncoder` para categóricas) dentro del `Pipeline` individual de cada estimador. Además, configuré `handle_unknown="ignore"` para prevenir errores ante categorías no vistas en prueba.
* **¿Cómo verificaste tu aportación?**  
  Instancié la función `crear_modelos()` y ejecuté un ajuste directo (`fit`) con un conjunto de entrenamiento de prueba (`X_train`, `y_train`), comprobando que la transformación y el entrenamiento corrieran sin errores de dimensión.
* **¿Qué observaste o aprendiste al revisar el trabajo de otra persona?** *(Revisión del PR de Abraham)*  
  Aprendí la utilidad de aislar la configuración global en `config.py`, lo que me permitió importar directamente las listas de columnas numéricas y categóricas sin redefinir variables en `preprocessing.py`.
* **¿Qué mejorarías en la siguiente versión?**  
  Incorporaría técnicas para el manejo de desbalance de clases (como `class_weight="balanced"` o SMOTE) dentro de los pipelines de los modelos.

---

### Angel Said Vazquez Cisneros
* **¿Qué implementaste y qué decisión técnica tomaste para realizarla?**  
  Implementé los módulos `evaluation.py` y `train.py`. Diseñé las funciones de evaluación para devolver un `DataFrame` comparativo formateado con las métricas clave y configuré la selección automática del mejor modelo basándose en el `F1` de validación antes de evaluar una sola vez en prueba.
* **¿Cómo verificaste tu aportación?**  
  Ejecuté el flujo completo con `uv run python train.py` en la consola, comprobando que se cargaran los datos, se dividieran, se entrenaran los pipelines, se mostrara la tabla en validación y se imprimieran las métricas finales en prueba.
* **¿Qué observaste o aprendiste al revisar el trabajo de otra persona?** *(Revisión del PR de Cristopher)*  
  Aprendí la importancia de integrar el preprocesador directamente en la estructura de `Pipeline` de cada modelo, garantizando que las transformaciones se ajusten únicamente con los datos de entrenamiento para evitar la fuga de datos (*data leakage*).
* **¿Qué mejorarías en la siguiente versión?**  
  Añadiría la métrica `ROC-AUC` dentro del reporte de evaluación y la exportación automática del modelo ganador a un archivo `.joblib` para su consumo posterior.

---

### Adrián Carrera Ahumada
* **¿Qué implementaste y qué decisión técnica tomaste para realizarla?**  
  Implementé la estructuración, redacción y consolidación final de la documentación del proyecto en `README.md`, así como la supervisión de los cambios introducidos por el equipo. Tomé la decisión técnica de auditar cada módulo (`src/`), estandarizar las instrucciones de ejecución con `uv`, documentar los resultados cuantitativos de validación/prueba y coordinar la sección de reflexión y la tabla de contribuciones para garantizar la coherencia global del repositorio.
* **¿Cómo verificaste tu aportación?**  
  Revisé que el `README.md` se renderizara correctamente en GitHub sin errores de sintaxis Markdown, verifiqué que todos los comandos (`uv sync`, `uv run python train.py`) funcionaran en un entorno desde cero y audité que los módulos y tablas coincidieran exactamente con el código fuente.
* **¿Qué observaste o aprendiste al revisar el trabajo de otra persona?** *(Supervisión general de PRs)*  
  Al supervisar la integración de código de mis compañeros (Abraham, Cristopher y Angel Said), aprendí el impacto positivo que tiene la arquitectura modular en Python para mantener un repositorio limpio y entendí cómo la revisión sustantiva de Pull Requests evita inconsistencias en las variables y formatos antes de fusionar con la rama principal.
* **¿Qué mejorarías en la siguiente versión?**  
  Integraría una plantilla estandarizada de Pull Request (`.github/PULL_REQUEST_TEMPLATE.md`) y automatizaría la verificación de formato de código y documentación mediante GitHub Actions (CI/CD) en cada contribución.
