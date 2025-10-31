
# Análisis Exploratorio de Datos: Predictores de Divorcio

Este proyecto proporciona un análisis exploratorio de datos (EDA) completo del conjunto de datos de Predictores de Divorcio de UCI, que contiene respuestas a un cuestionario de escala de divorcio basado en la escala de medición de Gottman.

## 📊 Sobre el Dataset

**Fuente:** [UCI Machine Learning Repository - Divorce Predictors](https://archive.ics.uci.edu/ml/datasets/Divorce+Predictors+data+set)

**Descripción:**
- **Instancias:** 170 participantes (86 divorciados, 84 casados)
- **Características:** 54 atributos basados en la Escala de Medición de Divorcio de Gottman
- **Tipo de problema:** Clasificación binaria (Divorciado vs. Casado)
- **Formato:** Datos numéricos en escala Likert (0-4)

**Contexto:**
El conjunto de datos fue recopilado mediante un cuestionario de 54 preguntas que evalúan diferentes aspectos de las relaciones matrimoniales, basado en los principios de la terapia de pareja de Gottman. Cada pregunta se califica en una escala de 0 (Nunca) a 4 (Siempre).

## 🎯 Objetivos del Proyecto

1. **Exploración de datos:** Comprender la distribución y características del dataset
2. **Análisis de correlaciones:** Identificar las variables más predictivas del divorcio
3. **Visualización:** Crear gráficos informativos para comunicar hallazgos
4. **Preparación de datos:** Limpiar y preparar datos para modelado futuro
5. **Insights:** Extraer patrones y relaciones significativas

## 📁 Estructura del Proyecto

```
01-divorce-eda/
├── data/
│   ├── raw/                    # Datos originales (no versionados)
│   └── processed/              # Datos procesados (no versionados)
├── notebooks/
│   ├── 00_data_loading.ipynb          # Carga y primera inspección
│   ├── 01_univariate_analysis.ipynb   # Análisis univariado
│   ├── 02_bivariate_analysis.ipynb    # Análisis bivariado
│   └── 03_multivariate_analysis.ipynb # Análisis multivariado
├── scripts/
│   ├── load_data.py            # Script para cargar datos desde UCI
│   ├── preprocessing.py        # Funciones de preprocesamiento
│   └── visualization.py        # Funciones de visualización reutilizables
└── README.md
```

## 🚀 Comenzando

### 1. Instalación de Dependencias

Este proyecto usa `uv` para la gestión de paquetes. Las dependencias necesarias ya están incluidas en el `pyproject.toml` del repositorio principal:

```bash
# Desde el directorio raíz del repositorio
uv sync
```

Las dependencias incluyen:
- `pandas>=2.3.3` - Manipulación de datos
- `numpy` - Operaciones numéricas
- `matplotlib>=3.6.0` - Visualización básica
- `seaborn>=0.12.0` - Visualización estadística
- `scipy>=1.9.0` - Análisis estadístico
- `scikit-learn>=1.7.2` - Preprocesamiento y análisis
- `jupyter>=1.1.1` - Notebooks interactivos
- `openpyxl>=3.0.0` - Lectura de archivos Excel

### 2. Descarga de Datos

Ejecuta el script de carga de datos para descargar el dataset desde UCI:

```bash
uv run python 01-divorce-eda/scripts/load_data.py
```

Esto descargará los datos al directorio `01-divorce-eda/data/raw/`.

### 3. Exploración con Notebooks

Los notebooks están diseñados para ejecutarse en secuencia:

1. **00_data_loading.ipynb:** Carga inicial y comprensión del dataset
2. **01_univariate_analysis.ipynb:** Análisis de cada variable individualmente
3. **02_bivariate_analysis.ipynb:** Relaciones entre pares de variables
4. **03_multivariate_analysis.ipynb:** Análisis de múltiples variables simultáneamente

```bash
# Iniciar Jupyter desde el directorio del proyecto
jupyter notebook 01-divorce-eda/notebooks/
```

## 📈 Análisis Principales

### Variables Clave
El dataset incluye preguntas sobre:
- Comunicación en la pareja
- Resolución de conflictos
- Expresión emocional
- Apoyo mutuo
- Intimidad y conexión
- Manejo del estrés

### Técnicas de Análisis
- **Estadística descriptiva:** Media, mediana, desviación estándar
- **Visualizaciones:** Histogramas, boxplots, heatmaps de correlación
- **Análisis de correlación:** Identificar predictores fuertes
- **Pruebas estadísticas:** Tests de hipótesis para diferencias entre grupos
- **Reducción de dimensionalidad:** PCA para visualización

## 📚 Referencias

- Yöntem, M. K., Adem, K., İlhan, T., & Kılıçarslan, S. (2019). Divorce Prediction Using Correlation Based Feature Selection and Artificial Neural Networks. Nevşehir Hacı Bektaş Veli University SBE Dergisi, 9(1), 259-273.
- Gottman, J. M., & Silver, N. (1999). The Seven Principles for Making Marriage Work. Harmony Books.

---

**Nota:** Los datos están en el `.gitignore` y no se versionan. Debes ejecutar el script de descarga para obtenerlos localmente.
