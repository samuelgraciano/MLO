# 🚀 MLOps Pipeline - Guía Rápida

## 📦 Inicio Rápido

### 1. Instalación

```bash
# Instalar dependencias
uv sync

# Crear directorios necesarios
mkdir -p models logs
```

### 2. Entrenar Modelo

```bash
# Opción A: Script directo
uv run python src/train_model.py

# Opción B: Con Prefect (recomendado)
uv run python flows/ml_pipeline.py
```

### 3. Desplegar API

```bash
# Iniciar servidor FastAPI
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Acceder a documentación: http://localhost:8000/docs
```

### 4. Usar Docker

```bash
# Iniciar servicios (API + MLflow)
docker-compose up -d

# Verificar
curl http://localhost:8000/health
curl http://localhost:5000
```

## 🎯 Ejemplos de Uso

### Predicción Individual (API)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @example_request.json
```

### Predicción Batch

```bash
uv run python src/batch_predict.py \
  --input data.csv \
  --output predictions.csv \
  --model models/best_model_*.pkl \
  --scaler models/scaler_*.pkl
```

## 📊 Visualizar Experimentos

```bash
# Iniciar MLflow UI
mlflow ui --port 5000

# Abrir: http://localhost:5000
```

## 📁 Estructura

```
MLO/
├── src/
│   ├── train_model.py       # Entrenamiento
│   ├── batch_predict.py     # Predicción batch
│   └── api/main.py          # API REST
├── flows/
│   └── ml_pipeline.py       # Orquestación Prefect
├── 01-divorce-eda/scripts/  # Scripts reutilizables
├── models/                  # Modelos entrenados
├── Dockerfile               # Imagen Docker
└── docker-compose.yml       # Servicios
```

## 📖 Documentación Completa

Ver [MLOPS_DOCUMENTATION.md](MLOPS_DOCUMENTATION.md) para:
- Planteamiento del problema
- Metodología detallada
- Arquitectura del sistema
- Resultados y conclusiones

## 🔗 Endpoints API

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Info de la API |
| `/health` | GET | Health check |
| `/model-info` | GET | Info del modelo |
| `/predict` | POST | Predicción individual |
| `/batch-predict` | POST | Predicciones múltiples |
| `/predict-file` | POST | Predicción desde CSV |

## 🛠️ Tecnologías

- **Orquestación**: Prefect 3.1+
- **Tracking**: MLflow 3.2+
- **ML**: Scikit-learn, XGBoost
- **API**: FastAPI
- **Container**: Docker

## ⚡ Comandos Útiles

```bash
# Ver logs de Docker
docker-compose logs -f api

# Detener servicios
docker-compose down

# Limpiar todo
docker-compose down -v

# Ver modelos entrenados
ls -lh models/

# Ver experimentos MLflow
ls -lh mlruns/
```

## 📝 Notas Importantes

- El modelo se entrena con 150 instancias (después de limpieza)
- Mejor modelo: XGBoost (F1=0.9825, ROC-AUC=0.9950)
- Dataset balanceado: 50% divorciados, 50% casados
- Features: 54 atributos en escala Likert (0-4)