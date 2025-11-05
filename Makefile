.PHONY: help install train api docker-up docker-down mlflow clean test

help:
	@echo "🚀 MLOps Pipeline - Comandos Disponibles"
	@echo ""
	@echo "  make install      - Instalar dependencias"
	@echo "  make train        - Entrenar modelos"
	@echo "  make train-flow   - Entrenar con Prefect"
	@echo "  make api          - Iniciar API FastAPI"
	@echo "  make mlflow       - Iniciar MLflow UI"
	@echo "  make docker-up    - Iniciar servicios Docker"
	@echo "  make docker-down  - Detener servicios Docker"
	@echo "  make clean        - Limpiar archivos generados"
	@echo "  make test         - Ejecutar pruebas"
	@echo ""

install:
	@echo "📦 Instalando dependencias..."
	uv sync
	@echo "✅ Dependencias instaladas"

train:
	@echo "🤖 Entrenando modelos..."
	uv run python src/train_model.py

train-flow:
	@echo "🔄 Ejecutando pipeline con Prefect..."
	uv run python flows/ml_pipeline.py

api:
	@echo "🚀 Iniciando API FastAPI..."
	uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow:
	@echo "📊 Iniciando MLflow UI..."
	mlflow ui --port 5000

docker-up:
	@echo "🐳 Iniciando servicios Docker..."
	docker-compose up -d
	@echo "✅ Servicios iniciados"
	@echo "   API: http://localhost:8000"
	@echo "   MLflow: http://localhost:5000"

docker-down:
	@echo "🛑 Deteniendo servicios Docker..."
	docker-compose down

docker-logs:
	@echo "📋 Mostrando logs..."
	docker-compose logs -f

clean:
	@echo "🧹 Limpiando archivos generados..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Limpieza completada"

test:
	@echo "🧪 Ejecutando pruebas..."
	@echo "⚠️  Tests no implementados aún"

batch-predict:
	@echo "🔮 Ejecutando predicción batch..."
	@echo "Uso: make batch-predict INPUT=data.csv OUTPUT=predictions.csv"
	uv run python src/batch_predict.py \
		--input $(INPUT) \
		--output $(OUTPUT) \
		--model $(shell ls -t models/best_model_*.pkl | head -1) \
		--scaler $(shell ls -t models/scaler_*.pkl | head -1)

setup:
	@echo "🔧 Configurando proyecto..."
	mkdir -p models logs data/raw data/processed
	@echo "✅ Directorios creados"
