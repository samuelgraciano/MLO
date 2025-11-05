# Dockerfile for Divorce Prediction API
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY 01-divorce-eda/scripts/ ./01-divorce-eda/scripts/
COPY models/ ./models/
COPY flows/ ./flows/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    pandas>=2.3.3 \
    numpy>=1.23.0 \
    scikit-learn>=1.7.2 \
    xgboost>=3.1.1 \
    mlflow>=3.2.0 \
    fastapi>=0.115.0 \
    uvicorn>=0.32.0 \
    pydantic>=2.10.0 \
    python-multipart>=0.0.12 \
    joblib>=1.4.0 \
    prefect>=3.1.0

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
