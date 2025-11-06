"""
FastAPI REST API for Divorce Prediction Model
Provides endpoints for single and batch predictions with validation.
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Divorce Prediction API",
    description="API for predicting divorce probability based on relationship questionnaire responses",
    version="1.0.0"
)

# Global variables for model and scaler
model = None
scaler = None
feature_names = None
model_metadata = None


class QuestionnaireResponse(BaseModel):
    """Single questionnaire response with 54 attributes."""
    
    Atr1: Optional[int] = Field(None, ge=0, le=4, description="Response to question 1 (0-4 Likert scale)")
    Atr2: Optional[int] = Field(None, ge=0, le=4, description="Response to question 2 (0-4 Likert scale)")
    Atr3: Optional[int] = Field(None, ge=0, le=4, description="Response to question 3 (0-4 Likert scale)")
    Atr4: Optional[int] = Field(None, ge=0, le=4, description="Response to question 4 (0-4 Likert scale)")
    Atr5: Optional[int] = Field(None, ge=0, le=4, description="Response to question 5 (0-4 Likert scale)")
    Atr6: Optional[int] = Field(None, ge=0, le=4, description="Response to question 6 (0-4 Likert scale)")
    Atr7: Optional[int] = Field(None, ge=0, le=4, description="Response to question 7 (0-4 Likert scale)")
    Atr8: Optional[int] = Field(None, ge=0, le=4, description="Response to question 8 (0-4 Likert scale)")
    Atr9: Optional[int] = Field(None, ge=0, le=4, description="Response to question 9 (0-4 Likert scale)")
    Atr10: Optional[int] = Field(None, ge=0, le=4, description="Response to question 10 (0-4 Likert scale)")
    Atr11: Optional[int] = Field(None, ge=0, le=4, description="Response to question 11 (0-4 Likert scale)")
    Atr12: Optional[int] = Field(None, ge=0, le=4, description="Response to question 12 (0-4 Likert scale)")
    Atr13: Optional[int] = Field(None, ge=0, le=4, description="Response to question 13 (0-4 Likert scale)")
    Atr14: Optional[int] = Field(None, ge=0, le=4, description="Response to question 14 (0-4 Likert scale)")
    Atr15: Optional[int] = Field(None, ge=0, le=4, description="Response to question 15 (0-4 Likert scale)")
    Atr16: Optional[int] = Field(None, ge=0, le=4, description="Response to question 16 (0-4 Likert scale)")
    Atr17: Optional[int] = Field(None, ge=0, le=4, description="Response to question 17 (0-4 Likert scale)")
    Atr18: Optional[int] = Field(None, ge=0, le=4, description="Response to question 18 (0-4 Likert scale)")
    Atr19: Optional[int] = Field(None, ge=0, le=4, description="Response to question 19 (0-4 Likert scale)")
    Atr20: Optional[int] = Field(None, ge=0, le=4, description="Response to question 20 (0-4 Likert scale)")
    Atr21: Optional[int] = Field(None, ge=0, le=4, description="Response to question 21 (0-4 Likert scale)")
    Atr22: Optional[int] = Field(None, ge=0, le=4, description="Response to question 22 (0-4 Likert scale)")
    Atr23: Optional[int] = Field(None, ge=0, le=4, description="Response to question 23 (0-4 Likert scale)")
    Atr24: Optional[int] = Field(None, ge=0, le=4, description="Response to question 24 (0-4 Likert scale)")
    Atr25: Optional[int] = Field(None, ge=0, le=4, description="Response to question 25 (0-4 Likert scale)")
    Atr26: Optional[int] = Field(None, ge=0, le=4, description="Response to question 26 (0-4 Likert scale)")
    Atr27: Optional[int] = Field(None, ge=0, le=4, description="Response to question 27 (0-4 Likert scale)")
    Atr28: Optional[int] = Field(None, ge=0, le=4, description="Response to question 28 (0-4 Likert scale)")
    Atr29: Optional[int] = Field(None, ge=0, le=4, description="Response to question 29 (0-4 Likert scale)")
    Atr30: Optional[int] = Field(None, ge=0, le=4, description="Response to question 30 (0-4 Likert scale)")
    Atr31: Optional[int] = Field(None, ge=0, le=4, description="Response to question 31 (0-4 Likert scale)")
    Atr32: Optional[int] = Field(None, ge=0, le=4, description="Response to question 32 (0-4 Likert scale)")
    Atr33: Optional[int] = Field(None, ge=0, le=4, description="Response to question 33 (0-4 Likert scale)")
    Atr34: Optional[int] = Field(None, ge=0, le=4, description="Response to question 34 (0-4 Likert scale)")
    Atr35: Optional[int] = Field(None, ge=0, le=4, description="Response to question 35 (0-4 Likert scale)")
    Atr36: Optional[int] = Field(None, ge=0, le=4, description="Response to question 36 (0-4 Likert scale)")
    Atr37: Optional[int] = Field(None, ge=0, le=4, description="Response to question 37 (0-4 Likert scale)")
    Atr38: Optional[int] = Field(None, ge=0, le=4, description="Response to question 38 (0-4 Likert scale)")
    Atr39: Optional[int] = Field(None, ge=0, le=4, description="Response to question 39 (0-4 Likert scale)")
    Atr40: Optional[int] = Field(None, ge=0, le=4, description="Response to question 40 (0-4 Likert scale)")
    Atr41: Optional[int] = Field(None, ge=0, le=4, description="Response to question 41 (0-4 Likert scale)")
    Atr42: Optional[int] = Field(None, ge=0, le=4, description="Response to question 42 (0-4 Likert scale)")
    Atr43: Optional[int] = Field(None, ge=0, le=4, description="Response to question 43 (0-4 Likert scale)")
    Atr44: Optional[int] = Field(None, ge=0, le=4, description="Response to question 44 (0-4 Likert scale)")
    Atr45: Optional[int] = Field(None, ge=0, le=4, description="Response to question 45 (0-4 Likert scale)")
    Atr46: Optional[int] = Field(None, ge=0, le=4, description="Response to question 46 (0-4 Likert scale)")
    Atr47: Optional[int] = Field(None, ge=0, le=4, description="Response to question 47 (0-4 Likert scale)")
    Atr48: Optional[int] = Field(None, ge=0, le=4, description="Response to question 48 (0-4 Likert scale)")
    Atr49: Optional[int] = Field(None, ge=0, le=4, description="Response to question 49 (0-4 Likert scale)")
    Atr50: Optional[int] = Field(None, ge=0, le=4, description="Response to question 50 (0-4 Likert scale)")
    Atr51: Optional[int] = Field(None, ge=0, le=4, description="Response to question 51 (0-4 Likert scale)")
    Atr52: Optional[int] = Field(None, ge=0, le=4, description="Response to question 52 (0-4 Likert scale)")
    Atr53: Optional[int] = Field(None, ge=0, le=4, description="Response to question 53 (0-4 Likert scale)")
    Atr54: Optional[int] = Field(None, ge=0, le=4, description="Response to question 54 (0-4 Likert scale)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "Atr1": 2, "Atr2": 2, "Atr3": 1, "Atr4": 0, "Atr5": 0,
                "Atr6": 4, "Atr7": 1, "Atr8": 3, "Atr9": 3, "Atr10": 3,
                "Atr11": 3, "Atr12": 3, "Atr13": 3, "Atr14": 3, "Atr15": 3,
                "Atr16": 3, "Atr17": 3, "Atr18": 3, "Atr19": 3, "Atr20": 3,
                "Atr21": 2, "Atr22": 2, "Atr23": 2, "Atr24": 3, "Atr25": 2,
                "Atr26": 3, "Atr27": 2, "Atr28": 3, "Atr29": 2, "Atr30": 3,
                "Atr31": 1, "Atr32": 1, "Atr33": 1, "Atr34": 1, "Atr35": 1,
                "Atr36": 1, "Atr37": 1, "Atr38": 2, "Atr39": 2, "Atr40": 2,
                "Atr41": 2, "Atr42": 2, "Atr43": 2, "Atr44": 2, "Atr45": 2,
                "Atr46": 2, "Atr47": 1, "Atr48": 1, "Atr49": 1, "Atr50": 1,
                "Atr51": 1, "Atr52": 1, "Atr53": 1, "Atr54": 1
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: int = Field(..., description="Predicted class (0=Married, 1=Divorced)")
    probability: float = Field(..., description="Probability of divorce (0-1)")
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    responses: List[QuestionnaireResponse] = Field(..., description="List of questionnaire responses")


def load_model_artifacts(model_dir: str = "models"):
    """Load model, scaler, and metadata."""
    global model, scaler, feature_names, model_metadata
    
    model_path = Path(model_dir)
    
    # Find latest model file
    model_files = list(model_path.glob("best_model_*.pkl"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading model from: {latest_model}")
    model = joblib.load(latest_model)
    
    # Load scaler
    timestamp = latest_model.stem.split('_')[-1]
    scaler_file = model_path / f"scaler_{timestamp}.pkl"
    
    if scaler_file.exists():
        logger.info(f"Loading scaler from: {scaler_file}")
        scaler = joblib.load(scaler_file)
    else:
        logger.warning("Scaler not found, predictions may be inaccurate")
        scaler = None
    
    # Load metadata
    metadata_file = model_path / f"model_metadata_{timestamp}.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            model_metadata = json.load(f)
        feature_names = model_metadata.get('feature_names', [])
        logger.info(f"Model metadata loaded: {model_metadata['model_name']}")
    else:
        logger.warning("Model metadata not found")
        feature_names = [f"Atr{i+1}" for i in range(54)]
        model_metadata = {"model_name": "unknown", "training_date": "unknown"}
    
    logger.info("Model artifacts loaded successfully")


def get_risk_level(probability: float) -> str:
    """Determine risk level based on probability."""
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"


def prepare_features(response: QuestionnaireResponse) -> pd.DataFrame:
    """Convert response to DataFrame with proper feature names."""
    data = response.model_dump()
    df = pd.DataFrame([data])
    
    # Fill missing values with neutral value (2 = middle of 0-4 Likert scale)
    df = df.fillna(2)
    
    # Ensure correct column order
    if feature_names:
        df = df[feature_names]
    
    return df


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model_artifacts()
        logger.info("API started successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Divorce Prediction API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model-info", tags=["Model"])
async def model_info():
    """Get information about the loaded model."""
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not available")
    
    return {
        "model_name": model_metadata.get('model_name', 'unknown'),
        "training_date": model_metadata.get('training_date', 'unknown'),
        "metrics": model_metadata.get('metrics', {}),
        "features_count": len(feature_names) if feature_names else 0,
        "model_type": str(type(model).__name__) if model else "Not loaded"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(response: QuestionnaireResponse):
    """
    Make a single prediction.
    
    Args:
        response: Questionnaire responses (54 attributes, each 0-4)
        
    Returns:
        Prediction with probability and risk level
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Prepare features
        X = prepare_features(response)
        
        # Scale features
        if scaler:
            X_scaled = scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        else:
            X_scaled = X
        
        # Make prediction
        prediction = int(model.predict(X_scaled)[0])
        probability = float(model.predict_proba(X_scaled)[0, 1]) if hasattr(model, 'predict_proba') else 0.5
        
        # Determine risk level
        risk_level = get_risk_level(probability)
        
        logger.info(f"Prediction made: {prediction} (probability: {probability:.4f})")
        
        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk_level,
            timestamp=datetime.now().isoformat(),
            model_version=model_metadata.get('model_name', 'unknown')
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict", tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Make batch predictions.
    
    Args:
        request: List of questionnaire responses
        
    Returns:
        List of predictions
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        predictions = []
        
        for idx, response in enumerate(request.responses):
            # Prepare features
            X = prepare_features(response)
            
            # Scale features
            if scaler:
                X_scaled = scaler.transform(X)
                X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            else:
                X_scaled = X
            
            # Make prediction
            prediction = int(model.predict(X_scaled)[0])
            probability = float(model.predict_proba(X_scaled)[0, 1]) if hasattr(model, 'predict_proba') else 0.5
            risk_level = get_risk_level(probability)
            
            predictions.append({
                "index": idx,
                "prediction": prediction,
                "probability": round(probability, 4),
                "risk_level": risk_level
            })
        
        logger.info(f"Batch prediction completed: {len(predictions)} predictions")
        
        return {
            "predictions": predictions,
            "count": len(predictions),
            "timestamp": datetime.now().isoformat(),
            "model_version": model_metadata.get('model_name', 'unknown')
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/predict-file", tags=["Prediction"])
async def predict_file(file: UploadFile = File(...)):
    """
    Make predictions from uploaded CSV file.
    
    Args:
        file: CSV file with questionnaire responses
        
    Returns:
        CSV file with predictions
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        logger.info(f"File uploaded: {file.filename}, shape: {df.shape}")
        
        # Validate columns
        expected_cols = [f"Atr{i+1}" for i in range(54)]
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing_cols}"
            )
        
        # Prepare features
        X = df[expected_cols]
        
        # Scale features
        if scaler:
            X_scaled = scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_scaled = X
        
        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1] if hasattr(model, 'predict_proba') else [0.5] * len(X)
        
        # Add predictions to DataFrame
        df['prediction'] = predictions
        df['probability'] = probabilities
        df['risk_level'] = [get_risk_level(p) for p in probabilities]
        df['timestamp'] = datetime.now().isoformat()
        
        # Save to temporary file
        output_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"File predictions completed: {len(predictions)} predictions")
        
        return FileResponse(
            output_file,
            media_type='text/csv',
            filename=output_file
        )
        
    except Exception as e:
        logger.error(f"File prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
