"""
Prefect Orchestration Flows for ML Pipeline
Orchestrates data acquisition, processing, training, and evaluation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01-divorce-eda', 'scripts'))

from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict

# Import existing modules
from load_data import load_divorce_data
from data_cleaning import clean_divorce_data
from train_model import DivorceModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@task(name="acquire_data", retries=2, retry_delay_seconds=10)
def acquire_data_task(data_path: str = "data") -> pd.DataFrame:
    """
    Task: Acquire data from source.
    
    Args:
        data_path: Path to data directory
        
    Returns:
        Raw DataFrame
    """
    logger.info("📥 Acquiring data...")
    df = load_divorce_data(data_path=data_path)
    logger.info(f"✅ Data acquired: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


@task(name="clean_data")
def clean_data_task(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Task: Clean and validate data.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Tuple of (cleaned DataFrame, cleaning decisions)
    """
    logger.info("🧹 Cleaning data...")
    df_clean, decisions = clean_divorce_data(
        df,
        remove_dups=True,
        impute_strategy='median',
        validate_ranges=True
    )
    
    for decision in decisions:
        logger.info(f"  - {decision}")
    
    logger.info(f"✅ Data cleaned: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    return df_clean, decisions


@task(name="save_processed_data")
def save_processed_data_task(df: pd.DataFrame, decisions: list, 
                             output_dir: str = "data/processed") -> str:
    """
    Task: Save processed data and cleaning report.
    
    Args:
        df: Cleaned DataFrame
        decisions: List of cleaning decisions
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    logger.info("💾 Saving processed data...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save cleaned data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = output_path / f"divorce_clean_{timestamp}.csv"
    df.to_csv(data_file, index=False)
    
    # Save cleaning report
    report_file = output_path / f"cleaning_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write("CLEANING REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Columns: {len(df.columns)}\n\n")
        f.write("Decisions:\n")
        for decision in decisions:
            f.write(f"  - {decision}\n")
    
    logger.info(f"✅ Data saved to: {data_file}")
    logger.info(f"✅ Report saved to: {report_file}")
    
    return str(data_file)


@task(name="train_models")
def train_models_task(df: pd.DataFrame, experiment_name: str = "divorce-prediction") -> Dict:
    """
    Task: Train multiple models and track with MLflow.
    
    Args:
        df: Cleaned DataFrame
        experiment_name: MLflow experiment name
        
    Returns:
        Dictionary with training results
    """
    logger.info("🤖 Training models...")
    
    # Initialize trainer
    trainer = DivorceModelTrainer(
        experiment_name=experiment_name,
        models_dir="models"
    )
    
    # Prepare data
    X = df.drop('Divorce', axis=1)
    y = df['Divorce']
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns,
        index=X_test.index
    )
    
    # Train all models
    results = trainer.train_all_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Save best model
    model_path = trainer.save_best_model(scaler, X_train_scaled)
    
    logger.info(f"✅ Training completed. Best model: {trainer.best_model_name}")
    
    return {
        'best_model_name': trainer.best_model_name,
        'best_score': trainer.best_score,
        'model_path': model_path,
        'results': {name: res['metrics'] for name, res in results.items()}
    }


@task(name="evaluate_model")
def evaluate_model_task(training_results: Dict) -> Dict:
    """
    Task: Evaluate and validate the best model.
    
    Args:
        training_results: Results from training task
        
    Returns:
        Evaluation metrics
    """
    logger.info("📊 Evaluating best model...")
    
    best_model_name = training_results['best_model_name']
    best_score = training_results['best_score']
    
    # Define production readiness criteria
    min_f1_score = 0.75
    min_accuracy = 0.75
    
    results = training_results['results'][best_model_name]
    is_production_ready = (
        results['f1_score'] >= min_f1_score and
        results['accuracy'] >= min_accuracy
    )
    
    evaluation = {
        'model_name': best_model_name,
        'metrics': results,
        'production_ready': is_production_ready,
        'criteria': {
            'min_f1_score': min_f1_score,
            'min_accuracy': min_accuracy
        }
    }
    
    if is_production_ready:
        logger.info(f"✅ Model {best_model_name} is PRODUCTION READY")
    else:
        logger.warning(f"⚠️  Model {best_model_name} does NOT meet production criteria")
    
    logger.info(f"   F1 Score: {results['f1_score']:.4f}")
    logger.info(f"   Accuracy: {results['accuracy']:.4f}")
    
    return evaluation


@flow(name="ml_training_pipeline", task_runner=ConcurrentTaskRunner())
def ml_training_pipeline(data_path: str = "data", 
                        experiment_name: str = "divorce-prediction") -> Dict:
    """
    Complete ML training pipeline flow.
    
    This flow orchestrates:
    1. Data acquisition
    2. Data cleaning and validation
    3. Model training with MLflow tracking
    4. Model evaluation and selection
    
    Args:
        data_path: Path to data directory
        experiment_name: MLflow experiment name
        
    Returns:
        Dictionary with pipeline results
    """
    logger.info("="*60)
    logger.info("🚀 STARTING ML TRAINING PIPELINE")
    logger.info("="*60)
    
    # Step 1: Acquire data
    df_raw = acquire_data_task(data_path)
    
    # Step 2: Clean data
    df_clean, decisions = clean_data_task(df_raw)
    
    # Step 3: Save processed data
    data_file = save_processed_data_task(df_clean, decisions)
    
    # Step 4: Train models
    training_results = train_models_task(df_clean, experiment_name)
    
    # Step 5: Evaluate model
    evaluation = evaluate_model_task(training_results)
    
    logger.info("="*60)
    logger.info("✅ ML TRAINING PIPELINE COMPLETED")
    logger.info("="*60)
    
    return {
        'data_file': data_file,
        'training_results': training_results,
        'evaluation': evaluation,
        'timestamp': datetime.now().isoformat()
    }


@flow(name="batch_prediction_pipeline")
def batch_prediction_pipeline(input_file: str, model_path: str, 
                              output_file: str = None) -> str:
    """
    Batch prediction pipeline flow.
    
    Args:
        input_file: Path to input CSV file
        model_path: Path to trained model
        output_file: Path to save predictions (optional)
        
    Returns:
        Path to predictions file
    """
    import joblib
    
    logger.info("="*60)
    logger.info("🔮 STARTING BATCH PREDICTION PIPELINE")
    logger.info("="*60)
    
    # Load model
    logger.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    
    # Load scaler
    scaler_path = model_path.replace('best_model', 'scaler').replace('.pkl', '.pkl')
    if Path(scaler_path).exists():
        scaler = joblib.load(scaler_path)
    else:
        logger.warning("Scaler not found, predictions may be inaccurate")
        scaler = None
    
    # Load input data
    logger.info(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    
    # Prepare features
    if 'Divorce' in df.columns:
        X = df.drop('Divorce', axis=1)
    else:
        X = df
    
    # Scale if scaler available
    if scaler:
        X_scaled = scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    else:
        X_scaled = X
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Create output DataFrame
    output_df = df.copy()
    output_df['prediction'] = predictions
    if probabilities is not None:
        output_df['probability'] = probabilities
    output_df['timestamp'] = datetime.now().isoformat()
    
    # Save predictions
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"predictions_{timestamp}.csv"
    
    output_df.to_csv(output_file, index=False)
    logger.info(f"✅ Predictions saved to: {output_file}")
    
    logger.info("="*60)
    logger.info("✅ BATCH PREDICTION PIPELINE COMPLETED")
    logger.info("="*60)
    
    return output_file


if __name__ == "__main__":
    # Run the training pipeline
    results = ml_training_pipeline()
    
    print("\n" + "="*60)
    print("PIPELINE RESULTS")
    print("="*60)
    print(f"Best Model: {results['evaluation']['model_name']}")
    print(f"F1 Score: {results['evaluation']['metrics']['f1_score']:.4f}")
    print(f"Production Ready: {results['evaluation']['production_ready']}")
    print(f"Model Path: {results['training_results']['model_path']}")
    print("="*60)
