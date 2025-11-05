"""
Model Training Module with MLflow Tracking
Trains multiple models and tracks experiments using MLflow.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01-divorce-eda', 'scripts'))

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, classification_report)
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
from datetime import datetime

# Import existing data loading and cleaning functions
from load_data import load_divorce_data
from data_cleaning import clean_divorce_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DivorceModelTrainer:
    """Trains and evaluates divorce prediction models with MLflow tracking."""
    
    def __init__(self, experiment_name: str = "divorce-prediction", 
                 models_dir: str = "models",
                 random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            experiment_name: Name for MLflow experiment
            models_dir: Directory to save trained models
            random_state: Random seed for reproducibility
        """
        self.experiment_name = experiment_name
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        
        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to: {experiment_name}")
        
        # Initialize models
        self.models = self._get_models()
        
        # Storage for results
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        
    def _get_models(self) -> Dict[str, Any]:
        """Define models to train."""
        return {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state,
                learning_rate=0.1
            ),
            'svm': SVC(
                kernel='rbf',
                random_state=self.random_state,
                class_weight='balanced',
                probability=True
            ),
            'xgboost': XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        }
    
    def load_and_prepare_data(self, test_size: float = 0.2) -> Tuple:
        """
        Load and prepare data for training.
        
        Args:
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, scaler)
        """
        logger.info("Loading data...")
        
        # Load data using existing function
        df = load_divorce_data(data_path="data")
        
        # Clean data using existing function
        logger.info("Cleaning data...")
        df_clean, cleaning_decisions = clean_divorce_data(
            df,
            remove_dups=True,
            impute_strategy='median',
            validate_ranges=True
        )
        
        for decision in cleaning_decisions:
            logger.info(f"  - {decision}")
        
        # Separate features and target
        X = df_clean.drop('Divorce', axis=1)
        y = df_clean['Divorce']
        
        logger.info(f"Dataset shape: {X.shape}")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrames to preserve column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
        
        logger.info(f"Training set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def evaluate_model(self, model, X_test, y_test) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        return metrics
    
    def train_model(self, model_name: str, model, X_train, X_test, y_train, y_test, 
                   log_to_mlflow: bool = True) -> Dict[str, float]:
        """
        Train a single model and log to MLflow.
        
        Args:
            model_name: Name of the model
            model: Model instance
            X_train, X_test, y_train, y_test: Train/test data
            log_to_mlflow: Whether to log to MLflow
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"\nTraining {model_name}...")
        
        if log_to_mlflow:
            with mlflow.start_run(run_name=model_name):
                # Log parameters
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    mlflow.log_params(params)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                metrics = self.evaluate_model(model, X_test, y_test)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                # Log additional info
                mlflow.set_tag("model_type", model_name)
                mlflow.set_tag("training_date", datetime.now().isoformat())
                
                logger.info(f"Metrics for {model_name}:")
                for metric_name, value in metrics.items():
                    logger.info(f"  {metric_name}: {value:.4f}")
                
                return metrics
        else:
            model.fit(X_train, y_train)
            metrics = self.evaluate_model(model, X_test, y_test)
            return metrics
    
    def train_all_models(self, X_train, X_test, y_train, y_test) -> Dict[str, Dict]:
        """
        Train all models and track with MLflow.
        
        Args:
            X_train, X_test, y_train, y_test: Train/test data
            
        Returns:
            Dictionary of results for all models
        """
        logger.info("="*60)
        logger.info("TRAINING ALL MODELS")
        logger.info("="*60)
        
        for model_name, model in self.models.items():
            try:
                metrics = self.train_model(
                    model_name, model, X_train, X_test, y_train, y_test
                )
                self.results[model_name] = {
                    'model': model,
                    'metrics': metrics
                }
                
                # Track best model
                if metrics['f1_score'] > self.best_score:
                    self.best_score = metrics['f1_score']
                    self.best_model = model
                    self.best_model_name = model_name
                    
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETED")
        logger.info("="*60)
        logger.info(f"Best model: {self.best_model_name} (F1: {self.best_score:.4f})")
        
        return self.results
    
    def save_best_model(self, scaler, X_train) -> str:
        """
        Save the best model and scaler.
        
        Args:
            scaler: Fitted scaler
            X_train: Training data (for feature names)
            
        Returns:
            Path to saved model
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.models_dir / f"best_model_{self.best_model_name}_{timestamp}.pkl"
        scaler_path = self.models_dir / f"scaler_{timestamp}.pkl"
        metadata_path = self.models_dir / f"model_metadata_{timestamp}.json"
        
        # Save model
        joblib.dump(self.best_model, model_path)
        logger.info(f"Best model saved to: {model_path}")
        
        # Save scaler
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to: {scaler_path}")
        
        # Save metadata
        import json
        metadata = {
            'model_name': self.best_model_name,
            'f1_score': self.best_score,
            'metrics': self.results[self.best_model_name]['metrics'],
            'feature_names': list(X_train.columns),
            'training_date': timestamp,
            'model_path': str(model_path),
            'scaler_path': str(scaler_path)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return str(model_path)
    
    def print_results_summary(self):
        """Print a summary of all model results."""
        logger.info("\n" + "="*60)
        logger.info("RESULTS SUMMARY")
        logger.info("="*60)
        
        # Create results DataFrame
        results_data = []
        for model_name, result in self.results.items():
            row = {'Model': model_name}
            row.update(result['metrics'])
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('f1_score', ascending=False)
        
        print("\n", results_df.to_string(index=False))
        print("\n" + "="*60)


def main():
    """Main training pipeline."""
    # Initialize trainer
    trainer = DivorceModelTrainer(
        experiment_name="divorce-prediction-baseline",
        models_dir="models"
    )
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, scaler = trainer.load_and_prepare_data(test_size=0.2)
    
    # Train all models
    results = trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    # Print summary
    trainer.print_results_summary()
    
    # Save best model
    model_path = trainer.save_best_model(scaler, X_train)
    
    logger.info(f"\nTraining pipeline completed successfully!")
    logger.info(f"Best model: {trainer.best_model_name}")
    logger.info(f"Best F1 Score: {trainer.best_score:.4f}")
    logger.info(f"Model saved to: {model_path}")
    
    return trainer


if __name__ == "__main__":
    trainer = main()
