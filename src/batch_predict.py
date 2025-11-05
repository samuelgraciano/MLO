"""
Batch Prediction Script
Processes multiple records from input files and generates predictions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01-divorce-eda', 'scripts'))

import pandas as pd
import joblib
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/batch_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BatchPredictor:
    """Handles batch predictions from various file formats."""
    
    def __init__(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Initialize batch predictor.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to fitted scaler (optional)
        """
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path) if scaler_path else None
        
        # Load model
        logger.info(f"Loading model from: {self.model_path}")
        self.model = joblib.load(self.model_path)
        
        # Load scaler if provided
        if self.scaler_path and self.scaler_path.exists():
            logger.info(f"Loading scaler from: {self.scaler_path}")
            self.scaler = joblib.load(self.scaler_path)
        else:
            logger.warning("Scaler not provided or not found. Predictions may be inaccurate.")
            self.scaler = None
        
        logger.info("Batch predictor initialized successfully")
    
    def load_data(self, input_file: str) -> pd.DataFrame:
        """
        Load data from file (CSV, JSON, Parquet, or Pickle).
        
        Args:
            input_file: Path to input file
            
        Returns:
            DataFrame with input data
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        logger.info(f"Loading data from: {input_file}")
        
        # Determine file type and load
        suffix = input_path.suffix.lower()
        
        if suffix == '.csv':
            df = pd.read_csv(input_file)
        elif suffix == '.json':
            df = pd.read_json(input_file)
        elif suffix == '.parquet':
            df = pd.read_parquet(input_file)
        elif suffix in ['.pkl', '.pickle']:
            df = pd.read_pickle(input_file)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate input data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if valid
        """
        logger.info("Validating input data...")
        
        # Check for required columns
        expected_cols = [f"Atr{i+1}" for i in range(54)]
        missing_cols = set(expected_cols) - set(df.columns)
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check value ranges (0-4 for Likert scale)
        for col in expected_cols:
            if df[col].min() < 0 or df[col].max() > 4:
                logger.warning(f"Column {col} has values outside valid range [0-4]")
        
        # Check for missing values
        missing_count = df[expected_cols].isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values in features")
        
        logger.info("Data validation completed")
        return True
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on input data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with predictions
        """
        logger.info("Making predictions...")
        
        # Extract features
        feature_cols = [f"Atr{i+1}" for i in range(54)]
        X = df[feature_cols]
        
        # Scale features if scaler available
        if self.scaler:
            X_scaled = self.scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        else:
            X_scaled = X
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
        else:
            probabilities = [0.5] * len(X)
        
        # Add predictions to DataFrame
        result_df = df.copy()
        result_df['prediction'] = predictions
        result_df['probability'] = probabilities
        result_df['risk_level'] = result_df['probability'].apply(self._get_risk_level)
        result_df['timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Predictions completed: {len(predictions)} records")
        logger.info(f"Prediction distribution: {pd.Series(predictions).value_counts().to_dict()}")
        
        return result_df
    
    def _get_risk_level(self, probability: float) -> str:
        """Determine risk level based on probability."""
        if probability < 0.3:
            return "Low"
        elif probability < 0.7:
            return "Medium"
        else:
            return "High"
    
    def save_predictions(self, df: pd.DataFrame, output_file: str) -> None:
        """
        Save predictions to file.
        
        Args:
            df: DataFrame with predictions
            output_file: Path to output file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine output format
        suffix = output_path.suffix.lower()
        
        logger.info(f"Saving predictions to: {output_file}")
        
        if suffix == '.csv':
            df.to_csv(output_file, index=False)
        elif suffix == '.json':
            df.to_json(output_file, orient='records', indent=2)
        elif suffix == '.parquet':
            df.to_parquet(output_file, index=False)
        elif suffix in ['.pkl', '.pickle']:
            df.to_pickle(output_file)
        else:
            # Default to CSV
            output_file = str(output_path.with_suffix('.csv'))
            df.to_csv(output_file, index=False)
        
        logger.info(f"Predictions saved successfully: {output_file}")
    
    def process(self, input_file: str, output_file: Optional[str] = None) -> str:
        """
        Complete batch prediction pipeline.
        
        Args:
            input_file: Path to input file
            output_file: Path to output file (optional)
            
        Returns:
            Path to output file
        """
        logger.info("="*60)
        logger.info("BATCH PREDICTION PIPELINE")
        logger.info("="*60)
        
        # Load data
        df = self.load_data(input_file)
        
        # Validate data
        if not self.validate_data(df):
            raise ValueError("Data validation failed")
        
        # Make predictions
        result_df = self.predict(df)
        
        # Generate output filename if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_path = Path(input_file)
            output_file = f"predictions_{input_path.stem}_{timestamp}.csv"
        
        # Save predictions
        self.save_predictions(result_df, output_file)
        
        logger.info("="*60)
        logger.info("BATCH PREDICTION COMPLETED")
        logger.info("="*60)
        
        return output_file


def main():
    """Main entry point for batch prediction script."""
    parser = argparse.ArgumentParser(description='Batch Prediction for Divorce Prediction Model')
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input file (CSV, JSON, Parquet, or Pickle)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output file (optional, auto-generated if not provided)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file'
    )
    
    parser.add_argument(
        '--scaler',
        type=str,
        default=None,
        help='Path to fitted scaler file (optional)'
    )
    
    args = parser.parse_args()
    
    try:
        # Create logs directory
        Path('logs').mkdir(exist_ok=True)
        
        # Initialize predictor
        predictor = BatchPredictor(
            model_path=args.model,
            scaler_path=args.scaler
        )
        
        # Process batch predictions
        output_file = predictor.process(
            input_file=args.input,
            output_file=args.output
        )
        
        print(f"\n✅ Batch prediction completed successfully!")
        print(f"📁 Output file: {output_file}")
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
