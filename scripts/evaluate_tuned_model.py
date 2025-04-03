#!/usr/bin/env python
"""
evaluate_tuned_model.py

Evaluates a tuned model on the held-out test set to provide an unbiased 
assessment of generalization performance.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
import joblib
import logging
import argparse
import glob
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_data(data_dir: str) -> Tuple[pd.DataFrame, ColumnTransformer, List[str], List[str]]:
    """Load test data and preprocessor."""
    logger.info(f"Loading test data and preprocessor from {data_dir}...")
    try:
        test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        preprocessor = joblib.load(os.path.join(data_dir, 'preprocessor.pkl'))
        logger.info(f"Test data loaded: {test_df.shape[0]} samples, {test_df.shape[1]} columns")
        logger.info("Preprocessor loaded successfully")
        
        # Infer feature columns from preprocessor
        numerical_cols = []
        categorical_cols = []
        
        if hasattr(preprocessor, 'transformers_'):
            for name, transformer, features in preprocessor.transformers_:
                if name == 'num': numerical_cols.extend(features)
                elif name == 'cat': categorical_cols.extend(features)
        
        logger.info(f"Inferred {len(numerical_cols)} numerical and {len(categorical_cols)} categorical features")
        return test_df, preprocessor, numerical_cols, categorical_cols
        
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise

def find_latest_model(model_dir: str) -> str:
    """Find the most recently created tuned model in the specified directory."""
    model_files = glob.glob(os.path.join(model_dir, 'tuned_rf_model_*.pkl'))
    if not model_files:
        raise FileNotFoundError(f"No tuned models found in {model_dir}")
    
    # Sort by creation time (newest first)
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

def evaluate_model(model: Any, X: Any, y: pd.Series) -> Dict[str, float]:
    """Evaluate model on test data and return performance metrics."""
    start_time = datetime.now()
    y_pred = model.predict(X)
    inference_time = (datetime.now() - start_time).total_seconds()
    
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    logger.info("Test Set Performance Metrics:")
    logger.info(f"  R² Score:     {r2:.4f}")
    logger.info(f"  MAE:          {mae:.4f}")
    logger.info(f"  RMSE:         {rmse:.4f}")
    logger.info(f"  Inference Time: {inference_time:.4f} seconds")
    
    return {
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse,
        'MSE': mse,
        'InferenceTime': inference_time
    }

def main(data_dir: str, model_dir: str, model_path: str = None, output_dir: str = "model_evaluation_results/"):
    """Main evaluation function."""
    try:
        # 1. Load test data and preprocessor
        test_df, preprocessor, numerical_cols, categorical_cols = load_test_data(data_dir)
        
        # 2. Load tuned model
        if model_path:
            # Use specified model path
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            model_file = model_path
        else:
            # Find latest model
            model_file = find_latest_model(model_dir)
        
        logger.info(f"Loading tuned model from {model_file}")
        model = joblib.load(model_file)
        logger.info(f"Model loaded successfully: {type(model).__name__}")
        
        # 3. Process test features
        feature_cols = numerical_cols + categorical_cols
        target_col = 'occupancy'  # Assuming the target column name
        
        # Verify all required columns exist
        missing_features = [col for col in feature_cols if col not in test_df.columns]
        if missing_features:
            raise ValueError(f"Missing features in test data: {missing_features}")
        if target_col not in test_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in test data")
        
        # Separate features and target
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        # Transform features using preprocessor
        logger.info("Transforming test features...")
        X_test_proc = preprocessor.transform(X_test)
        logger.info(f"Test data processed shape: {X_test_proc.shape}")
        
        # 4. Evaluate model
        logger.info("Evaluating model on test set...")
        metrics = evaluate_model(model, X_test_proc, y_test)
        
        # 5. Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        
        results_file = os.path.join(output_dir, f"test_evaluation_{timestamp}.txt")
        with open(results_file, 'w') as f:
            f.write("=== Test Set Evaluation Results ===\n\n")
            f.write(f"Model: {os.path.basename(model_file)}\n")
            f.write(f"Test Data: {os.path.join(data_dir, 'test.csv')}\n")
            f.write(f"Samples: {len(y_test)}\n\n")
            f.write("Performance Metrics:\n")
            f.write(f"  R² Score:     {metrics['R2']:.4f}\n")
            f.write(f"  MAE:          {metrics['MAE']:.4f}\n")
            f.write(f"  RMSE:         {metrics['RMSE']:.4f}\n")
            f.write(f"  Inference Time: {metrics['InferenceTime']:.4f} seconds\n")
        
        logger.info(f"Evaluation results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate tuned model on test set")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/splits_full/",
        help="Directory containing test.csv and preprocessor.pkl"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/",
        help="Directory containing tuned model files (if no specific model path provided)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Specific path to the tuned model file (optional, will use latest from model-dir if not specified)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model_evaluation_results/",
        help="Directory to save evaluation results"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args.data_dir, args.model_dir, args.model_path, args.output_dir) 