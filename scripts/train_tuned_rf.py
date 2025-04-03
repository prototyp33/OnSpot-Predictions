#!/usr/bin/env python
"""
train_tuned_rf.py

Trains a Random Forest model using the optimal hyperparameters 
without repeating the full grid search.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
import joblib
import logging
import argparse
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_training_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer, List[str], List[str]]:
    """Load training and validation data with preprocessor."""
    logger.info(f"Loading train/validation data and preprocessor from {data_dir}...")
    try:
        train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        val_df = pd.read_csv(os.path.join(data_dir, 'validation.csv'))
        preprocessor = joblib.load(os.path.join(data_dir, 'preprocessor.pkl'))
        
        logger.info(f"Train data loaded: {train_df.shape[0]} samples, {train_df.shape[1]} columns")
        logger.info(f"Validation data loaded: {val_df.shape[0]} samples, {val_df.shape[1]} columns")
        
        # Infer feature columns from preprocessor
        numerical_cols = []
        categorical_cols = []
        
        if hasattr(preprocessor, 'transformers_'):
            for name, transformer, features in preprocessor.transformers_:
                if name == 'num': numerical_cols.extend(features)
                elif name == 'cat': categorical_cols.extend(features)
        
        logger.info(f"Inferred {len(numerical_cols)} numerical and {len(categorical_cols)} categorical features")
        return train_df, val_df, preprocessor, numerical_cols, categorical_cols
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def prepare_data(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame,
    preprocessor: ColumnTransformer, 
    numerical_cols: List[str], 
    categorical_cols: List[str],
    target_col: str = 'occupancy'
) -> Tuple[Any, Any, pd.Series, pd.Series]:
    """Prepare data using the loaded preprocessor."""
    logger.info("Processing train and validation data...")
    
    feature_cols = numerical_cols + categorical_cols
    
    # Separate features and target
    y_train = train_df[target_col]
    X_train = train_df[feature_cols]
    
    y_val = val_df[target_col]
    X_val = val_df[feature_cols]
    
    # Apply the preprocessor
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    
    logger.info(f"Processed training data shape: {X_train_proc.shape}")
    logger.info(f"Processed validation data shape: {X_val_proc.shape}")
    
    return X_train_proc, X_val_proc, y_train, y_val

def train_model(
    X_train: Any, 
    y_train: pd.Series,
    X_val: Any, 
    y_val: pd.Series,
    params: Dict[str, Any]
) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """Train Random Forest with optimal parameters and evaluate on validation set."""
    logger.info(f"Training Random Forest with parameters: {params}")
    
    # Convert 'null' from JSON to None
    if 'max_depth' in params and params['max_depth'] == 'null':
        params['max_depth'] = None
        
    # Create and train the model
    start_time = datetime.now()
    model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    
    # Evaluate on validation set
    start_time = datetime.now()
    val_pred = model.predict(X_val)
    inference_time = (datetime.now() - start_time).total_seconds()
    
    val_mse = mean_squared_error(y_val, val_pred)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    logger.info("Validation Set Performance:")
    logger.info(f"  R² Score:     {val_r2:.4f}")
    logger.info(f"  MAE:          {val_mae:.4f}")
    logger.info(f"  RMSE:         {val_rmse:.4f}")
    logger.info(f"  Inference Time: {inference_time:.4f} seconds")
    
    metrics = {
        'R2': val_r2,
        'MAE': val_mae,
        'RMSE': val_rmse,
        'MSE': val_mse,
        'InferenceTime': inference_time,
        'TrainingTime': training_time
    }
    
    return model, metrics

def main(data_dir: str, params_path: str = None, params_json: str = None, output_dir: str = None):
    """Main function to load data, train model, and save."""
    try:
        # 1. Load data
        train_df, val_df, preprocessor, numerical_cols, categorical_cols = load_training_data(data_dir)
        
        # 2. Prepare features
        X_train_proc, X_val_proc, y_train, y_val = prepare_data(
            train_df, val_df, preprocessor, numerical_cols, categorical_cols
        )
        
        # 3. Get model parameters
        if params_path:
            logger.info(f"Loading parameters from file: {params_path}")
            with open(params_path, 'r') as f:
                params = json.load(f)
        elif params_json:
            logger.info("Using provided parameter JSON string")
            params = json.loads(params_json)
        else:
            # Default best parameters from previous tuning
            logger.info("Using default best parameters")
            params = {
                'max_depth': None, 
                'max_features': 'sqrt',
                'min_samples_leaf': 1, 
                'min_samples_split': 2, 
                'n_estimators': 300
            }
        
        # 4. Train and evaluate
        model, metrics = train_model(X_train_proc, y_train, X_val_proc, y_val, params)
        
        # 5. Save model and metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use provided output directory or default to 'models'
        if output_dir is None:
            output_dir = 'models'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_filename = os.path.join(output_dir, f"tuned_rf_model_{timestamp}.pkl")
        joblib.dump(model, model_filename)
        logger.info(f"Model saved to {model_filename}")
        
        # Save metrics
        metrics_filename = os.path.join(output_dir, f"model_metrics_{timestamp}.json")
        with open(metrics_filename, 'w') as f:
            json.dump({
                'parameters': params,
                'metrics': metrics
            }, f, indent=4)
        logger.info(f"Metrics saved to {metrics_filename}")
        
        return model_filename
        
    except Exception as e:
        logger.error(f"Error in training process: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest with optimal parameters")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/splits_full/",
        help="Directory containing train.csv, validation.csv, and preprocessor.pkl"
    )
    parser.add_argument(
        "--params-path",
        type=str,
        help="Path to JSON file containing model parameters"
    )
    parser.add_argument(
        "--params",
        type=str,
        help="JSON string of model parameters"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save the trained model and metrics"
    )
    
    args = parser.parse_args()
    main(args.data_dir, args.params_path, args.params, args.output_dir) 