#!/usr/bin/env python
"""
Script for experimenting with advanced model architectures for parking occupancy prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import os
import logging
import argparse
import joblib
import sys

# Try to import CatBoost, but don't fail if it's not available
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("CatBoost not available, skipping CatBoost models")

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from your existing modules
from scripts.parking_sim.advanced_features import engineer_advanced_features

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_advanced_models(data_path, output_dir="advanced_models"):
    """Train advanced models for parking occupancy prediction."""
    logger.info(f"Training advanced models using data from {data_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(data_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Generate advanced features
    df_advanced = engineer_advanced_features(df)
    
    # Prepare features and target
    exclude_cols = ['timestamp', 'date', 'occupancy']
    X = df_advanced.drop(columns=[col for col in exclude_cols if col in df_advanced.columns])
    y = df_advanced['occupancy']
    
    # Identify column types
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='drop'
    )
    
    # Split data for training and evaluation
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]  # Use last fold
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train XGBoost model
    logger.info("Training XGBoost model...")
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42))
    ])
    xgb_pipeline.fit(X_train, y_train)
    
    # Train LightGBM model
    logger.info("Training LightGBM model...")
    lgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42))
    ])
    lgb_pipeline.fit(X_train, y_train)
    
    # Train CatBoost model
    logger.info("Training CatBoost model...")
    cb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', cb.CatBoostRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, verbose=0))
    ])
    cb_pipeline.fit(X_train, y_train)
    
    # Create stacking ensemble
    logger.info("Training stacking ensemble...")
    estimators = [
        ('xgb', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)),
        ('lgb', lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)),
    ]

    if CATBOOST_AVAILABLE:
        estimators.append(('cb', cb.CatBoostRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbose=0)))

    stacking_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', StackingRegressor(estimators=estimators, final_estimator=xgb.XGBRegressor(n_estimators=50, random_state=42)))
    ])
    stacking_pipeline.fit(X_train, y_train)
    
    # Evaluate models
    models = {
        'xgboost': xgb_pipeline,
        'lightgbm': lgb_pipeline,
    }

    # Only add CatBoost if available
    if CATBOOST_AVAILABLE:
        models['catboost'] = cb_pipeline

    results = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        logger.info(f"{name.upper()} - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
        
        # Save model
        model_path = os.path.join(output_dir, f"{name}_model.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        results[name] = {
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'model_path': model_path
        }
    
    # Create summary report
    summary_path = os.path.join(output_dir, "advanced_models_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=== ADVANCED MODELS SUMMARY ===\n\n")
        
        # Sort models by performance
        sorted_models = sorted(results.items(), key=lambda x: x[1]['rmse'])
        
        for name, metrics in sorted_models:
            f.write(f"{name.upper()}:\n")
            f.write(f"  RMSE: {metrics['rmse']:.4f}\n")
            f.write(f"  R²: {metrics['r2']:.4f}\n")
            f.write(f"  MAE: {metrics['mae']:.4f}\n\n")
        
        # Best model
        best_model = sorted_models[0][0]
        f.write(f"Best model: {best_model.upper()} (RMSE: {sorted_models[0][1]['rmse']:.4f})\n")
    
    logger.info(f"Summary saved to {summary_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train advanced models for parking occupancy prediction")
    parser.add_argument("--data", default="data/prepared_data_improved.csv", help="Path to the prepared data file")
    parser.add_argument("--output", default="advanced_models", help="Output directory for trained models")
    
    args = parser.parse_args()
    
    train_advanced_models(args.data, args.output) 