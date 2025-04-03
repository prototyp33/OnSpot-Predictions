#!/usr/bin/env python
"""
tune_random_forest.py

Performs hyperparameter tuning for RandomForestRegressor using TimeSeriesSplit
cross-validation on pre-split data.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Needed for type hints potentially
import joblib
import logging
import argparse
import json
from datetime import datetime
from typing import Tuple, List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory for results
output_dir = "rf_tuning_results"
os.makedirs(output_dir, exist_ok=True)

# --- Data Loading (Adapted from compare_models.py) ---
def load_split_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer, List[str], List[str]]:
    """Load train, validation splits and the preprocessor."""
    logger.info(f"Loading train/validation splits and preprocessor from {data_dir}...")
    try:
        train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        val_df = pd.read_csv(os.path.join(data_dir, 'validation.csv'))
        # test_df = pd.read_csv(os.path.join(data_dir, 'test.csv')) # Test set not strictly needed for tuning
        preprocessor = joblib.load(os.path.join(data_dir, 'preprocessor.pkl'))
        logger.info("Train/validation splits and preprocessor loaded successfully.")

        # Infer numerical and categorical columns from the preprocessor
        numerical_cols = []
        categorical_cols = []
        
        if hasattr(preprocessor, 'transformers_'):
             for name, transformer, features in preprocessor.transformers_:
                 # Handle potential issues if feature names aren't stored directly
                 try:
                     if name == 'num' and isinstance(transformer, StandardScaler):
                         # Get feature names if available (depends on sklearn version and how it was saved)
                         num_features = getattr(transformer, 'feature_names_in_', features)
                         numerical_cols.extend(num_features)
                     elif name == 'cat' and isinstance(transformer, OneHotEncoder):
                         cat_features = getattr(transformer, 'feature_names_in_', features)
                         categorical_cols.extend(cat_features)
                 except AttributeError:
                      logger.warning(f"Could not get feature names for transformer '{name}'. Using original list: {features}")
                      if name == 'num': numerical_cols.extend(features)
                      if name == 'cat': categorical_cols.extend(features)

        else:
             logger.warning("Could not automatically infer feature names from preprocessor. Ensure column order matches.")
             # Need a way to get these if inference fails - maybe load from a config?
             # For now, this might lead to errors later if lists are empty.

        if not numerical_cols and not categorical_cols:
             logger.error("Failed to infer any numerical or categorical columns from the preprocessor!")
             # Attempt to infer from train_df as a fallback (less reliable)
             logger.warning("Attempting fallback: Inferring columns directly from train_df.")
             exclude_cols = ['timestamp', 'date', 'occupancy']
             temp_num = []
             temp_cat = []
             for col in train_df.columns:
                  if col in exclude_cols: continue
                  if np.issubdtype(train_df[col].dtype, np.number): temp_num.append(col)
                  else: temp_cat.append(col)
             numerical_cols = temp_num
             categorical_cols = temp_cat
             if not numerical_cols and not categorical_cols:
                   raise ValueError("Could not infer feature columns from preprocessor or training data.")


        logger.info(f"Inferred Numerical Columns: {len(numerical_cols)}")
        logger.info(f"Inferred Categorical Columns: {len(categorical_cols)}")

        # Convert timestamp columns
        for df in [train_df, val_df]:
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

        return train_df, val_df, preprocessor, numerical_cols, categorical_cols

    except FileNotFoundError as e:
        logger.error(f"Error loading files from {data_dir}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading: {e}")
        raise

# --- Data Preparation (Adapted from compare_models.py) ---
def prepare_data(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    preprocessor: ColumnTransformer, 
    numerical_cols: List[str], 
    categorical_cols: List[str],
    target_col: str = 'occupancy'
) -> Tuple[Any, Any, pd.Series, pd.Series]:
    """Prepare train and validation data using the loaded preprocessor."""
    logger.info("Applying loaded preprocessor to train/validation splits...")
    
    feature_cols = numerical_cols + categorical_cols
    
    # Basic checks
    if not feature_cols:
         raise ValueError("Feature column list is empty. Cannot proceed.")
    if target_col not in train_df.columns or target_col not in val_df.columns:
         raise ValueError(f"Target column '{target_col}' not found in loaded data.")
    missing_train_features = [col for col in feature_cols if col not in train_df.columns]
    if missing_train_features: raise ValueError(f"Missing features in train_df: {missing_train_features}")
    missing_val_features = [col for col in feature_cols if col not in val_df.columns]
    if missing_val_features: raise ValueError(f"Missing features in val_df: {missing_val_features}")

    # Separate features and target
    y_train = train_df[target_col]
    X_train = train_df[feature_cols]
    
    y_val = val_df[target_col]
    X_val = val_df[feature_cols]

    # Apply the *loaded* preprocessor (transform only)
    try:
        X_train_processed = preprocessor.transform(X_train)
        logger.info(f"Training data processed shape: {X_train_processed.shape}")
    except Exception as e:
        logger.error(f"Error transforming training data: {e}")
        raise
        
    try:
        X_val_processed = preprocessor.transform(X_val)
        logger.info(f"Validation data processed shape: {X_val_processed.shape}")
    except Exception as e:
        logger.error(f"Error transforming validation data: {e}")
        raise

    logger.info("Data preprocessing complete.")
    return X_train_processed, X_val_processed, y_train, y_val

# --- Evaluation Function (Similar to compare_models.py) ---
def evaluate(name: str, model: Any, X: Any, y: pd.Series) -> Dict[str, float]:
    """Evaluate a trained model."""
    start_time = datetime.now()
    y_pred = model.predict(X)
    end_time = datetime.now()
    
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    inference_time = (end_time - start_time).total_seconds()
    
    logger.info(f"{name} Evaluation:")
    logger.info(f"  MSE: {mse:.4f}")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  R2 Score: {r2:.4f}")
    logger.info(f"  Inference Time: {inference_time:.4f} seconds")
    
    return {'MSE': mse, 'MAE': mae, 'R2': r2, 'InferenceTime': inference_time}


# --- Main Tuning Logic ---
def main(data_dir: str, n_splits: int = 5, use_random_search: bool = False, n_iter: int = 10):
    """Load data, define search, run tuning, evaluate best model."""
    
    # Load data
    try:
        train_df, val_df, preprocessor, numerical_cols, categorical_cols = load_split_data(data_dir)
    except Exception as e:
        logger.error(f"Failed to load data. Exiting. Error: {e}")
        return

    # Prepare data
    try:
        X_train_proc, X_val_proc, y_train, y_val = prepare_data(
            train_df, val_df, preprocessor, numerical_cols, categorical_cols
        )
    except Exception as e:
         logger.error(f"Failed during data preparation. Exiting. Error: {e}")
         return

    # --- Define Hyperparameter Grid ---
    # Adjust grid based on computational resources and prior knowledge
    param_grid = {
        'n_estimators': [100, 200, 300], # Number of trees
        'max_depth': [10, 20, 30, None], # Max depth of trees (None means unlimited)
        'min_samples_split': [2, 5, 10], # Min samples required to split a node
        'min_samples_leaf': [1, 2, 4], # Min samples required at a leaf node
        'max_features': ['sqrt', 'log2', 1.0] # Number of features to consider at each split ('auto'/'sqrt', 'log2', float)
    }
    logger.info(f"Hyperparameter search grid: {param_grid}")

    # --- Setup Time Series Cross-Validation ---
    # Ensure n_splits is feasible given the training data size
    if n_splits >= len(train_df):
        logger.warning(f"n_splits ({n_splits}) >= number of samples ({len(train_df)}). Adjusting n_splits.")
        n_splits = max(2, len(train_df) // 2) # Adjust to a smaller feasible number
        
    tscv = TimeSeriesSplit(n_splits=n_splits)
    logger.info(f"Using TimeSeriesSplit with {n_splits} splits.")

    # --- Setup Search ---
    rf = RandomForestRegressor(random_state=42, n_jobs=-1) # Use multiple cores

    # Choose between GridSearchCV and RandomizedSearchCV
    if use_random_search:
        logger.info(f"Using RandomizedSearchCV with {n_iter} iterations.")
        search_cv = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_grid,
            n_iter=n_iter, # Number of parameter settings sampled
            cv=tscv,
            scoring='r2', # Or 'neg_mean_squared_error'
            n_jobs=1, # Can parallelize outer loop if needed, but RF already uses n_jobs=-1
            refit=True, # Refit the best estimator on the whole training data
            random_state=42,
            verbose=1 # Log progress
        )
    else:
        logger.info("Using GridSearchCV.")
        search_cv = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=tscv,
            scoring='r2', # Or 'neg_mean_squared_error'
            n_jobs=1, # Can parallelize outer loop if needed, but RF already uses n_jobs=-1
            refit=True, # Refit the best estimator on the whole training data
            verbose=1 # Log progress
        )

    # --- Run Search ---
    logger.info("Starting hyperparameter search...")
    start_time = datetime.now()
    search_cv.fit(X_train_proc, y_train)
    end_time = datetime.now()
    search_time = (end_time - start_time).total_seconds()
    logger.info(f"Hyperparameter search completed in {search_time:.2f} seconds.")

    # --- Best Model ---
    best_params = search_cv.best_params_
    best_score = search_cv.best_score_
    best_estimator = search_cv.best_estimator_ # Already refitted on full training data

    logger.info(f"\n--- Tuning Results ---")
    logger.info(f"Best Parameters Found: {best_params}")
    logger.info(f"Best CV Score (R2): {best_score:.4f}")

    # --- Evaluate Best Model on Validation Set ---
    logger.info("\n--- Evaluating Best Model on Validation Set ---")
    val_results = evaluate("Tuned Random Forest", best_estimator, X_val_proc, y_val)

    # --- Save Results ---
    results_summary = {
        'best_cv_r2': best_score,
        'best_params': best_params,
        'validation_results': val_results,
        'search_time_seconds': search_time,
        # Cast numpy types to standard python types for JSON serialization
        'n_splits_used': int(n_splits), # Cast to int
        'search_method': 'RandomizedSearchCV' if use_random_search else 'GridSearchCV',
        'iterations_or_grid_size': int(np.prod([len(v) for v in param_grid.values()])) if not use_random_search else int(n_iter) # Cast to int
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"rf_tuning_summary_{timestamp}.json")
    try:
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=4)
        logger.info(f"Tuning summary saved to {results_file}")
    except Exception as e:
        logger.error(f"Error saving tuning summary: {e}")

    # Optionally save the best model
    try:
        model_file = os.path.join(output_dir, f"tuned_rf_model_{timestamp}.pkl")
        joblib.dump(best_estimator, model_file)
        logger.info(f"Best tuned model saved to {model_file}")
    except Exception as e:
        logger.error(f"Error saving best model: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune RandomForestRegressor using TimeSeriesSplit.")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        required=True, 
        help="Directory containing train.csv, validation.csv, and preprocessor.pkl"
    )
    parser.add_argument(
        "--n-splits", 
        type=int, 
        default=5, 
        help="Number of splits for TimeSeriesSplit cross-validation."
    )
    parser.add_argument(
        "--random-search",
        action='store_true',
        help="Use RandomizedSearchCV instead of GridSearchCV.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=10,
        help="Number of iterations for RandomizedSearchCV (if used).",
    )
    
    args = parser.parse_args()
    main(
        data_dir=args.data_dir, 
        n_splits=args.n_splits, 
        use_random_search=args.random_search, 
        n_iter=args.n_iter
    ) 