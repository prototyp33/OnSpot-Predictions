#!/usr/bin/env python
"""
Script for comparing different machine learning models for parking occupancy prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import logging
import argparse
from datetime import datetime
import joblib
import json
import time
from typing import Tuple, List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory
output_dir = "model_comparison_results"
os.makedirs(output_dir, exist_ok=True)

def load_split_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ColumnTransformer, List[str], List[str]]:
    """Load train, validation, test splits and the preprocessor."""
    logger.info(f"Loading data splits and preprocessor from {data_dir}...")
    try:
        train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        val_df = pd.read_csv(os.path.join(data_dir, 'validation.csv'))
        test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        preprocessor = joblib.load(os.path.join(data_dir, 'preprocessor.pkl'))
        logger.info("Data splits and preprocessor loaded successfully.")

        # Infer numerical and categorical columns from the preprocessor
        numerical_cols = []
        categorical_cols = []
        
        # Check if preprocessor has transformers attribute
        if hasattr(preprocessor, 'transformers_'):
             for name, transformer, features in preprocessor.transformers_:
                 if name == 'num' and isinstance(transformer, StandardScaler):
                     numerical_cols.extend(features)
                 elif name == 'cat' and isinstance(transformer, OneHotEncoder):
                     categorical_cols.extend(features)
        else:
             logger.warning("Could not automatically infer feature names from preprocessor. Manual definition might be needed.")
             # Fallback or raise error if needed - for now, let it proceed maybe empty

        logger.info(f"Inferred Numerical Columns: {numerical_cols}")
        logger.info(f"Inferred Categorical Columns: {categorical_cols}")

        # Convert timestamp columns if they exist after loading
        for df in [train_df, val_df, test_df]:
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])


        return train_df, val_df, test_df, preprocessor, numerical_cols, categorical_cols

    except FileNotFoundError as e:
        logger.error(f"Error loading files from {data_dir}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading: {e}")
        raise

def prepare_data_for_models(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    preprocessor: ColumnTransformer, 
    numerical_cols: List[str], 
    categorical_cols: List[str],
    target_col: str = 'occupancy' # Define target column
) -> Tuple[Any, Any, Any, pd.Series, pd.Series, pd.Series]:
    """Prepare data using the loaded preprocessor."""
    logger.info("Applying loaded preprocessor to data splits...")
    
    feature_cols = numerical_cols + categorical_cols
    
    # Ensure feature columns exist in all dataframes
    for df_name, df in zip(['Train', 'Validation', 'Test'], [train_df, val_df, test_df]):
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required feature columns in {df_name} data: {missing_features}")
        missing_target = target_col not in df.columns
        if missing_target:
             raise ValueError(f"Missing target column '{target_col}' in {df_name} data.")


    # Separate features and target
    y_train = train_df[target_col]
    X_train = train_df[feature_cols]
    
    y_val = val_df[target_col]
    X_val = val_df[feature_cols]

    y_test = test_df[target_col]
    X_test = test_df[feature_cols]

    # Apply the *loaded* preprocessor (transform only)
    # Handle potential errors during transform
    try:
        X_train_processed = preprocessor.transform(X_train)
        logger.info(f"Training data processed shape: {X_train_processed.shape}")
    except Exception as e:
        logger.error(f"Error transforming training data: {e}")
        logger.error(f"Preprocessor features: {preprocessor.feature_names_in_}")
        logger.error(f"Input train features: {X_train.columns.tolist()}")
        raise
        
    try:
        X_val_processed = preprocessor.transform(X_val)
        logger.info(f"Validation data processed shape: {X_val_processed.shape}")
    except Exception as e:
        logger.error(f"Error transforming validation data: {e}")
        raise
        
    try:
        X_test_processed = preprocessor.transform(X_test)
        logger.info(f"Test data processed shape: {X_test_processed.shape}")
    except Exception as e:
        logger.error(f"Error transforming test data: {e}")
        raise

    logger.info("Data preprocessing complete using loaded preprocessor.")
    return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test

def prepare_time_series_data(df, numerical_cols, categorical_cols, sequence_length=24):
    """Prepare time series data for LSTM models."""
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # --- This part assumes access to the original columns before one-hot encoding ---
    # --- It also uses the raw numerical columns before scaling. Needs rework ---
    # --- For now, this function is incompatible with the new flow ---
    logger.warning("prepare_time_series_data is not adapted for pre-split/preprocessed data yet.")
    
    # Group by location_id if it exists
    if 'location_id' in df.columns:
        locations = df['location_id'].unique()
        X_sequences = []
        y_values = []
        
        # Placeholder for scaled numerical features - requires preprocessor to be fitted
        # scaler = StandardScaler()
        # df[numerical_cols] = scaler.fit_transform(df[numerical_cols]) 

        for location in locations:
            loc_df = df[df['location_id'] == location].copy()
            
            # Create sequences
            for i in range(len(loc_df) - sequence_length):
                # --- This needs scaled data ---
                X_seq = loc_df.iloc[i:i+sequence_length][numerical_cols].values 
                y_val = loc_df.iloc[i+sequence_length]['occupancy']
                X_sequences.append(X_seq)
                y_values.append(y_val)
        
        X_sequences = np.array(X_sequences)
        y_values = np.array(y_values)
    else:
        # Create sequences without location grouping
        X_sequences = []
        y_values = []
        
        for i in range(len(df) - sequence_length):
            X_seq = df.iloc[i:i+sequence_length][numerical_cols].values
            y_val = df.iloc[i+sequence_length]['occupancy']
            X_sequences.append(X_seq)
            y_values.append(y_val)
        
        X_sequences = np.array(X_sequences)
        y_values = np.array(y_values)
    
    # Split data (using simple split here, might need time-series split)
    X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_values, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test, X_train_processed, X_test_processed):
    """Train and evaluate different models."""
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        # "SVR": SVR(), # SVR can be slow, uncomment if needed
        # "KNN": KNeighborsRegressor() # KNN can be memory intensive
    }
    
    results = []
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        start_time = time.time()
        
        # Train model
        model.fit(X_train_processed, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_processed)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        training_time = time.time() - start_time
        
        results.append({
            'model': name,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'training_time': training_time
        })
        
        # Save model
        joblib.dump(model, os.path.join(output_dir, f"{name.replace(' ', '_').lower()}_model.pkl"))
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{name}: Actual vs Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name.replace(' ', '_').lower()}_actual_vs_predicted.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot error distribution
        errors = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title(f'{name}: Error Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name.replace(' ', '_').lower()}_error_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Convert to DataFrame and sort by R²
    results_df = pd.DataFrame(results).sort_values('r2', ascending=False)
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, 'model_comparison_results.csv'), index=False)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model', y='r2', data=results_df)
    plt.title('Model Comparison: R² Score')
    plt.xlabel('Model')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_r2.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model', y='rmse', data=results_df)
    plt.title('Model Comparison: RMSE')
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_rmse.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return results_df

def train_lstm_model(X_train, X_test, y_train, y_test):
    """Train and evaluate LSTM model for time series prediction."""
    logger.info("Training LSTM model...")
    
    # Get input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Create LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='mse')
    
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train model
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Save results
    result = {
        'model': 'LSTM',
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'training_time': training_time
    }
    
    # Save model
    model.save(os.path.join(output_dir, 'lstm_model'))
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM: Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lstm_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('LSTM: Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lstm_actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return result

def train_ensemble_model(X_train, X_test, y_train, y_test, X_train_processed, X_test_processed, results_df):
    """Train an ensemble model combining the best performing models."""
    logger.info("Training ensemble model...")
    
    # Select top 3 models
    top_models = results_df.head(3)['model'].values
    
    # Load models
    models = []
    for model_name in top_models:
        model_path = os.path.join(output_dir, f"{model_name.replace(' ', '_').lower()}_model.pkl")
        models.append(joblib.load(model_path))
    
    # Make predictions
    predictions = []
    for model in models:
        predictions.append(model.predict(X_test_processed))
    
    # Simple averaging ensemble
    ensemble_pred = np.mean(predictions, axis=0)
    
    # Evaluate
    mse = mean_squared_error(y_test, ensemble_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, ensemble_pred)
    mae = mean_absolute_error(y_test, ensemble_pred)
    
    # Save results
    result = {
        'model': 'Ensemble',
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'training_time': 0,  # No training time for ensemble
        'components': ', '.join(top_models)
    }
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, ensemble_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Ensemble: Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot error distribution
    errors = y_test - ensemble_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Ensemble: Error Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return result

def evaluate_model(name, model, X_test, y_test):
    """Evaluate a model and return performance metrics."""
    start_time = datetime.now()
    y_pred = model.predict(X_test)
    end_time = datetime.now()
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    inference_time = (end_time - start_time).total_seconds()
    
    logger.info(f"{name} Evaluation:")
    logger.info(f"  MSE: {mse:.4f}")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  R2 Score: {r2:.4f}")
    logger.info(f"  Inference Time: {inference_time:.4f} seconds")
    
    return {'MSE': mse, 'MAE': mae, 'R2': r2, 'InferenceTime': inference_time}

def main(data_dir: str, run_lstm: bool = False):
    """Main function to load data, prepare, train, and compare models."""
    
    # Load the pre-split data and preprocessor
    try:
        train_df, val_df, test_df, preprocessor, numerical_cols, categorical_cols = load_split_data(data_dir)
    except Exception as e:
        logger.error(f"Failed to load data. Exiting. Error: {e}")
        return

    # Prepare data using the loaded preprocessor
    try:
        X_train_proc, X_val_proc, X_test_proc, y_train, y_val, y_test = prepare_data_for_models(
            train_df, val_df, test_df, preprocessor, numerical_cols, categorical_cols
        )
    except Exception as e:
         logger.error(f"Failed during data preparation with loaded preprocessor. Exiting. Error: {e}")
         return

    # --- Define Models (Keep as is) ---
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        # "SVR": SVR(), # SVR can be slow, uncomment if needed
        # "KNN": KNeighborsRegressor() # KNN can be memory intensive
    }

    results = {}

    # Train and evaluate standard models
    for name, model in models.items():
        logger.info(f"--- Training {name} ---")
        start_time = datetime.now()
        # Use preprocessed training data
        model.fit(X_train_proc, y_train) 
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        logger.info(f"{name} Training Time: {training_time:.4f} seconds")
        
        # Evaluate on preprocessed test data
        results[name] = evaluate_model(name, model, X_test_proc, y_test)
        results[name]['TrainingTime'] = training_time

    # --- LSTM Handling ---
    if run_lstm:
        logger.info("--- Preparing Data for LSTM ---")
        # --- !! IMPORTANT !! ---
        # The prepare_time_series_data function needs adaptation to work correctly
        # with the pre-split/preprocessed data flow. 
        # It currently re-reads the original columns and splits again.
        # Running it now will likely fail or use improperly processed data.
        # We need the *original* train/val/test dataframes here *before* one-hot encoding, 
        # but *after* scaling numerical features appropriately based on the training set.
        # This requires more significant changes to the workflow.
        # For now, we demonstrate the structure but acknowledge the incompatibility.
        
        # Ideally, we'd adapt prepare_time_series_data or create a new function
        # that takes train_df, val_df, test_df (with original features + scaled numerical)
        # and creates sequences.
        
        logger.warning("LSTM preparation and training is using an unadapted function and may not work correctly with the pre-split workflow.")
        
        # Using train_df as a placeholder input - THIS IS INCORRECT for the intended workflow
        # It needs the full dataset before splitting to do sequence generation across the whole timeline.
        # Or, adapt sequence generation to work within the pre-defined splits.
        # X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = prepare_time_series_data(
        #     pd.concat([train_df, val_df, test_df]), # Hacky concatenation - loses split integrity
        #     numerical_cols, 
        #     categorical_cols, 
        #     sequence_length=24 
        # )
        
        # --- Placeholder: Skip LSTM training for now due to incompatibility ---
        logger.error("Skipping LSTM training due to incompatible data preparation function.")
        # if X_lstm_train.size > 0: # Check if data prep returned something
        #     logger.info("--- Training LSTM ---")
        #     input_shape = (X_lstm_train.shape[1], X_lstm_train.shape[2])
        #     lstm_model = build_lstm_model(input_shape)
        #     
        #     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        #     
        #     start_time = datetime.now()
        #     history = lstm_model.fit(
        #         X_lstm_train, y_lstm_train,
        #         epochs=50, # Adjust epochs
        #         batch_size=32, # Adjust batch size
        #         validation_split=0.2, # Use portion of training data for validation during training
        #         callbacks=[early_stopping],
        #         verbose=1
        #     )
        #     end_time = datetime.now()
        #     training_time = (end_time - start_time).total_seconds()
        #     logger.info(f"LSTM Training Time: {training_time:.4f} seconds")
            
        #     results["LSTM"] = evaluate_model("LSTM", lstm_model, X_lstm_test, y_lstm_test)
        #     results["LSTM"]['TrainingTime'] = training_time
        # else:
        #     logger.warning("LSTM data preparation yielded no sequences. Skipping LSTM.")

    # --- Results Analysis (Keep as is) ---
    results_df = pd.DataFrame(results).T.sort_values(by='R2', ascending=False)
    logger.info("\\n--- Model Comparison Results ---")
    logger.info(results_df)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"model_comparison_{timestamp}.csv")
    results_df.to_csv(results_file)
    logger.info(f"Comparison results saved to {results_file}")

    # Plotting (optional)
    try:
        results_df[['R2', 'MAE', 'MSE']].plot(kind='bar', secondary_y=['MAE', 'MSE'], figsize=(12, 6))
        plt.title('Model Comparison Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"model_metrics_{timestamp}.png")
        plt.savefig(plot_file)
        logger.info(f"Metrics plot saved to {plot_file}")
        # plt.show() # Uncomment to display plot interactively
    except Exception as e:
        logger.warning(f"Could not generate plot: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare performance of various regression models on pre-split parking data.")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        required=True, 
        help="Directory containing train.csv, validation.csv, test.csv, and preprocessor.pkl"
    )
    parser.add_argument(
        "--run-lstm",
        action='store_true', # Makes it a flag, presence means True
        help="Include LSTM model in comparison (Note: Data prep might be incompatible)."
    )
    
    args = parser.parse_args()
    main(data_dir=args.data_dir, run_lstm=args.run_lstm) 