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
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import logging
import argparse
from datetime import datetime
import joblib
import json
import time

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory
output_dir = "model_comparison_results"
os.makedirs(output_dir, exist_ok=True)

def load_data(file_path):
    """Load and prepare the dataset for model comparison."""
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract time components
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'] >= 5
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week
    
    # Create cyclical features for time
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    logger.info(f"Dataset loaded with shape: {df.shape}")
    return df

def identify_variable_types(df):
    """Identify numerical and categorical variables in the dataset."""
    # Exclude target variable and timestamp
    exclude_cols = ['occupancy', 'timestamp', 'date']
    
    # Identify numerical and categorical columns
    numerical_cols = []
    categorical_cols = []
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        
        if np.issubdtype(df[col].dtype, np.number):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
    
    logger.info(f"Identified {len(numerical_cols)} numerical variables: {numerical_cols}")
    logger.info(f"Identified {len(categorical_cols)} categorical variables: {categorical_cols}")
    
    return numerical_cols, categorical_cols

def prepare_data_for_models(df, numerical_cols, categorical_cols, test_size=0.2):
    """Prepare data for model training and evaluation."""
    # Target variable
    y = df['occupancy']
    
    # Features
    X = df[numerical_cols + categorical_cols]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Preprocess data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_processed, X_test_processed, preprocessor

def prepare_time_series_data(df, numerical_cols, categorical_cols, sequence_length=24):
    """Prepare time series data for LSTM models."""
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Group by location_id if it exists
    if 'location_id' in df.columns:
        locations = df['location_id'].unique()
        X_sequences = []
        y_values = []
        
        for location in locations:
            loc_df = df[df['location_id'] == location].copy()
            
            # Create sequences
            for i in range(len(loc_df) - sequence_length):
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
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_values, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test, X_train_processed, X_test_processed):
    """Train and evaluate different models."""
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
        'CatBoost': cb.CatBoostRegressor(n_estimators=100, random_state=42, verbose=0)
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

def main(data_path):
    """Main function to run the model comparison."""
    logger.info("Starting model comparison...")
    
    # Load data
    df = load_data(data_path)
    
    # Identify variable types
    numerical_cols, categorical_cols = identify_variable_types(df)
    
    # Prepare data for traditional models
    X_train, X_test, y_train, y_test, X_train_processed, X_test_processed, preprocessor = prepare_data_for_models(
        df, numerical_cols, categorical_cols
    )
    
    # Train and evaluate traditional models
    results_df = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, X_train_processed, X_test_processed
    )
    
    # Prepare data for LSTM
    try:
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = prepare_time_series_data(
            df, numerical_cols, categorical_cols
        )
        
        # Train and evaluate LSTM
        lstm_result = train_lstm_model(X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm)
        
        # Add LSTM result to results_df
        results_df = pd.concat([results_df, pd.DataFrame([lstm_result])], ignore_index=True)
    except Exception as e:
        logger.warning(f"Could not train LSTM model: {e}")
    
    # Train ensemble model
    ensemble_result = train_ensemble_model(
        X_train, X_test, y_train, y_test, X_train_processed, X_test_processed, results_df
    )
    
    # Add ensemble result to results_df
    results_df = pd.concat([results_df, pd.DataFrame([ensemble_result])], ignore_index=True)
    
    # Sort by R²
    results_df = results_df.sort_values('r2', ascending=False)
    
    # Save final results
    results_df.to_csv(os.path.join(output_dir, 'final_model_comparison_results.csv'), index=False)
    
    # Create summary report
    with open(os.path.join(output_dir, 'model_comparison_summary.txt'), 'w') as f:
        f.write("=== MODEL COMPARISON SUMMARY ===\n\n")
        
        f.write("Model Performance (sorted by R²):\n")
        for _, row in results_df.iterrows():
            f.write(f"- {row['model']}: R² = {row['r2']:.4f}, RMSE = {row['rmse']:.4f}, MAE = {row['mae']:.4f}, Training Time = {row['training_time']:.2f}s\n")
        
        f.write("\nBest Model: " + results_df.iloc[0]['model'] + "\n")
        
        if 'components' in results_df.columns and not pd.isna(results_df.iloc[0].get('components', None)):
            f.write(f"Ensemble Components: {results_df.iloc[0]['components']}\n")
    
    logger.info(f"Model comparison completed. Results saved to {output_dir}")
    logger.info(f"Best model: {results_df.iloc[0]['model']} with R² = {results_df.iloc[0]['r2']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare different machine learning models for parking occupancy prediction")
    parser.add_argument("--data", default="data/prepared_data_improved.csv", help="Path to the prepared data file")
    
    args = parser.parse_args()
    main(args.data) 