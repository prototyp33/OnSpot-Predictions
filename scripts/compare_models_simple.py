#!/usr/bin/env python
"""
Simplified script for comparing machine learning models for parking occupancy prediction.
Uses only sklearn models to avoid dependency issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import os
import logging
import argparse
from datetime import datetime
import joblib
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

def train_and_evaluate_models(X_train, X_test, y_train, y_test, X_train_processed, X_test_processed):
    """Train and evaluate different models."""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'KNN': KNeighborsRegressor(n_neighbors=5)
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

def train_ensemble_model(X_train, X_test, y_train, y_test, X_train_processed, X_test_processed, results_df):
    """Train a simple ensemble model combining the top 3 models."""
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

def analyze_feature_importance(X_train, y_train, X_train_processed, preprocessor, numerical_cols, categorical_cols):
    """Analyze feature importance using Random Forest."""
    logger.info("Analyzing feature importance...")
    
    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_processed, y_train)
    
    # Get feature names after preprocessing
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols)
        elif name == 'cat':
            for col in cols:
                feature_names.extend([f"{col}_{cat}" for cat in X_train[col].unique()])
    
    # Get importance
    importance = model.feature_importances_
    
    # Create a DataFrame for feature importance
    if len(feature_names) == len(importance):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    else:
        # Handle case where feature names don't match importance array length
        logger.warning(f"Feature names length ({len(feature_names)}) doesn't match importance length ({len(importance)})")
        importance_df = pd.DataFrame({
            'feature_index': range(len(importance)),
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    # Save feature importance
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 10))
    top_features = importance_df.head(20)
    sns.barplot(x='importance', y='feature' if 'feature' in importance_df.columns else 'feature_index', 
                data=top_features)
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return importance_df

def main(data_path):
    """Main function to run the model comparison."""
    logger.info("Starting model comparison...")
    
    # Load data
    df = load_data(data_path)
    
    # Identify variable types
    numerical_cols, categorical_cols = identify_variable_types(df)
    
    # Prepare data for models
    X_train, X_test, y_train, y_test, X_train_processed, X_test_processed, preprocessor = prepare_data_for_models(
        df, numerical_cols, categorical_cols
    )
    
    # Train and evaluate models
    results_df = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, X_train_processed, X_test_processed
    )
    
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
    
    # Analyze feature importance
    importance_df = analyze_feature_importance(
        X_train, y_train, X_train_processed, preprocessor, numerical_cols, categorical_cols
    )
    
    # Create summary report
    with open(os.path.join(output_dir, 'model_comparison_summary.txt'), 'w') as f:
        f.write("=== MODEL COMPARISON SUMMARY ===\n\n")
        
        f.write("Model Performance (sorted by R²):\n")
        for _, row in results_df.iterrows():
            f.write(f"- {row['model']}: R² = {row['r2']:.4f}, RMSE = {row['rmse']:.4f}, MAE = {row['mae']:.4f}, Training Time = {row['training_time']:.2f}s\n")
        
        f.write("\nBest Model: " + results_df.iloc[0]['model'] + "\n")
        
        if 'components' in results_df.columns and not pd.isna(results_df.iloc[0].get('components', None)):
            f.write(f"Ensemble Components: {results_df.iloc[0]['components']}\n")
        
        f.write("\nTop 10 Important Features:\n")
        for _, row in importance_df.head(10).iterrows():
            feature_name = row['feature'] if 'feature' in importance_df.columns else f"Feature {row['feature_index']}"
            f.write(f"- {feature_name}: {row['importance']:.4f}\n")
    
    logger.info(f"Model comparison completed. Results saved to {output_dir}")
    logger.info(f"Best model: {results_df.iloc[0]['model']} with R² = {results_df.iloc[0]['r2']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare different machine learning models for parking occupancy prediction")
    parser.add_argument("--data", default="data/prepared_data_improved.csv", help="Path to the prepared data file")
    
    args = parser.parse_args()
    main(args.data) 