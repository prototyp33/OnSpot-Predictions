#!/usr/bin/env python
"""
Script for implementing time series cross-validation on parking occupancy models.
Evaluates model performance across multiple time periods to ensure robustness.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os
import logging
import argparse
from datetime import datetime
import joblib
import sys

# Add the project root to the Python path to allow imports from scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from your existing modules
from scripts.parking_sim.advanced_features import engineer_advanced_features

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory
output_dir = "cross_validation_results"
os.makedirs(output_dir, exist_ok=True)

def load_data(file_path):
    """Load and prepare the dataset for cross-validation."""
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Sort by timestamp for time series cross-validation
        df = df.sort_values('timestamp')
    
    logger.info(f"Dataset loaded with shape: {df.shape}")
    return df

def identify_column_types(df):
    """Identify numeric and categorical columns in the dataframe."""
    # Exclude target and metadata columns
    exclude_cols = ['occupancy', 'timestamp', 'date']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Identify numeric and categorical columns
    numeric_cols = df[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    logger.info(f"Identified {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns")
    if categorical_cols:
        logger.info(f"Categorical columns: {categorical_cols}")
    
    return numeric_cols, categorical_cols

def cross_validate_global_model(df, feature_set_name, n_splits=5):
    """Perform time series cross-validation on a global model."""
    logger.info(f"Performing {n_splits}-fold time series cross-validation on global model with {feature_set_name} features...")
    
    # Prepare features and target
    exclude_cols = ['timestamp', 'date', 'occupancy']
    X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    y = df['occupancy']
    
    # Identify column types
    numeric_cols, categorical_cols = identify_column_types(df)
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='drop'
    )
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])
    
    # Set up time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Track metrics across folds
    fold_metrics = []
    all_y_true = []
    all_y_pred = []
    fold_sizes = []
    
    # Perform cross-validation
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        logger.info(f"Training fold {fold+1}/{n_splits}...")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Track fold size
        fold_sizes.append(len(test_idx))
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Store actual and predicted values
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Store metrics
        fold_metrics.append({
            'fold': fold + 1,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'train_size': len(train_idx),
            'test_size': len(test_idx)
        })
        
        logger.info(f"Fold {fold+1} - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
    
    # Calculate average metrics
    avg_rmse = np.mean([m['rmse'] for m in fold_metrics])
    avg_r2 = np.mean([m['r2'] for m in fold_metrics])
    avg_mae = np.mean([m['mae'] for m in fold_metrics])
    
    logger.info(f"Average metrics - RMSE: {avg_rmse:.4f}, R²: {avg_r2:.4f}, MAE: {avg_mae:.4f}")
    
    # Calculate weighted average metrics (weighted by fold size)
    weighted_rmse = np.average([m['rmse'] for m in fold_metrics], weights=fold_sizes)
    weighted_r2 = np.average([m['r2'] for m in fold_metrics], weights=fold_sizes)
    weighted_mae = np.average([m['mae'] for m in fold_metrics], weights=fold_sizes)
    
    logger.info(f"Weighted average metrics - RMSE: {weighted_rmse:.4f}, R²: {weighted_r2:.4f}, MAE: {weighted_mae:.4f}")
    
    # Calculate overall metrics on all predictions
    overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    overall_r2 = r2_score(all_y_true, all_y_pred)
    overall_mae = mean_absolute_error(all_y_true, all_y_pred)
    
    logger.info(f"Overall metrics - RMSE: {overall_rmse:.4f}, R²: {overall_r2:.4f}, MAE: {overall_mae:.4f}")
    
    # Visualize cross-validation results
    visualize_cv_results(fold_metrics, feature_set_name, "global")
    
    # Train final model on all data
    logger.info("Training final model on all data...")
    pipeline.fit(X, y)
    
    # Save final model
    model_path = os.path.join(output_dir, f"global_model_{feature_set_name}_cv.pkl")
    joblib.dump(pipeline, model_path)
    logger.info(f"Final model saved to {model_path}")
    
    return {
        'fold_metrics': fold_metrics,
        'avg_metrics': {
            'rmse': avg_rmse,
            'r2': avg_r2,
            'mae': avg_mae
        },
        'weighted_metrics': {
            'rmse': weighted_rmse,
            'r2': weighted_r2,
            'mae': weighted_mae
        },
        'overall_metrics': {
            'rmse': overall_rmse,
            'r2': overall_r2,
            'mae': overall_mae
        },
        'model_path': model_path
    }

def cross_validate_location_models(df, feature_set_name, n_splits=5):
    """Perform cross-validation on location-specific models."""
    logger.info(f"Performing {n_splits}-fold time series cross-validation on location-specific models with {feature_set_name} features...")
    
    # Get unique locations
    locations = df['location_id'].unique()
    logger.info(f"Training models for {len(locations)} locations")
    
    results = {}
    
    for location in locations:
        logger.info(f"Cross-validating model for location {location}...")
        
        # Filter data for this location
        loc_df = df[df['location_id'] == location].copy()
        
        # Prepare features and target
        exclude_cols = ['timestamp', 'date', 'occupancy', 'location_id']
        X = loc_df.drop(columns=[col for col in exclude_cols if col in loc_df.columns])
        y = loc_df['occupancy']
        
        # Identify column types - IMPORTANT: location_id is now removed
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        logger.info(f"Identified {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns")
        logger.info(f"Categorical columns: {categorical_cols}")
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ],
            remainder='drop'
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
        
        # Set up time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Initialize metrics
        fold_metrics = []
        all_y_true = []
        all_y_pred = []
        
        # Perform cross-validation
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            logger.info(f"Training fold {fold}/{n_splits} for location {location}...")
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            logger.info(f"Fold {fold} - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
            
            # Store metrics
            fold_metrics.append({
                'fold': fold,
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'n_samples': len(y_test)
            })
            
            # Store predictions for overall metrics
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
        
        # Calculate average metrics
        avg_rmse = np.mean([m['rmse'] for m in fold_metrics])
        avg_r2 = np.mean([m['r2'] for m in fold_metrics])
        avg_mae = np.mean([m['mae'] for m in fold_metrics])
        
        # Calculate weighted average metrics
        total_samples = sum(m['n_samples'] for m in fold_metrics)
        weighted_rmse = sum(m['rmse'] * m['n_samples'] / total_samples for m in fold_metrics)
        weighted_r2 = sum(m['r2'] * m['n_samples'] / total_samples for m in fold_metrics)
        weighted_mae = sum(m['mae'] * m['n_samples'] / total_samples for m in fold_metrics)
        
        # Calculate overall metrics
        overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
        overall_r2 = r2_score(all_y_true, all_y_pred)
        overall_mae = mean_absolute_error(all_y_true, all_y_pred)
        
        logger.info(f"Average metrics - RMSE: {avg_rmse:.4f}, R²: {avg_r2:.4f}, MAE: {avg_mae:.4f}")
        logger.info(f"Weighted average metrics - RMSE: {weighted_rmse:.4f}, R²: {weighted_r2:.4f}, MAE: {weighted_mae:.4f}")
        logger.info(f"Overall metrics - RMSE: {overall_rmse:.4f}, R²: {overall_r2:.4f}, MAE: {overall_mae:.4f}")
        
        # Train final model on all data
        logger.info("Training final model on all data...")
        pipeline.fit(X, y)
        
        # Save model
        model_dir = os.path.join(output_dir, "location_models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"location_{location}_{feature_set_name}_cv.pkl")
        joblib.dump(pipeline, model_path)
        logger.info(f"Final model saved to {model_path}")
        
        # Store results
        results[location] = {
            'fold_metrics': fold_metrics,
            'avg_rmse': avg_rmse,
            'avg_r2': avg_r2,
            'avg_mae': avg_mae,
            'weighted_rmse': weighted_rmse,
            'weighted_r2': weighted_r2,
            'weighted_mae': weighted_mae,
            'overall_rmse': overall_rmse,
            'overall_r2': overall_r2,
            'overall_mae': overall_mae,
            'model_path': model_path
        }
        
        # Visualize results
        visualize_cv_results(fold_metrics, feature_set_name, f"location_{location}")
    
    return results

def visualize_cv_results(fold_metrics, feature_set_name, model_type):
    """Visualize cross-validation results."""
    # Convert to DataFrame
    metrics_df = pd.DataFrame(fold_metrics)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot RMSE by fold
    if 'location' in metrics_df.columns:
        sns.boxplot(x='fold', y='rmse', data=metrics_df, ax=axes[0])
        sns.stripplot(x='fold', y='rmse', data=metrics_df, hue='location', dodge=True, ax=axes[0])
        axes[0].set_title(f'RMSE by Fold - {model_type.replace("_", " ").title()} Model with {feature_set_name.replace("_", " ").title()} Features')
    else:
        sns.barplot(x='fold', y='rmse', data=metrics_df, ax=axes[0])
        axes[0].set_title(f'RMSE by Fold - {model_type.replace("_", " ").title()} Model with {feature_set_name.replace("_", " ").title()} Features')
    
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('RMSE')
    axes[0].grid(True, alpha=0.3)
    
    # Plot R² by fold
    if 'location' in metrics_df.columns:
        sns.boxplot(x='fold', y='r2', data=metrics_df, ax=axes[1])
        sns.stripplot(x='fold', y='r2', data=metrics_df, hue='location', dodge=True, ax=axes[1])
        axes[1].set_title(f'R² by Fold - {model_type.replace("_", " ").title()} Model with {feature_set_name.replace("_", " ").title()} Features')
    else:
        sns.barplot(x='fold', y='r2', data=metrics_df, ax=axes[1])
        axes[1].set_title(f'R² by Fold - {model_type.replace("_", " ").title()} Model with {feature_set_name.replace("_", " ").title()} Features')
    
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('R²')
    axes[1].grid(True, alpha=0.3)
    
    # Plot MAE by fold
    if 'location' in metrics_df.columns:
        sns.boxplot(x='fold', y='mae', data=metrics_df, ax=axes[2])
        sns.stripplot(x='fold', y='mae', data=metrics_df, hue='location', dodge=True, ax=axes[2])
        axes[2].set_title(f'MAE by Fold - {model_type.replace("_", " ").title()} Model with {feature_set_name.replace("_", " ").title()} Features')
    else:
        sns.barplot(x='fold', y='mae', data=metrics_df, ax=axes[2])
        axes[2].set_title(f'MAE by Fold - {model_type.replace("_", " ").title()} Model with {feature_set_name.replace("_", " ").title()} Features')
    
    axes[2].set_xlabel('Fold')
    axes[2].set_ylabel('MAE')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cv_results_{model_type}_{feature_set_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def visualize_location_comparison(location_results, feature_set_name):
    """Visualize comparison of metrics across locations."""
    # Create DataFrame with location metrics
    loc_metrics = []
    for loc, results in location_results.items():
        loc_metrics.append({
            'location': loc,
            'rmse': results['overall_metrics']['rmse'],
            'r2': results['overall_metrics']['r2'],
            'mae': results['overall_metrics']['mae'],
            'data_size': results['data_size']
        })
    
    loc_df = pd.DataFrame(loc_metrics)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot RMSE by location
    sns.barplot(x='location', y='rmse', data=loc_df, ax=axes[0, 0])
    axes[0, 0].set_title(f'RMSE by Location - {feature_set_name.replace("_", " ").title()} Features')
    axes[0, 0].set_xlabel('Location')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot R² by location
    sns.barplot(x='location', y='r2', data=loc_df, ax=axes[0, 1])
    axes[0, 1].set_title(f'R² by Location - {feature_set_name.replace("_", " ").title()} Features')
    axes[0, 1].set_xlabel('Location')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot MAE by location
    sns.barplot(x='location', y='mae', data=loc_df, ax=axes[1, 0])
    axes[1, 0].set_title(f'MAE by Location - {feature_set_name.replace("_", " ").title()} Features')
    axes[1, 0].set_xlabel('Location')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot data size vs R²
    sns.scatterplot(x='data_size', y='r2', data=loc_df, s=100, ax=axes[1, 1])
    for i, row in loc_df.iterrows():
        axes[1, 1].text(row['data_size'], row['r2'], row['location'], fontsize=10)
    
    axes[1, 1].set_title(f'R² vs Data Size - {feature_set_name.replace("_", " ").title()} Features')
    axes[1, 1].set_xlabel('Number of Data Points')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"location_comparison_{feature_set_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_report(global_basic_results, global_advanced_results, loc_basic_results, loc_advanced_results):
    """Create a summary report of cross-validation results."""
    logger.info("Creating summary report...")
    
    summary_path = os.path.join(output_dir, "cv_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("=== CROSS-VALIDATION SUMMARY ===\n\n")
        
        # Global model results
        f.write("GLOBAL MODEL RESULTS\n")
        f.write("-------------------\n\n")
        
        f.write("Basic Features:\n")
        f.write(f"  RMSE: {global_basic_results['overall_metrics']['rmse']:.4f}\n")
        f.write(f"  R²: {global_basic_results['overall_metrics']['r2']:.4f}\n")
        f.write(f"  MAE: {global_basic_results['overall_metrics']['mae']:.4f}\n\n")
        
        f.write("Advanced Features:\n")
        f.write(f"  RMSE: {global_advanced_results['overall_metrics']['rmse']:.4f}\n")
        f.write(f"  R²: {global_advanced_results['overall_metrics']['r2']:.4f}\n")
        f.write(f"  MAE: {global_advanced_results['overall_metrics']['mae']:.4f}\n\n")
        
        # Location-specific model results
        f.write("LOCATION-SPECIFIC MODEL RESULTS\n")
        f.write("------------------------------\n\n")
        
        if loc_basic_results and loc_advanced_results:
            # Calculate average metrics across locations for basic features
            basic_avg_rmse = np.mean([loc_data['overall_rmse'] for loc_data in loc_basic_results.values()])
            basic_avg_r2 = np.mean([loc_data['overall_r2'] for loc_data in loc_basic_results.values()])
            basic_avg_mae = np.mean([loc_data['overall_mae'] for loc_data in loc_basic_results.values()])
            
            # Calculate average metrics across locations for advanced features
            adv_avg_rmse = np.mean([loc_data['overall_rmse'] for loc_data in loc_advanced_results.values()])
            adv_avg_r2 = np.mean([loc_data['overall_r2'] for loc_data in loc_advanced_results.values()])
            adv_avg_mae = np.mean([loc_data['overall_mae'] for loc_data in loc_advanced_results.values()])
            
            f.write("Average across all locations:\n\n")
            
            f.write("Basic Features:\n")
            f.write(f"  RMSE: {basic_avg_rmse:.4f}\n")
            f.write(f"  R²: {basic_avg_r2:.4f}\n")
            f.write(f"  MAE: {basic_avg_mae:.4f}\n\n")
            
            f.write("Advanced Features:\n")
            f.write(f"  RMSE: {adv_avg_rmse:.4f}\n")
            f.write(f"  R²: {adv_avg_r2:.4f}\n")
            f.write(f"  MAE: {adv_avg_mae:.4f}\n\n")
            
            f.write("Individual Location Results:\n\n")
            
            # Get common locations
            common_locations = set(loc_basic_results.keys()) & set(loc_advanced_results.keys())
            
            for loc in common_locations:
                f.write(f"Location {loc}:\n")
                
                f.write("  Basic Features:\n")
                f.write(f"    RMSE: {loc_basic_results[loc]['overall_rmse']:.4f}\n")
                f.write(f"    R²: {loc_basic_results[loc]['overall_r2']:.4f}\n")
                f.write(f"    MAE: {loc_basic_results[loc]['overall_mae']:.4f}\n\n")
                
                f.write("  Advanced Features:\n")
                f.write(f"    RMSE: {loc_advanced_results[loc]['overall_rmse']:.4f}\n")
                f.write(f"    R²: {loc_advanced_results[loc]['overall_r2']:.4f}\n")
                f.write(f"    MAE: {loc_advanced_results[loc]['overall_mae']:.4f}\n\n")
        else:
            f.write("Location-specific models were not evaluated.\n\n")
        
        # Comparison and recommendations
        f.write("COMPARISON AND RECOMMENDATIONS\n")
        f.write("-----------------------------\n\n")
        
        # Compare global models
        global_basic_rmse = global_basic_results['overall_metrics']['rmse']
        global_advanced_rmse = global_advanced_results['overall_metrics']['rmse']
        
        f.write("Global Model Comparison:\n")
        f.write(f"  Basic Features RMSE: {global_basic_rmse:.4f}\n")
        f.write(f"  Advanced Features RMSE: {global_advanced_rmse:.4f}\n")
        f.write(f"  Improvement: {(1 - global_advanced_rmse / global_basic_rmse) * 100:.2f}%\n\n")
        
        # Compare location-specific vs global
        if loc_basic_results and loc_advanced_results:
            f.write("Location-specific vs Global Model:\n")
            f.write(f"  Global Advanced RMSE: {global_advanced_rmse:.4f}\n")
            f.write(f"  Avg Location Advanced RMSE: {adv_avg_rmse:.4f}\n")
            f.write(f"  Improvement: {(1 - adv_avg_rmse / global_advanced_rmse) * 100:.2f}%\n\n")
            
            # Recommendations
            f.write("Recommendations:\n")
            if adv_avg_rmse < global_advanced_rmse:
                f.write("  - Use location-specific models with advanced features for best performance\n")
                f.write("  - Consider ensemble of location-specific models\n")
            else:
                f.write("  - Use global model with advanced features for best performance\n")
                f.write("  - Location-specific models don't provide significant improvement\n")
        else:
            f.write("Recommendations:\n")
            f.write("  - Use global model with advanced features for best performance\n")
    
    logger.info(f"Summary report saved to {summary_path}")

def main(data_path, n_splits=5, evaluate_basic=True, evaluate_advanced=True, evaluate_location_models=True):
    """Run the cross-validation process."""
    logger.info("Starting cross-validation...")
    
    # Load data
    df = load_data(data_path)
    
    # Generate advanced features if needed
    if evaluate_advanced:
        logger.info("Generating advanced features...")
        df_advanced = engineer_advanced_features(df)
        logger.info(f"Advanced features shape: {df_advanced.shape}")
    
    # Initialize results
    global_basic_results = None
    global_advanced_results = None
    loc_basic_results = None
    loc_advanced_results = None
    
    # Cross-validate global model with basic features
    if evaluate_basic:
        logger.info("Cross-validating global model with basic features...")
        global_basic_results = cross_validate_global_model(df, "basic_features", n_splits)
    
    # Cross-validate global model with advanced features
    if evaluate_advanced:
        logger.info("Cross-validating global model with advanced features...")
        global_advanced_results = cross_validate_global_model(df_advanced, "advanced_features", n_splits)
    
    # Cross-validate location-specific models with basic features
    if evaluate_location_models and evaluate_basic:
        logger.info("Cross-validating location-specific models with basic features...")
        loc_basic_results = cross_validate_location_models(df, "basic_features", n_splits)
    
    # Cross-validate location-specific models with advanced features
    if evaluate_location_models and evaluate_advanced:
        logger.info("Cross-validating location-specific models with advanced features...")
        loc_advanced_results = cross_validate_location_models(df_advanced, "advanced_features", n_splits)
    
    # Compare global models
    if evaluate_basic and evaluate_advanced:
        compare_global_models(global_basic_results, global_advanced_results)
    
    # Compare location-specific models
    if evaluate_location_models and evaluate_basic and evaluate_advanced and loc_basic_results and loc_advanced_results:
        compare_location_models(loc_basic_results, loc_advanced_results)
    
    # Create summary report
    create_summary_report(global_basic_results, global_advanced_results, loc_basic_results, loc_advanced_results)
    
    return {
        'global_basic_results': global_basic_results,
        'global_advanced_results': global_advanced_results,
        'loc_basic_results': loc_basic_results,
        'loc_advanced_results': loc_advanced_results
    }

def compare_global_models(basic_results, advanced_results):
    """Compare global models with basic and advanced features."""
    logger.info("Creating global model comparison visualizations...")
    
    # Create metrics comparison
    metrics = ['rmse', 'r2', 'mae']
    metric_labels = ['RMSE', 'R²', 'MAE']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric in enumerate(metrics):
        basic_values = [fold[metric] for fold in basic_results['fold_metrics']]
        advanced_values = [fold[metric] for fold in advanced_results['fold_metrics']]
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Fold': list(range(1, len(basic_values) + 1)) * 2,
            'Feature Set': ['Basic'] * len(basic_values) + ['Advanced'] * len(advanced_values),
            metric: basic_values + advanced_values
        })
        
        # Plot
        sns.barplot(x='Fold', y=metric, hue='Feature Set', data=df, ax=axes[i])
        axes[i].set_title(f'{metric_labels[i]} by Fold')
        axes[i].set_ylabel(metric_labels[i])
        axes[i].grid(True, alpha=0.3)
        
        # Add overall values as horizontal lines
        axes[i].axhline(y=basic_results['overall_metrics'][metric], color='blue', linestyle='--', 
                        label=f'Basic Overall: {basic_results["overall_metrics"][metric]:.4f}')
        axes[i].axhline(y=advanced_results['overall_metrics'][metric], color='orange', linestyle='--',
                        label=f'Advanced Overall: {advanced_results["overall_metrics"][metric]:.4f}')
        
        # Add legend
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create actual vs predicted plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Basic features
    axes[0].scatter(basic_results['all_y_true'], basic_results['all_y_pred'], alpha=0.5)
    axes[0].plot([min(basic_results['all_y_true']), max(basic_results['all_y_true'])], 
                [min(basic_results['all_y_true']), max(basic_results['all_y_true'])], 'r--')
    axes[0].set_xlabel('Actual Occupancy')
    axes[0].set_ylabel('Predicted Occupancy')
    axes[0].set_title(f'Basic Features - R²: {basic_results["overall_metrics"]["r2"]:.4f}')
    axes[0].grid(True, alpha=0.3)
    
    # Advanced features
    axes[1].scatter(advanced_results['all_y_true'], advanced_results['all_y_pred'], alpha=0.5)
    axes[1].plot([min(advanced_results['all_y_true']), max(advanced_results['all_y_true'])], 
                [min(advanced_results['all_y_true']), max(advanced_results['all_y_true'])], 'r--')
    axes[1].set_xlabel('Actual Occupancy')
    axes[1].set_ylabel('Predicted Occupancy')
    axes[1].set_title(f'Advanced Features - R²: {advanced_results["overall_metrics"]["r2"]:.4f}')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_actual_vs_predicted.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create error distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Basic features
    basic_errors = np.array(basic_results['all_y_true']) - np.array(basic_results['all_y_pred'])
    sns.histplot(basic_errors, kde=True, ax=axes[0])
    axes[0].set_xlabel('Prediction Error')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Basic Features - RMSE: {basic_results["overall_metrics"]["rmse"]:.4f}')
    axes[0].grid(True, alpha=0.3)
    
    # Advanced features
    advanced_errors = np.array(advanced_results['all_y_true']) - np.array(advanced_results['all_y_pred'])
    sns.histplot(advanced_errors, kde=True, ax=axes[1])
    axes[1].set_xlabel('Prediction Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Advanced Features - RMSE: {advanced_results["overall_metrics"]["rmse"]:.4f}')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_error_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()

def compare_location_models(basic_results, advanced_results):
    """Compare location-specific models with basic and advanced features."""
    logger.info("Creating location model comparison visualizations...")
    
    # Get common locations
    common_locations = set(basic_results.keys()) & set(advanced_results.keys())
    
    if not common_locations:
        logger.warning("No common locations to compare")
        return
    
    # Create metrics comparison
    metrics = ['rmse', 'r2', 'mae']
    metric_labels = ['RMSE', 'R²', 'MAE']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric in enumerate(metrics):
        # Collect metrics for each location
        locations = []
        basic_values = []
        advanced_values = []
        
        for loc in common_locations:
            locations.append(loc)
            basic_values.append(basic_results[loc][f'overall_{metric}'])
            advanced_values.append(advanced_results[loc][f'overall_{metric}'])
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Location': locations * 2,
            'Feature Set': ['Basic'] * len(locations) + ['Advanced'] * len(locations),
            metric: basic_values + advanced_values
        })
        
        # Plot
        sns.barplot(x='Location', y=metric, hue='Feature Set', data=df, ax=axes[i])
        axes[i].set_title(f'{metric_labels[i]} by Location')
        axes[i].set_ylabel(metric_labels[i])
        axes[i].grid(True, alpha=0.3)
        
        # Add average values as horizontal lines
        basic_avg = np.mean(basic_values)
        adv_avg = np.mean(advanced_values)
        axes[i].axhline(y=basic_avg, color='blue', linestyle='--', 
                        label=f'Basic Avg: {basic_avg:.4f}')
        axes[i].axhline(y=adv_avg, color='orange', linestyle='--',
                        label=f'Advanced Avg: {adv_avg:.4f}')
        
        # Add legend
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "location_model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create R² vs data size plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect data for each location
    locations = []
    data_sizes = []
    basic_r2 = []
    advanced_r2 = []
    
    for loc in common_locations:
        locations.append(loc)
        # Estimate data size from fold metrics
        data_size = sum(m['n_samples'] for m in basic_results[loc]['fold_metrics'])
        data_sizes.append(data_size)
        basic_r2.append(basic_results[loc]['overall_r2'])
        advanced_r2.append(advanced_results[loc]['overall_r2'])
    
    # Plot
    plt.scatter(data_sizes, basic_r2, label='Basic Features', s=100, marker='o')
    plt.scatter(data_sizes, advanced_r2, label='Advanced Features', s=100, marker='x')
    
    # Add location labels
    for i, loc in enumerate(locations):
        plt.annotate(loc, (data_sizes[i], basic_r2[i]), xytext=(5, 5), textcoords='offset points')
        plt.annotate(loc, (data_sizes[i], advanced_r2[i]), xytext=(5, -10), textcoords='offset points')
    
    # Connect basic and advanced points for the same location
    for i in range(len(locations)):
        plt.plot([data_sizes[i], data_sizes[i]], [basic_r2[i], advanced_r2[i]], 'k--', alpha=0.3)
    
    plt.xlabel('Number of Data Points')
    plt.ylabel('R²')
    plt.title('R² vs Data Size by Location and Feature Set')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "location_r2_vs_data_size.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform time series cross-validation on parking occupancy models")
    parser.add_argument("--data", default="data/prepared_data_improved.csv", help="Path to the prepared data file")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of cross-validation splits")
    parser.add_argument("--basic_only", action="store_true", help="Only evaluate basic features")
    parser.add_argument("--advanced_only", action="store_true", help="Only evaluate advanced features")
    parser.add_argument("--global_only", action="store_true", help="Only evaluate global models")
    
    args = parser.parse_args()
    
    # Determine which models to evaluate
    evaluate_basic = not args.advanced_only
    evaluate_advanced = not args.basic_only
    evaluate_location_models = not args.global_only
    
    main(args.data, args.n_splits, evaluate_basic, evaluate_advanced, evaluate_location_models)