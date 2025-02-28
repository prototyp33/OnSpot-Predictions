#!/usr/bin/env python
"""
Script for hyperparameter tuning with time series cross-validation.
Finds the optimal hyperparameters for parking occupancy prediction models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os
import logging
import argparse
from datetime import datetime
import joblib
import sys
from scipy.stats import randint, uniform

# Add the project root to the Python path to allow imports from scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from your existing modules
from scripts.parking_sim.advanced_features import engineer_advanced_features

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory
output_dir = "hyperparameter_tuning_results"
os.makedirs(output_dir, exist_ok=True)

def load_data(file_path):
    """Load and prepare the dataset for hyperparameter tuning."""
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

def tune_global_model(df, feature_set_name, n_splits=5, n_iter=50, model_type='gbm'):
    """Tune hyperparameters for a global model using time series cross-validation."""
    logger.info(f"Tuning hyperparameters for global model with {feature_set_name} features...")
    
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
    
    # Create base pipeline with preprocessing
    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])
    
    # Set up model and parameter grid based on model type
    if model_type == 'gbm':
        pipeline.steps.append(('model', GradientBoostingRegressor()))
        param_grid = {
            'model__n_estimators': randint(50, 500),
            'model__learning_rate': uniform(0.01, 0.3),
            'model__max_depth': randint(3, 10),
            'model__min_samples_split': randint(2, 20),
            'model__min_samples_leaf': randint(1, 10),
            'model__subsample': uniform(0.5, 0.5),  # 0.5 to 1.0
            'model__max_features': ['sqrt', 'log2', None]
        }
    elif model_type == 'rf':
        pipeline.steps.append(('model', RandomForestRegressor()))
        param_grid = {
            'model__n_estimators': randint(50, 500),
            'model__max_depth': randint(3, 20),
            'model__min_samples_split': randint(2, 20),
            'model__min_samples_leaf': randint(1, 10),
            'model__max_features': ['sqrt', 'log2', None]
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Set up time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Create custom scorer for RMSE
    rmse_scorer = make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)
    
    # Set up randomized search
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=tscv,
        scoring=rmse_scorer,
        n_jobs=-1,
        verbose=2,
        random_state=42,
        return_train_score=True
    )
    
    # Fit randomized search
    logger.info("Starting hyperparameter search...")
    random_search.fit(X, y)
    
    # Get best parameters and score
    best_params = random_search.best_params_
    best_score = -random_search.best_score_  # Convert back to RMSE
    
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best RMSE: {best_score:.4f}")
    
    # Save best model
    best_model = random_search.best_estimator_
    model_path = os.path.join(output_dir, f"global_{model_type}_{feature_set_name}_tuned.pkl")
    joblib.dump(best_model, model_path)
    logger.info(f"Best model saved to {model_path}")
    
    # Save results
    results_df = pd.DataFrame(random_search.cv_results_)
    results_path = os.path.join(output_dir, f"global_{model_type}_{feature_set_name}_tuning_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Visualize results
    visualize_tuning_results(random_search, feature_set_name, model_type, "global")
    
    return {
        'best_model': best_model,
        'best_params': best_params,
        'best_score': best_score,
        'cv_results': random_search.cv_results_
    }

def tune_location_model(df, location_id, feature_set_name, n_splits=5, n_iter=50, model_type='gbm'):
    """Tune hyperparameters for a location-specific model using time series cross-validation."""
    logger.info(f"Tuning hyperparameters for location {location_id} with {feature_set_name} features...")
    
    # Filter data for this location
    loc_df = df[df['location_id'] == location_id].copy()
    
    # Skip if not enough data
    if len(loc_df) < 100:  # Minimum threshold for reliable tuning
        logger.warning(f"Skipping location {location_id} - not enough data ({len(loc_df)} rows)")
        return None
    
    # Prepare features and target
    exclude_cols = ['timestamp', 'date', 'occupancy', 'location_id']
    X = loc_df.drop(columns=[col for col in exclude_cols if col in loc_df.columns])
    y = loc_df['occupancy']
    
    # Identify column types - make sure to exclude location_id
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
    
    # Create base pipeline with preprocessing
    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])
    
    # Set up model and parameter grid based on model type
    if model_type == 'gbm':
        pipeline.steps.append(('model', GradientBoostingRegressor()))
        param_grid = {
            'model__n_estimators': randint(50, 500),
            'model__learning_rate': uniform(0.01, 0.3),
            'model__max_depth': randint(3, 10),
            'model__min_samples_split': randint(2, 20),
            'model__min_samples_leaf': randint(1, 10),
            'model__subsample': uniform(0.5, 0.5),  # 0.5 to 1.0
            'model__max_features': ['sqrt', 'log2', None]
        }
    elif model_type == 'rf':
        pipeline.steps.append(('model', RandomForestRegressor()))
        param_grid = {
            'model__n_estimators': randint(50, 500),
            'model__max_depth': randint(3, 20),
            'model__min_samples_split': randint(2, 20),
            'model__min_samples_leaf': randint(1, 10),
            'model__max_features': ['sqrt', 'log2', None],
            'model__bootstrap': [True, False]
        }
    else:
        logger.error(f"Unsupported model type: {model_type}")
        return None
    
    # Set up time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Create RMSE scorer (negative because RandomizedSearchCV maximizes score)
    rmse_scorer = make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)))
    
    # Set up randomized search
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=tscv,
        scoring=rmse_scorer,
        n_jobs=-1,
        verbose=2,
        random_state=42,
        return_train_score=True,
        error_score=np.nan  # Return NaN for failed fits instead of raising error
    )
    
    # Fit the model
    try:
        random_search.fit(X, y)
        
        # Get best parameters and score
        best_params = random_search.best_params_
        best_score = -random_search.best_score_  # Convert back to positive RMSE
        
        logger.info(f"Best RMSE for location {location_id}: {best_score:.4f}")
        logger.info(f"Best parameters for location {location_id}: {best_params}")
        
        # Save best model
        model_dir = os.path.join(output_dir, "location_models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"location_{location_id}_{model_type}_{feature_set_name}.pkl")
        joblib.dump(random_search.best_estimator_, model_path)
        logger.info(f"Best model for location {location_id} saved to {model_path}")
        
        # Visualize results
        visualize_tuning_results(random_search, feature_set_name, model_type, f"location_{location_id}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_estimator_path': model_path,
            'cv_results': random_search.cv_results_
        }
    
    except Exception as e:
        logger.error(f"Error tuning model for location {location_id}: {e}")
        return None

def visualize_tuning_results(random_search, feature_set_name, model_type, model_scope):
    """Visualize hyperparameter tuning results."""
    logger.info(f"Visualizing tuning results for {model_scope} {model_type} model with {feature_set_name} features...")
    
    # Get results
    results = pd.DataFrame(random_search.cv_results_)
    
    # Create output directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot RMSE vs iteration
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(results) + 1), -results['mean_test_score'], 'o-', label='Test RMSE')
    plt.plot(range(1, len(results) + 1), -results['mean_train_score'], 'o-', label='Train RMSE')
    plt.axhline(y=-random_search.best_score_, color='r', linestyle='--', 
                label=f'Best RMSE: {-random_search.best_score_:.4f}')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title(f'RMSE vs Iteration - {model_scope.title()} {model_type.upper()} Model with {feature_set_name.replace("_", " ").title()} Features')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"{model_scope}_{model_type}_{feature_set_name}_rmse_vs_iteration.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Extract key hyperparameters for visualization
    key_params = []
    
    if model_type == 'gbm':
        key_params = ['model__n_estimators', 'model__learning_rate', 'model__max_depth', 'model__subsample']
    elif model_type == 'rf':
        key_params = ['model__n_estimators', 'model__max_depth', 'model__min_samples_split']
    
    # Plot RMSE vs key hyperparameters
    for param in key_params:
        if param in results.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(results[param], -results['mean_test_score'], alpha=0.7)
            plt.xlabel(param)
            plt.ylabel('RMSE')
            plt.title(f'RMSE vs {param} - {model_scope.title()} {model_type.upper()} Model')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"{model_scope}_{model_type}_{feature_set_name}_rmse_vs_{param.split('__')[1]}.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create parameter importance plot
    if len(results) > 10:  # Only if we have enough samples
        try:
            from sklearn.inspection import permutation_importance
            
            # Get best model
            best_model = random_search.best_estimator_
            
            # Prepare data
            exclude_cols = ['timestamp', 'date', 'occupancy']
            if 'location_id' in random_search.best_estimator_.feature_names_in_:
                exclude_cols.append('location_id')
            
            X = pd.DataFrame(random_search.X_)
            y = random_search.y_
            
            # Calculate permutation importance
            perm_importance = permutation_importance(best_model, X, y, n_repeats=10, random_state=42)
            
            # Create DataFrame for plotting
            importance_df = pd.DataFrame({
                'Feature': best_model.feature_names_in_,
                'Importance': perm_importance.importances_mean
            }).sort_values('Importance', ascending=False)
            
            # Plot top 20 features
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
            plt.title(f'Feature Importance - {model_scope.title()} {model_type.upper()} Model with {feature_set_name.replace("_", " ").title()} Features')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"{model_scope}_{model_type}_{feature_set_name}_feature_importance.png"), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Could not create feature importance plot: {e}")

def main(data_path, n_splits=5, n_iter=50, model_types=None, feature_sets=None, tune_location_models=True):
    """Main function to run hyperparameter tuning."""
    logger.info("Starting hyperparameter tuning...")
    
    # Set default model types and feature sets if not provided
    if model_types is None:
        model_types = ['gbm', 'rf']
    
    if feature_sets is None:
        feature_sets = ['basic', 'advanced']
    
    # Load data
    df = load_data(data_path)
    
    # Prepare feature sets
    feature_dfs = {}
    
    if 'basic' in feature_sets:
        feature_dfs['basic'] = df.copy()
    
    if 'advanced' in feature_sets:
        logger.info("Generating advanced features...")
        feature_dfs['advanced'] = engineer_advanced_features(df)
        logger.info(f"Advanced features shape: {feature_dfs['advanced'].shape}")
    
    # Tune global models
    global_results = {}
    
    for feature_set in feature_sets:
        feature_df = feature_dfs[feature_set]
        
        for model_type in model_types:
            logger.info(f"Tuning global {model_type} model with {feature_set} features...")
            result = tune_global_model(feature_df, feature_set, n_splits, n_iter, model_type)
            global_results[f"{model_type}_{feature_set}"] = result
    
    # Tune location-specific models if requested
    location_results = {}
    
    if tune_location_models and 'location_id' in df.columns:
        locations = df['location_id'].unique()
        logger.info(f"Found {len(locations)} unique locations")
        
        for feature_set in feature_sets:
            feature_df = feature_dfs[feature_set]
            
            for model_type in model_types:
                location_results[f"{model_type}_{feature_set}"] = {}
                
                for location in locations:
                    logger.info(f"Tuning {model_type} model for location {location} with {feature_set} features...")
                    result = tune_location_model(feature_df, location, feature_set, n_splits, n_iter, model_type)
                    
                    if result is not None:
                        location_results[f"{model_type}_{feature_set}"][location] = result
    
    # Create summary report
    create_summary_report(global_results, location_results, output_dir)
    
    logger.info("Hyperparameter tuning completed successfully")
    return global_results, location_results

def create_summary_report(global_results, location_results, output_dir):
    """Create a summary report of hyperparameter tuning results."""
    logger.info("Creating summary report...")
    
    summary_path = os.path.join(output_dir, "tuning_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("=== HYPERPARAMETER TUNING SUMMARY ===\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("GLOBAL MODELS:\n")
        f.write("-------------\n\n")
        
        # Sort global models by performance
        sorted_global = []
        for model_key, results in global_results.items():
            sorted_global.append((model_key, results))
        
        sorted_global = sorted(sorted_global, key=lambda x: x[1]['best_score'])
        
        for model_key, results in sorted_global:
            model_type, feature_set = model_key.split('_', 1)
            f.write(f"{model_type.upper()} with {feature_set.replace('_', ' ')} features:\n")
            f.write(f"  RMSE: {results['best_score']:.4f}\n")
            f.write("  Best parameters:\n")
            
            for param, value in results['best_params'].items():
                f.write(f"    - {param}: {value}\n")
            
            f.write("\n")
        
        # Find best global model
        best_global_key = sorted_global[0][0]
        best_global_rmse = sorted_global[0][1]['best_score']
        best_global_type, best_global_features = best_global_key.split('_', 1)
        
        f.write(f"Best global model: {best_global_type.upper()} with {best_global_features.replace('_', ' ')} features (RMSE: {best_global_rmse:.4f})\n\n")
        
        # Location-specific models
        if location_results:
            # Check if any location models were successfully tuned
            any_successful_models = False
            for model_key in location_results:
                if location_results[model_key] and any(location_results[model_key].values()):
                    any_successful_models = True
                    break
            
            if any_successful_models:
                f.write("LOCATION-SPECIFIC MODELS:\n")
                f.write("------------------------\n\n")
                
                # Calculate average performance by model type and feature set
                avg_performance = {}
                
                for model_key in location_results:
                    if location_results[model_key]:  # Check if there are any results for this model type
                        valid_results = [result for result in location_results[model_key].values() if result is not None]
                        if valid_results:
                            scores = [result['best_score'] for result in valid_results]
                            avg_performance[model_key] = sum(scores) / len(scores)
                
                # Sort by average performance
                sorted_loc_models = sorted(avg_performance.items(), key=lambda x: x[1])
                
                for model_key, avg_rmse in sorted_loc_models:
                    model_type, feature_set = model_key.split('_', 1)
                    f.write(f"{model_type.upper()} with {feature_set.replace('_', ' ')} features:\n")
                    f.write(f"  Average RMSE: {avg_rmse:.4f}\n")
                    
                    # Get valid results for this model type
                    valid_results = {loc: res for loc, res in location_results[model_key].items() if res is not None}
                    
                    if valid_results:
                        # Best location for this model
                        best_loc = min(valid_results.items(), key=lambda x: x[1]['best_score'])
                        f.write(f"  Best location: {best_loc[0]} (RMSE: {best_loc[1]['best_score']:.4f})\n")
                        
                        # Worst location for this model
                        worst_loc = max(valid_results.items(), key=lambda x: x[1]['best_score'])
                        f.write(f"  Worst location: {worst_loc[0]} (RMSE: {worst_loc[1]['best_score']:.4f})\n\n")
                
                # Find best location-specific model
                if sorted_loc_models:
                    best_loc_model_key = sorted_loc_models[0][0]
                    best_loc_avg_rmse = sorted_loc_models[0][1]
                    best_loc_type, best_loc_features = best_loc_model_key.split('_', 1)
                    
                    f.write(f"Best location-specific model: {best_loc_type.upper()} with {best_loc_features.replace('_', ' ')} features (Avg RMSE: {best_loc_avg_rmse:.4f})\n\n")
                
                    # Overall recommendation
                    f.write("RECOMMENDATION:\n")
                    f.write("--------------\n\n")
                    
                    if best_loc_avg_rmse < best_global_rmse:
                        improvement = (best_global_rmse - best_loc_avg_rmse) / best_global_rmse * 100
                        
                        f.write(f"Use location-specific {best_loc_type.upper()} models with {best_loc_features.replace('_', ' ')} features.\n")
                        f.write(f"This approach provides {improvement:.1f}% better performance than the best global model.\n")
                    else:
                        f.write(f"Use a global {best_global_type.upper()} model with {best_global_features.replace('_', ' ')} features.\n")
                        f.write("Location-specific models do not provide significant improvement over the global model.\n")
            else:
                f.write("LOCATION-SPECIFIC MODELS:\n")
                f.write("------------------------\n\n")
                f.write("No successful location-specific models were tuned.\n\n")
                
                f.write("RECOMMENDATION:\n")
                f.write("--------------\n\n")
                f.write(f"Use a global {best_global_type.upper()} model with {best_global_features.replace('_', ' ')} features.\n")
        else:
            f.write("LOCATION-SPECIFIC MODELS:\n")
            f.write("------------------------\n\n")
            f.write("Location-specific models were not evaluated.\n\n")
            
            f.write("RECOMMENDATION:\n")
            f.write("--------------\n\n")
            f.write(f"Use a global {best_global_type.upper()} model with {best_global_features.replace('_', ' ')} features.\n")
    
    logger.info(f"Summary report saved to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune hyperparameters for parking occupancy prediction models")
    parser.add_argument("--data", default="data/prepared_data_improved.csv", help="Path to the prepared data file")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of cross-validation splits")
    parser.add_argument("--n_iter", type=int, default=50, help="Number of parameter settings to try")
    parser.add_argument("--model_types", nargs='+', choices=['gbm', 'rf'], default=['gbm', 'rf'], help="Model types to tune")
    parser.add_argument("--feature_sets", nargs='+', choices=['basic', 'advanced'], default=['basic', 'advanced'], help="Feature sets to use")
    parser.add_argument("--global_only", action="store_true", help="Only tune global models")
    
    args = parser.parse_args()
    
    main(
        args.data, 
        args.n_splits, 
        args.n_iter, 
        args.model_types, 
        args.feature_sets, 
        not args.global_only
    )