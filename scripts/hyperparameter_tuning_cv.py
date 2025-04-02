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
    logger.info(f"Tuning hyperparameters for location {location_id} with {feature_set_name} features using {model_type}...")

    # Filter data for this location and sort by time
    loc_df = df[df['location_id'] == location_id].copy()
    if 'timestamp' in loc_df.columns:
        loc_df.sort_values('timestamp', inplace=True)
    else:
        logger.warning(f"Timestamp column not found for location {location_id}, cannot guarantee temporal order for TS-Split.")

    # Skip if not enough data for the specified number of splits
    if len(loc_df) < n_splits + 1: # TimeSeriesSplit requires at least n_splits + 1 samples
        logger.warning(f"Skipping location {location_id} - not enough data ({len(loc_df)} rows) for {n_splits} splits.")
        return None

    # Prepare features and target
    exclude_cols = ['timestamp', 'date', 'occupancy', 'location_id'] # Ensure location_id is excluded
    X = loc_df.drop(columns=[col for col in exclude_cols if col in loc_df.columns])
    y = loc_df['occupancy']

    # Identify column types - IMPORTANT: Ensure this matches train_pipeline.py
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    logger.info(f"Location {location_id}: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical features.")
    if not numeric_cols and not categorical_cols:
        logger.error(f"Location {location_id}: No features identified after exclusions. Skipping.")
        return None

    # Create preprocessing pipeline - MUST MATCH train_pipeline.py
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='drop' # Ensure this matches train_pipeline.py
    )

    # Create base pipeline with preprocessing
    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])

    # Set up model and parameter grid based on model type
    if model_type == 'gbm':
        pipeline.steps.append(('model', GradientBoostingRegressor(random_state=42)))
        # Updated parameter grid for GBM
        param_grid = {
            'model__n_estimators': randint(100, 1000),
            'model__learning_rate': uniform(0.01, 0.2), # loc=0.01, scale=0.2
            'model__max_depth': randint(3, 8),
            'model__min_samples_split': randint(2, 50),
            'model__min_samples_leaf': randint(1, 50),
            'model__subsample': uniform(0.6, 0.4),  # loc=0.6, scale=0.4 (range 0.6 to 1.0)
            # 'model__max_features': ['sqrt', 'log2', None] # Keep simple or use uniform(0.5, 0.5)
        }
    elif model_type == 'rf':
        pipeline.steps.append(('model', RandomForestRegressor(random_state=42)))
        # Example grid for RandomForest (can be adjusted)
        param_grid = {
            'model__n_estimators': randint(100, 1000),
            'model__max_depth': randint(5, 30),
            'model__min_samples_split': randint(2, 50),
            'model__min_samples_leaf': randint(1, 50),
            'model__max_features': ['sqrt', 'log2', None]
        }
    else:
        logger.error(f"Unsupported model type: {model_type} for location {location_id}")
        return None

    # Set up time series cross-validation
    # Increase gap if needed, e.g., gap = prediction_horizon
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=0)

    # Set up randomized search
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=tscv,
        scoring='r2', # Use R2 score (higher is better)
        n_jobs=-1,
        verbose=1, # Reduce verbosity a bit
        random_state=42,
        return_train_score=True
    )

    # Fit randomized search
    logger.info(f"Starting hyperparameter search for location {location_id} ({n_iter} iterations)...")
    try:
        random_search.fit(X, y)
    except Exception as e:
        logger.error(f"RandomizedSearchCV failed for location {location_id}: {e}")
        # Log feature names on error for debugging
        logger.error(f"Features used: {X.columns.tolist()}")
        logger.error(f"Numeric features: {numeric_cols}")
        logger.error(f"Categorical features: {categorical_cols}")
        return None

    # Get best parameters and score
    best_params = random_search.best_params_
    best_score = random_search.best_score_ # R2 score

    logger.info(f"Location {location_id}: Best R² score = {best_score:.4f}")
    logger.info(f"Location {location_id}: Best parameters = {best_params}")

    # --- Optional: Refit best model on full location data (use with caution) ---
    # logger.info(f"Refitting best model on full data for location {location_id}")
    # best_model = random_search.best_estimator_
    # best_model.fit(X, y) # This fits on the entire loc_df
    # ----------------------------------------------------------------------------
    best_model = random_search.best_estimator_ # Use the one fitted during search

    # Save best model for this location
    model_filename = f"location_{location_id}_{model_type}_{feature_set_name}_tuned.pkl"
    model_path = os.path.join(output_dir, model_filename)
    joblib.dump(best_model, model_path)
    logger.info(f"Best model for location {location_id} saved to {model_path}")

    # Save results
    results_df = pd.DataFrame(random_search.cv_results_)
    results_filename = f"location_{location_id}_{model_type}_{feature_set_name}_tuning_results.csv"
    results_path = os.path.join(output_dir, results_filename)
    results_df.to_csv(results_path, index=False)

    # Visualize results
    visualize_tuning_results(random_search, feature_set_name, model_type, f"location_{location_id}")

    return {
        'location_id': location_id,
        'best_model_path': model_path,
        'best_params': best_params,
        'best_score': best_score, # R2 score
        'cv_results_path': results_path
    }

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
    """Main function to orchestrate hyperparameter tuning."""
    logger.info("Starting hyperparameter tuning process...")

    # Validate inputs
    if model_types is None:
        model_types = ['gbm'] # Default to GBM
    if feature_sets is None:
        feature_sets = ['standard'] # Assume a default feature set name if none provided

    # Create overall results directory if not already done
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}. Aborting.")
        return
    except Exception as e:
        logger.error(f"Error loading data: {e}. Aborting.")
        return

    all_location_results = {}
    all_best_params = {} # Dictionary to store best params per location/model

    # --- Location-Specific Tuning ---
    if tune_location_models:
        if 'location_id' not in df.columns:
            logger.error("Cannot perform location-specific tuning: 'location_id' column missing.")
        else:
            locations = df['location_id'].unique()
            logger.info(f"Starting tuning for {len(locations)} locations.")
            for model_type in model_types:
                all_best_params[model_type] = {} # Init dict for this model type
                location_results_for_model = []
                for loc_id in locations:
                    # Assuming only one feature set for now
                    feature_set_name = feature_sets[0]
                    loc_result = tune_location_model(
                        df,
                        location_id=loc_id,
                        feature_set_name=feature_set_name,
                        n_splits=n_splits,
                        n_iter=n_iter,
                        model_type=model_type
                    )
                    if loc_result:
                        location_results_for_model.append(loc_result)
                        all_best_params[model_type][loc_id] = loc_result['best_params']

                all_location_results[model_type] = location_results_for_model
                logger.info(f"Finished tuning for {model_type} across all locations.")

                # Save best parameters to JSON
                best_params_path = os.path.join(output_dir, f"best_params_{model_type}.json")
                try:
                    import json
                    with open(best_params_path, 'w') as f:
                        json.dump(all_best_params[model_type], f, indent=4)
                    logger.info(f"Best parameters for {model_type} saved to {best_params_path}")
                except Exception as e:
                    logger.error(f"Error saving best parameters to JSON: {e}")

    else:
        logger.info("Skipping location-specific model tuning as requested.")

    # --- Global Model Tuning (Placeholder/Optional) ---
    # global_results = {}
    # for model_type in model_types:
    #     for feature_set in feature_sets:
    #         global_results[(model_type, feature_set)] = tune_global_model(
    #             df, feature_set, n_splits, n_iter, model_type)

    # --- Summary Report ---
    # create_summary_report(global_results, all_location_results, output_dir)

    logger.info("Hyperparameter tuning process finished.")

def create_summary_report(global_results, location_results, output_dir):
    """Create a summary report of tuning results."""
    summary_path = os.path.join(output_dir, "tuning_summary_report.txt")
    logger.info(f"Creating summary report at {summary_path}")

    with open(summary_path, 'w') as f:
        f.write("=== HYPERPARAMETER TUNING SUMMARY ===\n\n")
        f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summarize location-specific results
        if location_results:
            f.write("Location-Specific Model Tuning Results:\n")
            f.write("-------------------------------------\n")
            for model_type, results_list in location_results.items():
                f.write(f"Model Type: {model_type.upper()}\n")
                if not results_list:
                    f.write("- No successful tuning runs.\n")
                    continue

                avg_score = np.mean([r['best_score'] for r in results_list])
                f.write(f"- Average Best Score (R²): {avg_score:.4f}\n")
                f.write("  Best Parameters per Location (example):\n")
                # Show params for the best performing location
                best_loc = max(results_list, key=lambda x: x['best_score'])
                f.write(f"  - Location {best_loc['location_id']}: Score={best_loc['best_score']:.4f}\n")
                params_str = "\n".join([f"      {k}: {v}" for k, v in best_loc['best_params'].items()])
                f.write(f"{params_str}\n")
            f.write("\n")

        # Summarize global results (if implemented)
        # ... (add similar summary for global_results if needed)

    logger.info("Summary report created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for parking occupancy models.")
    parser.add_argument("--data", required=True, help="Path to the prepared data file (CSV).")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits for TimeSeriesSplit.")
    parser.add_argument("--n_iter", type=int, default=50, help="Number of iterations for RandomizedSearchCV.")
    parser.add_argument("--model_types", nargs='+', default=['gbm'], help="Model types to tune (e.g., gbm rf).")
    # parser.add_argument("--feature_sets", nargs='+', default=['standard'], help="Feature set names used.") # Simplified for now
    parser.add_argument("--skip_location_tuning", action="store_true", help="Skip tuning location-specific models.")

    args = parser.parse_args()

    main(
        data_path=args.data,
        n_splits=args.n_splits,
        n_iter=args.n_iter,
        model_types=args.model_types,
        # feature_sets=args.feature_sets, # Simplified for now
        feature_sets=['standard'], # Hardcode default name for simplicity
        tune_location_models=(not args.skip_location_tuning)
    )