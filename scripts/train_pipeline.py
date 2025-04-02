import argparse
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import logging
import json

# Configure logging (assuming logger is configured elsewhere or add basic config)
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO) # Uncomment if basic config needed

def train_location_models_func(df, feature_set_name, output_dir, best_params_path=None):
    """Train separate models for each location using time-based split and potentially tuned hyperparameters."""
    logger.info(f"Training location-specific models with {feature_set_name} features...")
    
    # Load best hyperparameters if path is provided
    all_best_params = {}
    if best_params_path:
        try:
            with open(best_params_path, 'r') as f:
                all_best_params = json.load(f)
            logger.info(f"Successfully loaded best hyperparameters from {best_params_path}")
        except FileNotFoundError:
            logger.warning(f"Hyperparameter file not found: {best_params_path}. Using default parameters.")
        except Exception as e:
            logger.error(f"Error loading hyperparameters from {best_params_path}: {e}. Using default parameters.")

    # Check if location_id exists
    if 'location_id' not in df.columns:
        logger.warning("No location_id column found. Skipping location-specific models.")
        return {}
    
    # Check if timestamp exists for sorting
    if 'timestamp' not in df.columns:
        logger.error("Timestamp column ('timestamp') not found. Cannot perform time-based split.")
        return {}
    
    # Get unique locations
    locations = df['location_id'].unique()
    logger.info(f"Training models for {len(locations)} locations")
    
    location_models = {}
    
    # Exclude non-feature columns
    exclude_cols_final = ['timestamp', 'date', 'occupancy', 'location_id']
    
    for loc in locations:
        logger.info(f"Training model for location {loc}")
        
        # Filter data for this location
        loc_df = df[df['location_id'] == loc].copy()
        loc_df.sort_values('timestamp', inplace=True)
        
        # Skip if not enough data
        if len(loc_df) < 50: # Minimum data check remains
            logger.warning(f"Skipping location {loc} - not enough data ({len(loc_df)} rows)")
            continue
        
        # Prepare features and target *after* sorting
        X = loc_df.drop(columns=[col for col in exclude_cols_final if col in loc_df.columns])
        y = loc_df['occupancy']
        
        # Identify numeric and categorical columns
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
        
        # --- Get Tuned Hyperparameters --- 
        model_params = {'random_state': 42} # Start with default random_state
        loc_str = str(loc) # Ensure location ID is string for JSON key lookup
        if loc_str in all_best_params:
            logger.info(f"Using tuned hyperparameters for location {loc}")
            # Strip 'model__' prefix and update model_params
            tuned_params = {k.replace('model__', ''): v for k, v in all_best_params[loc_str].items()}
            model_params.update(tuned_params)
        else:
            logger.warning(f"No tuned hyperparameters found for location {loc}. Using defaults.")
            # Use default GBR parameters (implicitly handled by constructor)
            pass
        # ----------------------------------
        
        # Create pipeline with preprocessing and model (using potentially tuned params)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(**model_params)) # Instantiate with params
        ])
        
        # --- Time-based split ---
        split_index = int(len(X) * 0.8) # 80% for training
        
        if split_index < 1 or split_index >= len(X):
            logger.warning(f"Cannot create a valid train/test split for location {loc} with {len(X)} rows. Skipping.")
            continue
        
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        
        logger.info(f"Location {loc}: Train size={len(X_train)}, Test size={len(X_test)}")
        # ------------------------
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        logger.info(f"Location {loc} model - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
        
        # Save model
        model_path = os.path.join(output_dir, f"location_{loc}_model_{feature_set_name}.pkl")
        joblib.dump(pipeline, model_path)
        
        # Store model in dictionary
        location_models[loc] = {
            'model': pipeline,
            'metrics': {
                'rmse': rmse,
                'r2': r2,
                'mae': mae
            }
        }
    
    return location_models

def create_model_summary(global_model, location_models, feature_set, output_dir):
    """Create a summary of model performance."""
    logger.info("Creating model summary...")
    
    summary_path = os.path.join(output_dir, "model_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("=== MODEL PERFORMANCE SUMMARY ===\n\n")
        f.write(f"Feature Set: {feature_set}\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("GLOBAL MODEL:\n")
        f.write("--------------\n")
        if hasattr(global_model, 'metrics'):
            f.write(f"RMSE: {global_model['metrics']['rmse']:.4f}\n")
            f.write(f"R²: {global_model['metrics']['r2']:.4f}\n")
            f.write(f"MAE: {global_model['metrics']['mae']:.4f}\n\n")
        
        if location_models:
            f.write("LOCATION-SPECIFIC MODELS:\n")
            f.write("------------------------\n")
            
            # Calculate average metrics
            avg_rmse = np.mean([m['metrics']['rmse'] for m in location_models.values()])
            avg_r2 = np.mean([m['metrics']['r2'] for m in location_models.values()])
            avg_mae = np.mean([m['metrics']['mae'] for m in location_models.values()])
            
            f.write(f"Average RMSE: {avg_rmse:.4f}\n")
            f.write(f"Average R²: {avg_r2:.4f}\n")
            f.write(f"Average MAE: {avg_mae:.4f}\n\n")
            
            # Best performing locations
            best_loc = sorted(location_models.items(), key=lambda x: x[1]['metrics']['r2'], reverse=True)[:3]
            f.write("Top 3 best-performing locations:\n")
            for loc, model_info in best_loc:
                f.write(f"- Location {loc}: R² = {model_info['metrics']['r2']:.4f}, RMSE = {model_info['metrics']['rmse']:.4f}\n")
            
            # Worst performing locations
            worst_loc = sorted(location_models.items(), key=lambda x: x[1]['metrics']['r2'])[:3]
            f.write("\nTop 3 worst-performing locations:\n")
            for loc, model_info in worst_loc:
                f.write(f"- Location {loc}: R² = {model_info['metrics']['r2']:.4f}, RMSE = {model_info['metrics']['rmse']:.4f}\n")
    
    logger.info(f"Model summary saved to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train parking occupancy prediction models")
    parser.add_argument("--data", default="data/prepared_data_improved.csv", help="Path to the prepared data file")
    parser.add_argument("--output", default="trained_models", help="Output directory for trained models")
    parser.add_argument("--advanced", action="store_true", help="Use advanced features")
    parser.add_argument("--location_models", action="store_true", help="Train location-specific models")
    parser.add_argument("--params_file", default=None, help="Path to JSON file containing best hyperparameters")

    args = parser.parse_args()

    # --- Modified execution logic ---
    logger.info(f"Loading data from {args.data}")
    try:
        # Assuming timestamp needs parsing
        df = pd.read_csv(args.data, parse_dates=['timestamp'])
    except FileNotFoundError:
        logger.error(f"Data file not found: {args.data}")
        exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Determine feature set name (simple example)
    feature_set_name = "advanced" if args.advanced else "standard"

    location_models_results = {}
    global_model_results = None # Placeholder for potential future global model

    if args.location_models:
        logger.info("Starting location-specific model training...")
        location_models_results = train_location_models_func(
            df,
            feature_set_name,
            args.output,
            best_params_path=args.params_file
        )
        logger.info("Location-specific model training finished.")
    else:
        # Placeholder: Add logic for global model training if needed
        logger.info("Location-specific training not requested. Add global model logic here if desired.")
        # Example: global_model_results = train_global_model_func(df, feature_set_name, args.output)

    # Create summary using the results we have
    logger.info("Creating final summary...")
    create_model_summary(
        global_model_results, # Pass None if no global model trained
        location_models_results,
        feature_set_name, # Use the determined feature set name
        args.output
    )
    logger.info("Script finished successfully.")
    # ----------------------------- 