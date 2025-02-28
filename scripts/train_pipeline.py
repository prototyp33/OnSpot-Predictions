import argparse
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import ColumnTransformer, StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_location_models_func(df, feature_set_name, output_dir):
    """Train separate models for each location."""
    logger.info(f"Training location-specific models with {feature_set_name}...")
    
    # Check if location_id exists
    if 'location_id' not in df.columns:
        logger.warning("No location_id column found. Skipping location-specific models.")
        return {}
    
    # Get unique locations
    locations = df['location_id'].unique()
    logger.info(f"Training models for {len(locations)} locations")
    
    location_models = {}
    
    # Exclude non-feature columns
    exclude_cols = ['timestamp', 'date', 'occupancy', 'location_id']
    
    for loc in locations:
        logger.info(f"Training model for location {loc}")
        
        # Filter data for this location
        loc_df = df[df['location_id'] == loc]
        
        # Skip if not enough data
        if len(loc_df) < 50:
            logger.warning(f"Skipping location {loc} - not enough data ({len(loc_df)} rows)")
            continue
        
        # Prepare features and target
        X = loc_df.drop(columns=[col for col in exclude_cols if col in loc_df.columns])
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
        
        # Create pipeline with preprocessing and model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
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
    
    args = parser.parse_args()
    
    # Train models
    results = train_models(
        args.data, 
        args.output, 
        use_advanced_features=args.advanced, 
        train_location_models=args.location_models
    )
    
    # Create summary
    create_model_summary(
        results['global_model'],
        results['location_models'],
        results['feature_set'],
        args.output
    ) 