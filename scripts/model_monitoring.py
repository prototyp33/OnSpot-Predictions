#!/usr/bin/env python
"""
Script for monitoring model performance over time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import logging
import argparse
from datetime import datetime, timedelta
import sys
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from your existing modules
from scripts.parking_sim.advanced_features import engineer_advanced_features

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def monitor_model_performance(data_path, model_dir="production_models", output_dir="model_monitoring", window_size=7):
    """Monitor model performance over time."""
    logger.info(f"Monitoring model performance using data from {data_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(data_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
    
    # Generate advanced features
    df_advanced = engineer_advanced_features(df)
    
    # Load models
    models = {}
    
    # Load global model
    global_model_path = os.path.join(model_dir, "global_model_advanced_features.pkl")
    if os.path.exists(global_model_path):
        models['global'] = joblib.load(global_model_path)
        logger.info(f"Loaded global model from {global_model_path}")
    
    # Load location-specific models
    for model_file in os.listdir(model_dir):
        if model_file.startswith("location_") and model_file.endswith(".pkl"):
            location_id = model_file.split("_")[1]
            model_path = os.path.join(model_dir, model_file)
            models[f'location_{location_id}'] = joblib.load(model_path)
            logger.info(f"Loaded model for location {location_id} from {model_path}")
    
    if not models:
        logger.error(f"No models found in {model_dir}")
        return False
    
    # Prepare features and target
    exclude_cols = ['timestamp', 'date', 'occupancy']
    X = df_advanced.drop(columns=[col for col in exclude_cols if col in df_advanced.columns])
    y = df_advanced['occupancy']
    
    # Create time windows for monitoring
    if 'timestamp' in df.columns:
        df_advanced['date'] = df_advanced['timestamp'].dt.date
        dates = df_advanced['date'].unique()
        
        # Create windows of specified size
        windows = []
        for i in range(0, len(dates), window_size):
            window_dates = dates[i:i+window_size]
            if len(window_dates) == window_size:  # Only use complete windows
                windows.append(window_dates)
        
        # Calculate metrics for each window
        window_metrics = []
        
        for window_idx, window_dates in enumerate(windows):
            logger.info(f"Processing window {window_idx+1}/{len(windows)}")
            
            # Filter data for this window
            window_mask = df_advanced['date'].isin(window_dates)
            X_window = X[window_mask]
            y_window = y[window_mask]
            
            # Skip if window is empty
            if len(X_window) == 0:
                continue
            
            window_start = min(window_dates)
            window_end = max(window_dates)
            
            # Calculate metrics for each model
            for model_name, model in models.items():
                # For location-specific models, only use data for that location
                if model_name.startswith('location_'):
                    location_id = model_name.split('_')[1]
                    if 'location_id' in df_advanced.columns:
                        loc_mask = (df_advanced['location_id'] == location_id) & window_mask
                        X_loc = X[loc_mask]
                        y_loc = y[loc_mask]
                        
                        # Skip if no data for this location in this window
                        if len(X_loc) == 0:
                            continue
                    else:
                        continue
                else:
                    X_loc = X_window
                    y_loc = y_window
                
                # Make predictions
                y_pred = model.predict(X_loc)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_loc, y_pred))
                r2 = r2_score(y_loc, y_pred)
                mae = mean_absolute_error(y_loc, y_pred)
                
                # Add to results
                window_metrics.append({
                    'window_idx': window_idx,
                    'window_start': window_start,
                    'window_end': window_end,
                    'model': model_name,
                    'rmse': rmse,
                    'r2': r2,
                    'mae': mae,
                    'data_size': len(X_loc)
                })
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(window_metrics)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, "window_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Create visualizations
        if not metrics_df.empty:
            # Plot RMSE over time by model
            plt.figure(figsize=(12, 6))
            for model_name in metrics_df['model'].unique():
                model_data = metrics_df[metrics_df['model'] == model_name]
                plt.plot(model_data['window_idx'], model_data['rmse'], marker='o', label=model_name)
            
            plt.xlabel('Window Index')
            plt.ylabel('RMSE')
            plt.title('RMSE by Window and Model')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            rmse_path = os.path.join(output_dir, "rmse_by_window.png")
            plt.savefig(rmse_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"RMSE plot saved to {rmse_path}")
            
            # Plot R² over time by model
            plt.figure(figsize=(12, 6))
            for model_name in metrics_df['model'].unique():
                model_data = metrics_df[metrics_df['model'] == model_name]
                plt.plot(model_data['window_idx'], model_data['r2'], marker='o', label=model_name)
            
            plt.xlabel('Window Index')
            plt.ylabel('R²')
            plt.title('R² by Window and Model')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            r2_path = os.path.join(output_dir, "r2_by_window.png")
            plt.savefig(r2_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"R² plot saved to {r2_path}")
            
            # Check for performance degradation
            for model_name in metrics_df['model'].unique():
                model_data = metrics_df[metrics_df['model'] == model_name].sort_values('window_idx')
                
                if len(model_data) >= 3:  # Need at least 3 windows to detect trend
                    # Calculate trend in RMSE
                    rmse_values = model_data['rmse'].values
                    rmse_trend = np.polyfit(range(len(rmse_values)), rmse_values, 1)[0]
                    
                    # Calculate trend in R²
                    r2_values = model_data['r2'].values
                    r2_trend = np.polyfit(range(len(r2_values)), r2_values, 1)[0]
                    
                    # Check for degradation
                    if rmse_trend > 0.1 or r2_trend < -0.05:  # Thresholds for degradation
                        logger.warning(f"Performance degradation detected for {model_name}:")
                        logger.warning(f"  RMSE trend: {rmse_trend:.4f} (positive means degradation)")
                        logger.warning(f"  R² trend: {r2_trend:.4f} (negative means degradation)")
                        
                        # Add to alerts
                        alert_path = os.path.join(output_dir, "performance_alerts.txt")
                        with open(alert_path, 'a') as f:
                            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Performance degradation detected for {model_name}:\n")
                            f.write(f"  RMSE trend: {rmse_trend:.4f} (positive means degradation)\n")
                            f.write(f"  R² trend: {r2_trend:.4f} (negative means degradation)\n\n")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor model performance over time")
    parser.add_argument("--data", default="data/prepared_data_improved.csv", help="Path to the data file")
    parser.add_argument("--model_dir", default="production_models", help="Directory containing model files")
    parser.add_argument("--output", default="model_monitoring", help="Output directory for monitoring results")
    parser.add_argument("--window_size", type=int, default=7, help="Size of time windows in days")
    
    args = parser.parse_args()
    
    monitor_model_performance(args.data, args.model_dir, args.output, args.window_size) 