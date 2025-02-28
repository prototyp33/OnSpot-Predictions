#!/usr/bin/env python
"""
Script for creating ensemble models from multiple configurations.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import os
import argparse
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from parking_sim.model import ParkingModel
from parking_sim.data_ingestion import DataIngestion

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_ensemble_model(data_path, config_paths, output_dir=None):
    """
    Create an ensemble model from multiple configurations.
    
    Args:
        data_path: Path to parking data
        config_paths: List of paths to model configurations
        output_dir: Directory to save results
    """
    # Load data
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each location
    locations = data['location_id'].unique()
    
    # Store results for all locations
    all_results = {}
    
    for location_id in locations:
        logger.info(f"Creating ensemble for location: {location_id}")
        
        # Filter data for this location
        location_data = data[data['location_id'] == location_id].copy()
        
        # Split into train and test
        train_data = location_data[location_data['timestamp'] < '2023-06-01'].copy()
        test_data = location_data[location_data['timestamp'] >= '2023-06-01'].copy()
        
        # Convert timestamps to datetime
        train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])
        test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
        
        # Extract features
        train_timestamps = train_data['timestamp'].tolist()
        test_timestamps = test_data['timestamp'].tolist()
        
        # Extract weather data
        if all(col in train_data.columns for col in ['temperature', 'humidity', 'wind_speed', 'precipitation']):
            train_weather = (
                train_data['temperature'].values,
                train_data['humidity'].values,
                train_data['wind_speed'].values,
                train_data['precipitation'].values
            )
            test_weather = (
                test_data['temperature'].values,
                test_data['humidity'].values,
                test_data['wind_speed'].values,
                test_data['precipitation'].values
            )
        else:
            # Use synthetic weather data if not available
            train_weather = None
            test_weather = None
        
        # Determine parking type
        parking_type = location_data['parking_type'].iloc[0]
        if parking_type == 'Mixed':
            # Infer type based on patterns
            if location_data['occupancy'].mean() > 0.6:
                parking_type = 'Public'
            else:
                parking_type = 'Resident'
        
        # Create location factors
        location_factors = {
            'zone_type': location_data['zone_type'].iloc[0],
            'capacity': location_data['capacity'].iloc[0],
            'latitude': location_data['latitude'].iloc[0],
            'longitude': location_data['longitude'].iloc[0]
        }
        
        # Initialize models with different configurations
        models = []
        weights = []
        
        for i, config_path in enumerate(config_paths):
            logger.info(f"Training model {i+1}/{len(config_paths)}")
            
            # Load configuration
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Get ensemble weight
            ensemble_weight = config.get('ensemble_weight', 1.0 / len(config_paths))
            weights.append(ensemble_weight)
            
            # Create model
            model = ParkingModel(config)
            models.append(model)
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Make predictions with each model and combine
        ensemble_predictions = np.zeros(len(test_timestamps))
        
        for i, model in enumerate(models):
            # Use predict_occupancy instead of predict
            predictions = model.predict_occupancy(
                test_timestamps,
                test_weather,
                location_factors,
                parking_type,
                config.get('model_weights', {})
            )
            
            ensemble_predictions += weights[i] * predictions
        
        # Calculate metrics
        mse = mean_squared_error(test_data['occupancy'].values, ensemble_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_data['occupancy'].values, ensemble_predictions)
        r2 = r2_score(test_data['occupancy'].values, ensemble_predictions)
        
        logger.info(f"Location {location_id} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Store results
        all_results[location_id] = {
            'predictions': ensemble_predictions.tolist(),
            'actual': test_data['occupancy'].values.tolist(),
            'timestamps': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in test_timestamps],
            'metrics': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2)
            }
        }
        
        # Generate visualizations
        if output_dir:
            plt.figure(figsize=(14, 8))
            
            # Format timestamps for better x-axis labels
            formatted_dates = [ts.strftime('%m-%d %H:%M') for ts in test_data['timestamp']]
            x_ticks = np.arange(0, len(formatted_dates), max(1, len(formatted_dates)//10))
            
            # Plot with better styling
            plt.plot(test_data['timestamp'], test_data['occupancy'], 'b-', linewidth=2, label='Actual')
            plt.plot(test_data['timestamp'], ensemble_predictions, 'r-', linewidth=2, label='Ensemble Prediction')
            
            # Add shaded error region
            error = np.abs(test_data['occupancy'].values - ensemble_predictions)
            plt.fill_between(test_data['timestamp'], 
                            ensemble_predictions - error, 
                            ensemble_predictions + error, 
                            color='red', alpha=0.2)
            
            # Improve styling
            plt.title(f'Ensemble Model: Actual vs Predicted - Location {location_id}', fontsize=16)
            plt.xlabel('Time', fontsize=14)
            plt.ylabel('Occupancy (%)', fontsize=14)
            plt.xticks(test_data['timestamp'][x_ticks], [formatted_dates[i] for i in x_ticks], rotation=45)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'ensemble_predictions_{location_id}.png'), dpi=300)
            plt.close()
            
            # Error distribution
            plt.figure(figsize=(10, 6))
            error = test_data['occupancy'].values - ensemble_predictions
            
            # Create histogram with KDE
            sns.histplot(error, kde=True, bins=20, color='darkblue')
            
            # Add vertical line at zero
            plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
            
            # Add statistics
            plt.text(0.05, 0.95, f'Mean Error: {np.mean(error):.3f}\nStd Dev: {np.std(error):.3f}',
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8))
            
            plt.title(f'Ensemble Model: Error Distribution - Location {location_id}', fontsize=16)
            plt.xlabel('Error (Actual - Predicted)', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'ensemble_error_dist_{location_id}.png'), dpi=300)
            plt.close()
    
    # Save overall results
    if output_dir:
        with open(os.path.join(output_dir, 'ensemble_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Create summary report
        summary_df = pd.DataFrame([
            {
                'location_id': loc_id,
                'rmse': results['metrics']['rmse'],
                'r2': results['metrics']['r2'],
                'mae': results['metrics']['mae']
            }
            for loc_id, results in all_results.items()
        ])
        
        summary_df.to_csv(os.path.join(output_dir, 'ensemble_summary.csv'), index=False)
        
        # Create summary visualization
        plt.figure(figsize=(14, 8))
        
        # Sort by RMSE for better visualization
        summary_plot_df = summary_df.sort_values('rmse')
        
        # Create bar plot with gradient colors based on R²
        r2_values = summary_plot_df['r2'].values
        colors = plt.cm.RdYlGn(np.clip((r2_values + 0.5) / 1.5, 0, 1))  # Map R² to colors
        
        bars = plt.bar(summary_plot_df['location_id'], summary_plot_df['rmse'], color=colors)
        
        # Add R² values as text on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            r2 = summary_plot_df['r2'].iloc[i]
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'R²: {r2:.2f}', ha='center', fontsize=10)
        
        plt.title('Ensemble Model: RMSE by Location (Color indicates R² value)', fontsize=16)
        plt.xlabel('Location ID', fontsize=14)
        plt.ylabel('RMSE', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add color bar to show R² scale
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
                                  norm=plt.Normalize(vmin=-0.5, vmax=1.0))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('R² Value', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ensemble_rmse_by_location.png'), dpi=300)
        plt.close()
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ensemble model from multiple configurations")
    parser.add_argument("--data", required=True, help="Path to parking data")
    parser.add_argument("--configs", required=True, nargs='+', help="Paths to model configurations")
    parser.add_argument("--output", default="ensemble_results", help="Output directory")
    
    args = parser.parse_args()
    
    create_ensemble_model(args.data, args.configs, args.output) 