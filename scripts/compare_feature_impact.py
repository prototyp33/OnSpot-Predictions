#!/usr/bin/env python
"""
Script for comparing model performance with different feature sets and model strategies.
Tests the impact of advanced features and location-specific models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import os
import logging
import argparse
from datetime import datetime
import joblib
import time
from scripts.parking_sim.feature_engineering import create_features
from scripts.parking_sim.advanced_features import engineer_advanced_features

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory
output_dir = "feature_impact_results"
os.makedirs(output_dir, exist_ok=True)

def load_data(file_path):
    """Load and prepare the dataset for model comparison."""
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    logger.info(f"Dataset loaded with shape: {df.shape}")
    return df

def prepare_feature_sets(df):
    """Prepare different feature sets for comparison."""
    logger.info("Preparing feature sets...")
    
    # Basic features (existing features in the dataset)
    basic_features = df.copy()
    
    # Advanced features (using our advanced feature engineering)
    advanced_features = engineer_advanced_features(df)
    
    logger.info(f"Basic features shape: {basic_features.shape}")
    logger.info(f"Advanced features shape: {advanced_features.shape}")
    
    return basic_features, advanced_features

def train_global_model(df, feature_set_name):
    """Train a global model on the entire dataset."""
    logger.info(f"Training global model with {feature_set_name} features...")
    
    # Exclude non-feature columns
    exclude_cols = ['timestamp', 'date', 'occupancy']
    if 'location_id' in df.columns:
        exclude_cols.append('location_id')
    
    # Prepare features and target
    X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    y = df['occupancy']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    start_time = time.time()
    
    # Use Gradient Boosting for better performance
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Evaluate model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    logger.info(f"Global model with {feature_set_name} features - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
    
    # Save model
    model_path = os.path.join(output_dir, f"global_model_{feature_set_name}.pkl")
    joblib.dump(model, model_path)
    
    # Return metrics and test data for visualization
    return {
        'model_type': 'global',
        'feature_set': feature_set_name,
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'training_time': training_time,
        'test_data': (X_test, y_test, y_pred)
    }

def train_location_specific_models(df, feature_set_name):
    """Train separate models for each location."""
    logger.info(f"Training location-specific models with {feature_set_name} features...")
    
    if 'location_id' not in df.columns:
        logger.error("Cannot train location-specific models: 'location_id' column not found")
        return None
    
    # Get unique locations
    locations = df['location_id'].unique()
    logger.info(f"Training models for {len(locations)} locations")
    
    # Exclude non-feature columns
    exclude_cols = ['timestamp', 'date', 'occupancy', 'location_id']
    
    # Prepare for collecting results
    all_metrics = []
    all_test_data = []
    total_training_time = 0
    
    # Train a model for each location
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
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        start_time = time.time()
        
        # Use Gradient Boosting for better performance
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        loc_training_time = time.time() - start_time
        total_training_time += loc_training_time
        
        # Evaluate model
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        logger.info(f"Location {loc} model - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
        
        # Save model
        model_path = os.path.join(output_dir, f"location_{loc}_model_{feature_set_name}.pkl")
        joblib.dump(model, model_path)
        
        # Store metrics
        all_metrics.append({
            'location_id': loc,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'training_time': loc_training_time,
            'data_size': len(loc_df)
        })
        
        # Store test data
        all_test_data.append((X_test, y_test, y_pred, loc))
    
    # Calculate average metrics
    avg_metrics = {
        'model_type': 'location_specific',
        'feature_set': feature_set_name,
        'rmse': np.mean([m['rmse'] for m in all_metrics]),
        'r2': np.mean([m['r2'] for m in all_metrics]),
        'mae': np.mean([m['mae'] for m in all_metrics]),
        'training_time': total_training_time,
        'test_data': all_test_data,
        'location_metrics': all_metrics
    }
    
    logger.info(f"Average metrics for location-specific models - RMSE: {avg_metrics['rmse']:.4f}, R²: {avg_metrics['r2']:.4f}, MAE: {avg_metrics['mae']:.4f}")
    
    # Save detailed metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(output_dir, f"location_specific_metrics_{feature_set_name}.csv"), index=False)
    
    return avg_metrics

def visualize_results(results):
    """Create visualizations comparing model performance."""
    logger.info("Creating performance visualizations...")
    
    # Create DataFrame from results
    results_df = pd.DataFrame([
        {
            'Model Type': r['model_type'].replace('_', ' ').title(),
            'Feature Set': r['feature_set'].replace('_', ' ').title(),
            'RMSE': r['rmse'],
            'R²': r['r2'],
            'MAE': r['mae'],
            'Training Time (s)': r['training_time']
        }
        for r in results
    ])
    
    # Save results to CSV
    results_df.to_csv(os.path.join(output_dir, "model_comparison_results.csv"), index=False)
    
    # Create bar charts for each metric
    metrics = ['RMSE', 'R²', 'MAE']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        chart = sns.barplot(x='Model Type', y=metric, hue='Feature Set', data=results_df)
        
        # Add value labels on top of bars
        for i, p in enumerate(chart.patches):
            chart.annotate(f'{p.get_height():.4f}', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha = 'center', va = 'bottom',
                         xytext = (0, 5), textcoords = 'offset points')
        
        plt.title(f'Comparison of {metric} by Model Type and Feature Set')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric.lower().replace('²', '2')}_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create training time comparison
    plt.figure(figsize=(10, 6))
    chart = sns.barplot(x='Model Type', y='Training Time (s)', hue='Feature Set', data=results_df)
    
    # Add value labels
    for i, p in enumerate(chart.patches):
        chart.annotate(f'{p.get_height():.2f}s', 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha = 'center', va = 'bottom',
                     xytext = (0, 5), textcoords = 'offset points')
    
    plt.title('Training Time Comparison')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_time_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create scatter plots of actual vs predicted values
    for result in results:
        if result['model_type'] == 'global':
            X_test, y_test, y_pred = result['test_data']
            
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            
            # Add perfect prediction line
            max_val = max(y_test.max(), y_pred.max())
            min_val = min(y_test.min(), y_pred.min())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.title(f"Actual vs Predicted - {result['model_type'].title()} Model with {result['feature_set'].title()} Features")
            plt.xlabel('Actual Occupancy')
            plt.ylabel('Predicted Occupancy')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"scatter_{result['model_type']}_{result['feature_set']}.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    # If we have location-specific results, create a comparison by location
    for result in results:
        if result['model_type'] == 'location_specific' and 'location_metrics' in result:
            loc_metrics = result['location_metrics']
            loc_df = pd.DataFrame(loc_metrics)
            
            # Sort by data size
            loc_df = loc_df.sort_values('data_size', ascending=False)
            
            # Plot R² by location
            plt.figure(figsize=(12, 6))
            sns.barplot(x='location_id', y='r2', data=loc_df)
            plt.title(f"R² by Location - {result['feature_set'].title()} Features")
            plt.xlabel('Location ID')
            plt.ylabel('R²')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"r2_by_location_{result['feature_set']}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot data size vs R²
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='data_size', y='r2', data=loc_df)
            plt.title(f"R² vs Data Size - {result['feature_set'].title()} Features")
            plt.xlabel('Number of Data Points')
            plt.ylabel('R²')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"r2_vs_data_size_{result['feature_set']}.png"), dpi=300, bbox_inches='tight')
            plt.close()

def create_summary_report(results):
    """Create a summary report of the model comparison."""
    logger.info("Creating summary report...")
    
    with open(os.path.join(output_dir, "model_comparison_summary.txt"), 'w') as f:
        f.write("=== MODEL COMPARISON SUMMARY ===\n\n")
        
        f.write("PERFORMANCE COMPARISON:\n")
        f.write("-----------------------\n\n")
        
        # Sort results by R² (descending)
        sorted_results = sorted(results, key=lambda x: x['r2'], reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            model_type = result['model_type'].replace('_', ' ').title()
            feature_set = result['feature_set'].replace('_', ' ').title()
            
            f.write(f"{i}. {model_type} Model with {feature_set} Features:\n")
            f.write(f"   R² = {result['r2']:.4f}, RMSE = {result['rmse']:.4f}, MAE = {result['mae']:.4f}\n")
            f.write(f"   Training Time: {result['training_time']:.2f} seconds\n\n")
        
        # Calculate improvement from basic to advanced features
        global_basic = next((r for r in results if r['model_type'] == 'global' and r['feature_set'] == 'basic_features'), None)
        global_advanced = next((r for r in results if r['model_type'] == 'global' and r['feature_set'] == 'advanced_features'), None)
        
        if global_basic and global_advanced:
            r2_improvement = (global_advanced['r2'] - global_basic['r2']) / global_basic['r2'] * 100
            rmse_improvement = (global_basic['rmse'] - global_advanced['rmse']) / global_basic['rmse'] * 100
            
            f.write("FEATURE ENGINEERING IMPACT:\n")
            f.write("--------------------------\n")
            f.write(f"Advanced features improved global model performance by:\n")
            f.write(f"- R² improvement: {r2_improvement:.2f}%\n")
            f.write(f"- RMSE reduction: {rmse_improvement:.2f}%\n\n")
        
        # Compare global vs location-specific
        loc_advanced = next((r for r in results if r['model_type'] == 'location_specific' and r['feature_set'] == 'advanced_features'), None)
        
        if global_advanced and loc_advanced:
            r2_diff = (loc_advanced['r2'] - global_advanced['r2']) / global_advanced['r2'] * 100
            rmse_diff = (global_advanced['rmse'] - loc_advanced['rmse']) / global_advanced['rmse'] * 100
            
            f.write("LOCATION-SPECIFIC VS GLOBAL MODEL:\n")
            f.write("--------------------------------\n")
            
            if r2_diff > 0:
                f.write(f"Location-specific models outperform the global model by:\n")
                f.write(f"- R² improvement: {r2_diff:.2f}%\n")
                f.write(f"- RMSE reduction: {rmse_diff:.2f}%\n\n")
            else:
                f.write(f"The global model outperforms location-specific models by:\n")
                f.write(f"- R² improvement: {-r2_diff:.2f}%\n")
                f.write(f"- RMSE reduction: {-rmse_diff:.2f}%\n\n")
        
        # Add location-specific insights if available
        for result in results:
            if result['model_type'] == 'location_specific' and 'location_metrics' in result:
                loc_metrics = pd.DataFrame(result['location_metrics'])
                
                f.write(f"LOCATION-SPECIFIC INSIGHTS ({result['feature_set'].replace('_', ' ').title()}):\n")
                f.write("------------------------------------------\n")
                
                # Best performing locations
                best_locs = loc_metrics.sort_values('r2', ascending=False).head(3)
                f.write("Top 3 best-performing locations:\n")
                for _, row in best_locs.iterrows():
                    f.write(f"- Location {row['location_id']}: R² = {row['r2']:.4f}, RMSE = {row['rmse']:.4f}\n")
                
                # Worst performing locations
                worst_locs = loc_metrics.sort_values('r2').head(3)
                f.write("\nTop 3 worst-performing locations:\n")
                for _, row in worst_locs.iterrows():
                    f.write(f"- Location {row['location_id']}: R² = {row['r2']:.4f}, RMSE = {row['rmse']:.4f}\n")
                
                f.write("\n")
        
        # Overall recommendations
        f.write("RECOMMENDATIONS:\n")
        f.write("---------------\n")
        
        best_model = sorted_results[0]
        best_model_type = best_model['model_type'].replace('_', ' ').title()
        best_feature_set = best_model['feature_set'].replace('_', ' ').title()
        
        f.write(f"1. Use the {best_model_type} Model with {best_feature_set} Features for best performance.\n")
        
        if global_basic and global_advanced and global_advanced['r2'] > global_basic['r2']:
            f.write("2. The advanced feature engineering significantly improves model performance.\n")
        
        if loc_advanced and global_advanced:
            if loc_advanced['r2'] > global_advanced['r2']:
                f.write("3. Consider using location-specific models for better accuracy, especially for locations with sufficient data.\n")
            else:
                f.write("3. A single global model performs well across all locations and is simpler to maintain.\n")

def main(data_path):
    """Main function to run the model comparison."""
    logger.info("Starting model comparison...")
    
    # Load data
    df = load_data(data_path)
    
    # Prepare feature sets
    basic_features, advanced_features = prepare_feature_sets(df)
    
    # Train and evaluate models
    results = []
    
    # Global model with basic features
    global_basic_results = train_global_model(basic_features, "basic_features")
    results.append(global_basic_results)
    
    # Global model with advanced features
    global_advanced_results = train_global_model(advanced_features, "advanced_features")
    results.append(global_advanced_results)
    
    # Location-specific models with basic features
    loc_basic_results = train_location_specific_models(basic_features, "basic_features")
    if loc_basic_results:
        results.append(loc_basic_results)
    
    # Location-specific models with advanced features
    loc_advanced_results = train_location_specific_models(advanced_features, "advanced_features")
    if loc_advanced_results:
        results.append(loc_advanced_results)
    
    # Visualize results
    visualize_results(results)
    
    # Create summary report
    create_summary_report(results)
    
    logger.info(f"Model comparison completed. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare model performance with different feature sets and model strategies")
    parser.add_argument("--data", default="data/prepared_data_improved.csv", help="Path to the prepared data file")
    
    args = parser.parse_args()
    main(args.data) 