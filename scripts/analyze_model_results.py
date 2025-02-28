#!/usr/bin/env python
"""
Script for analyzing the results of model comparison for parking occupancy prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Input and output directories
input_dir = "model_comparison_results"
output_dir = "model_analysis_results"
os.makedirs(output_dir, exist_ok=True)

def load_results():
    """Load the model comparison results."""
    logger.info("Loading model comparison results...")
    
    # Load final results
    results_path = os.path.join(input_dir, 'final_model_comparison_results.csv')
    if not os.path.exists(results_path):
        logger.error(f"Results file not found: {results_path}")
        return None
    
    results_df = pd.read_csv(results_path)
    logger.info(f"Loaded results for {len(results_df)} models")
    
    # Load feature importance
    importance_path = os.path.join(input_dir, 'feature_importance.csv')
    if os.path.exists(importance_path):
        importance_df = pd.read_csv(importance_path)
        logger.info(f"Loaded importance for {len(importance_df)} features")
    else:
        importance_df = None
        logger.warning("Feature importance file not found")
    
    return results_df, importance_df

def analyze_model_performance(results_df):
    """Analyze and visualize model performance metrics."""
    logger.info("Analyzing model performance...")
    
    # Create performance comparison plots
    metrics = ['r2', 'rmse', 'mae']
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        # Sort by the current metric (ascending for error metrics, descending for R²)
        ascending = metric != 'r2'
        sorted_df = results_df.sort_values(metric, ascending=ascending)
        
        # Create bar plot
        ax = sns.barplot(x='model', y=metric, data=sorted_df)
        
        # Add value labels on top of bars
        for i, v in enumerate(sorted_df[metric]):
            ax.text(i, v + (v * 0.01), f"{v:.4f}", ha='center')
        
        plt.title(f'Model Comparison: {metric.upper()}')
        plt.xlabel('Model')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'performance_comparison_{metric}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a radar chart for the top 3 models
    top_models = results_df.head(3)['model'].tolist()
    
    # Normalize metrics for radar chart
    radar_metrics = ['r2', 'training_time']
    inverse_metrics = ['rmse', 'mae', 'mse']
    
    # Add inverse metrics (1 - normalized value)
    for metric in inverse_metrics:
        if metric in results_df.columns:
            normalized = 1 - (results_df[metric] / results_df[metric].max())
            results_df[f'inv_{metric}'] = normalized
            radar_metrics.append(f'inv_{metric}')
    
    # Filter for top models and prepare for radar chart
    radar_df = results_df[results_df['model'].isin(top_models)].copy()
    
    # Normalize training time (inverse, since lower is better)
    if 'training_time' in radar_df.columns:
        max_time = radar_df['training_time'].max()
        if max_time > 0:
            radar_df['training_time'] = 1 - (radar_df['training_time'] / max_time)
    
    # Create radar chart
    try:
        create_radar_chart(radar_df, radar_metrics, 'model')
    except Exception as e:
        logger.warning(f"Could not create radar chart: {e}")
    
    # Create training time vs performance plot
    if 'training_time' in results_df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['training_time'], results_df['r2'], s=100, alpha=0.7)
        
        # Add model labels
        for i, row in results_df.iterrows():
            plt.annotate(row['model'], 
                        (row['training_time'], row['r2']),
                        xytext=(5, 5),
                        textcoords='offset points')
        
        plt.title('Training Time vs R² Score')
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('R² Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_time_vs_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    return top_models

def create_radar_chart(df, metrics, label_col):
    """Create a radar chart comparing multiple models across metrics."""
    # Number of variables
    N = len(metrics)
    
    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add metric labels
    plt.xticks(angles[:-1], metrics, size=12)
    
    # Draw y-axis labels (0-1)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], size=10)
    plt.ylim(0, 1)
    
    # Plot each model
    for i, row in df.iterrows():
        model_name = row[label_col]
        values = [row[metric] for metric in metrics]
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Model Comparison Across Metrics', size=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_radar_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_feature_importance(importance_df):
    """Analyze feature importance in more detail."""
    if importance_df is None:
        logger.warning("No feature importance data available")
        return
    
    logger.info("Analyzing feature importance...")
    
    # Group features by type/category
    feature_categories = {
        'time': ['hour', 'day', 'month', 'is_weekend', 'day_of_year', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'],
        'weather': ['temperature', 'humidity', 'precipitation', 'wind_speed'],
        'location': ['location_id', 'zone_type', 'capacity'],
        'traffic': ['traffic_level']
    }
    
    # Initialize category importance
    category_importance = {cat: 0 for cat in feature_categories}
    
    # Calculate importance by category
    for idx, row in importance_df.iterrows():
        feature = row['feature'] if 'feature' in importance_df.columns else f"Feature {row['feature_index']}"
        importance = row['importance']
        
        # Check which category this feature belongs to
        for category, keywords in feature_categories.items():
            if any(keyword in feature.lower() for keyword in keywords):
                category_importance[category] += importance
                break
    
    # Create pie chart of category importance
    plt.figure(figsize=(10, 10))
    labels = [f"{cat} ({val:.2f})" for cat, val in category_importance.items() if val > 0]
    values = [val for val in category_importance.values() if val > 0]
    
    if values:  # Only create pie chart if we have values
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Feature Importance by Category')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance_by_category.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create horizontal bar chart for top 20 features
    plt.figure(figsize=(12, 10))
    top_features = importance_df.head(20)
    feature_col = 'feature' if 'feature' in importance_df.columns else 'feature_index'
    
    # Create bar chart
    sns.barplot(x='importance', y=feature_col, data=top_features)
    plt.title('Top 20 Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_features_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save category importance to file
    with open(os.path.join(output_dir, 'category_importance.txt'), 'w') as f:
        f.write("=== FEATURE IMPORTANCE BY CATEGORY ===\n\n")
        for category, importance in sorted(category_importance.items(), key=lambda x: x[1], reverse=True):
            if importance > 0:
                f.write(f"{category}: {importance:.4f} ({importance*100:.2f}%)\n")

def analyze_error_patterns(top_models):
    """Analyze error patterns in the best model's predictions."""
    logger.info("Analyzing error patterns...")
    
    # Try to load the best model
    best_model_name = top_models[0].replace(' ', '_').lower()
    model_path = os.path.join(input_dir, f"{best_model_name}_model.pkl")
    
    if not os.path.exists(model_path):
        logger.error(f"Best model file not found: {model_path}")
        return
    
    try:
        # Load the model and original data
        best_model = joblib.load(model_path)
        
        # We need to load the original data and use the same preprocessing
        data_path = "data/prepared_data_improved.csv"
        if not os.path.exists(data_path):
            logger.error(f"Original data file not found: {data_path}")
            return
        
        # Load original data
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time components (same as in the original script)
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'] >= 5
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Create cyclical features for time (same as in the original script)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # Load the preprocessor if available
        preprocessor_path = os.path.join(input_dir, "preprocessor.pkl")
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
        else:
            # We need to recreate the preprocessor
            logger.warning("Preprocessor not found, recreating it")
            
            # Identify numerical and categorical columns
            exclude_cols = ['occupancy', 'timestamp', 'date']
            numerical_cols = []
            categorical_cols = []
            
            for col in df.columns:
                if col in exclude_cols:
                    continue
                
                if np.issubdtype(df[col].dtype, np.number):
                    numerical_cols.append(col)
                else:
                    categorical_cols.append(col)
            
            # Create preprocessing pipeline
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.model_selection import train_test_split
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                ]
            )
        
        # Target variable
        y = df['occupancy']
        
        # Features (excluding target and timestamp)
        X = df.drop(['occupancy', 'timestamp'], axis=1)
        if 'date' in X.columns:
            X = X.drop('date', axis=1)
        
        # Split data with the same random state as original
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit preprocessor on training data
        X_train_processed = preprocessor.fit_transform(X_train)
        
        # Now process test data
        X_test_processed = preprocessor.transform(X_test)
        
        # Make predictions
        y_pred = best_model.predict(X_test_processed)
        
        # Calculate errors
        errors = y_test - y_pred
        
        # Add predictions and errors to test data
        X_test_with_results = X_test.copy()
        X_test_with_results['actual'] = y_test.values
        X_test_with_results['predicted'] = y_pred
        X_test_with_results['error'] = errors
        X_test_with_results['abs_error'] = np.abs(errors)
        
        # Add timestamp back
        test_indices = X_test.index
        X_test_with_results['timestamp'] = df.loc[test_indices, 'timestamp'].values
        
        # Extract time components
        X_test_with_results['hour'] = X_test_with_results['timestamp'].dt.hour
        X_test_with_results['day_of_week'] = X_test_with_results['timestamp'].dt.dayofweek
        X_test_with_results['month'] = X_test_with_results['timestamp'].dt.month
        X_test_with_results['is_weekend'] = X_test_with_results['day_of_week'] >= 5
        
        # Analyze errors by time of day
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='hour', y='abs_error', data=X_test_with_results)
        plt.title('Error by Hour of Day')
        plt.xlabel('Hour')
        plt.ylabel('Absolute Error')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_by_hour.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Analyze errors by day of week
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='day_of_week', y='abs_error', data=X_test_with_results)
        plt.title('Error by Day of Week')
        plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
        plt.ylabel('Absolute Error')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_by_day.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Analyze errors by occupancy level
        plt.figure(figsize=(12, 6))
        X_test_with_results['occupancy_bin'] = pd.cut(X_test_with_results['actual'], bins=10)
        sns.boxplot(x='occupancy_bin', y='abs_error', data=X_test_with_results)
        plt.title('Error by Occupancy Level')
        plt.xlabel('Occupancy Range')
        plt.ylabel('Absolute Error')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_by_occupancy.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # If location_id exists, analyze errors by location
        if 'location_id' in X_test_with_results.columns:
            plt.figure(figsize=(14, 6))
            sns.boxplot(x='location_id', y='abs_error', data=X_test_with_results)
            plt.title('Error by Location')
            plt.xlabel('Location ID')
            plt.ylabel('Absolute Error')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'error_by_location.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Find the worst predicted cases
        worst_predictions = X_test_with_results.sort_values('abs_error', ascending=False).head(20)
        
        # Save worst predictions to file
        worst_predictions.to_csv(os.path.join(output_dir, 'worst_predictions.csv'), index=False)
        
        # Create summary of error analysis
        with open(os.path.join(output_dir, 'error_analysis_summary.txt'), 'w') as f:
            f.write("=== ERROR ANALYSIS SUMMARY ===\n\n")
            
            f.write("Overall Error Metrics:\n")
            f.write(f"Mean Absolute Error: {np.mean(X_test_with_results['abs_error']):.4f}\n")
            f.write(f"Root Mean Squared Error: {np.sqrt(np.mean(errors**2)):.4f}\n")
            f.write(f"R² Score: {r2_score(y_test, y_pred):.4f}\n\n")
            
            f.write("Error by Time Period:\n")
            hour_errors = X_test_with_results.groupby('hour')['abs_error'].mean().sort_values(ascending=False)
            f.write("Top 3 hours with highest errors:\n")
            for hour, error in hour_errors.head(3).items():
                f.write(f"Hour {hour}: {error:.4f}\n")
            
            day_errors = X_test_with_results.groupby('day_of_week')['abs_error'].mean().sort_values(ascending=False)
            f.write("\nDay of week errors (0=Monday, 6=Sunday):\n")
            for day, error in day_errors.items():
                f.write(f"Day {day}: {error:.4f}\n")
            
            f.write("\nWeekend vs Weekday:\n")
            weekend_error = X_test_with_results[X_test_with_results['is_weekend']]['abs_error'].mean()
            weekday_error = X_test_with_results[~X_test_with_results['is_weekend']]['abs_error'].mean()
            f.write(f"Weekend error: {weekend_error:.4f}\n")
            f.write(f"Weekday error: {weekday_error:.4f}\n")
            
            if 'location_id' in X_test_with_results.columns:
                f.write("\nTop 3 locations with highest errors:\n")
                location_errors = X_test_with_results.groupby('location_id')['abs_error'].mean().sort_values(ascending=False)
                for loc, error in location_errors.head(3).items():
                    f.write(f"Location {loc}: {error:.4f}\n")
            
            f.write("\nWorst Predictions:\n")
            for i, row in worst_predictions.head(5).iterrows():
                f.write(f"- Actual: {row['actual']:.2f}, Predicted: {row['predicted']:.2f}, Error: {row['error']:.2f}")
                if 'timestamp' in row:
                    f.write(f", Time: {row['timestamp']}")
                if 'location_id' in row:
                    f.write(f", Location: {row['location_id']}")
                f.write("\n")
    
    except Exception as e:
        logger.error(f"Error analyzing error patterns: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Main function to run the analysis."""
    logger.info("Starting model results analysis...")
    
    # Load results
    results_df, importance_df = load_results()
    if results_df is None:
        logger.error("Could not load results. Exiting.")
        return
    
    # Analyze model performance
    top_models = analyze_model_performance(results_df)
    
    # Analyze feature importance
    analyze_feature_importance(importance_df)
    
    # Analyze error patterns
    analyze_error_patterns(top_models)
    
    logger.info(f"Analysis completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 