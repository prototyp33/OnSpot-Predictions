#!/usr/bin/env python
"""
analyze_model.py

Analyzes the trained Random Forest model, including:
1. Feature importance analysis with visualizations
2. Model performance metrics and graphs
3. Model validation and robustness checks

Usage:
    python analyze_model.py --model [model_path] --data [data_path] --output [output_dir]
"""

import os
import pandas as pd
import numpy as np
import pickle
import logging
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from typing import List, Dict, Tuple, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model(model_path: str) -> Any:
    """
    Load the trained Random Forest model from disk.
    
    Args:
        model_path: Path to the saved model pickle file
        
    Returns:
        The loaded model object
    """
    logger.info(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Model loaded successfully: {type(model).__name__}")
    return model

def load_and_prepare_data(data_path: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Load and prepare the data for model analysis.
    
    Args:
        data_path: Path to the test data CSV file
        
    Returns:
        Tuple containing:
        - Features DataFrame
        - Target array
        - List of feature names
    """
    logger.info(f"Loading test data from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Convert timestamp to datetime if it exists
    if 'timestamp' in df.columns and df['timestamp'].dtype != 'datetime64[ns]':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract datetime features if needed
        if 'hour' not in df.columns:
            df['hour'] = df['timestamp'].dt.hour
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['timestamp'].dt.dayofweek
        if 'month' not in df.columns:
            df['month'] = df['timestamp'].dt.month
        if 'is_weekend' not in df.columns:
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Encode location_id as categorical if needed
    if 'location_id' in df.columns and 'location_id_categorical' not in df.columns:
        df['location_id_categorical'] = df['location_id'].astype('category').cat.codes
    
    # For this analysis, we'll use a fixed set of features
    # This should match what the model was trained on
    feature_names = [
        'hour', 'day_of_week', 'month', 'is_weekend',
        'location_id_categorical', 'capacity'
    ]
    
    # Check if all features are available
    missing_features = [feat for feat in feature_names if feat not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Extract features and target
    X = df[feature_names]
    y = df['occupancy'] if 'occupancy' in df.columns else None
    
    if y is None:
        logger.warning("Target variable 'occupancy' not found in the dataset.")
        
    logger.info(f"Data prepared with {len(feature_names)} features")
    return X, y, feature_names

def analyze_feature_importance(
    model: Any, 
    X: pd.DataFrame, 
    y: np.ndarray, 
    feature_names: List[str],
    output_dir: str
) -> None:
    """
    Analyze feature importance using multiple methods and create visualizations.
    
    Args:
        model: The trained model
        X: Feature DataFrame
        y: Target array
        feature_names: List of feature names
        output_dir: Directory to save the output plots
    """
    logger.info("Analyzing feature importance...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Method 1: Built-in feature importance
    if hasattr(model, 'feature_importances_'):
        # Get and normalize the feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance (MDI)')
        plt.bar(range(X.shape[1]), importances[indices], align='center')
        plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance_mdi.png'), dpi=300)
        plt.close()
        
        # Save feature importance to CSV
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
        importance_df.to_csv(os.path.join(output_dir, 'feature_importance_mdi.csv'), index=False)
        
        logger.info("Top 5 features by MDI importance:")
        for i in range(min(5, len(indices))):
            logger.info(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Method 2: Permutation importance (more reliable but slower)
    if y is not None:
        logger.info("Calculating permutation feature importance...")
        try:
            result = permutation_importance(
                model, X, y, n_repeats=10, random_state=42, n_jobs=-1
            )
            
            # Sort features by importance
            perm_importance = result.importances_mean
            perm_indices = np.argsort(perm_importance)[::-1]
            
            # Plot permutation importance
            plt.figure(figsize=(12, 8))
            plt.title('Feature Importance (Permutation)')
            plt.bar(range(X.shape[1]), perm_importance[perm_indices], align='center')
            plt.xticks(range(X.shape[1]), [feature_names[i] for i in perm_indices], rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance_permutation.png'), dpi=300)
            plt.close()
            
            # Save permutation importance to CSV
            perm_importance_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in perm_indices],
                'Importance': perm_importance[perm_indices]
            })
            perm_importance_df.to_csv(
                os.path.join(output_dir, 'feature_importance_permutation.csv'), 
                index=False
            )
            
            logger.info("Top 5 features by permutation importance:")
            for i in range(min(5, len(perm_indices))):
                logger.info(
                    f"  {i+1}. {feature_names[perm_indices[i]]}: {perm_importance[perm_indices[i]]:.4f}"
                )
                
        except Exception as e:
            logger.error(f"Error calculating permutation importance: {e}")
    
    logger.info("Feature importance analysis completed")

def evaluate_model_performance(
    model: Any, 
    X: pd.DataFrame, 
    y: np.ndarray,
    output_dir: str
) -> Dict[str, float]:
    """
    Evaluate model performance and create performance visualizations.
    
    Args:
        model: The trained model
        X: Feature DataFrame
        y: Target array
        output_dir: Directory to save the output plots
        
    Returns:
        Dictionary of performance metrics
    """
    if y is None:
        logger.warning("Cannot evaluate model performance: target variable not available")
        return {}
    
    logger.info("Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    metrics = {
        'r2_score': r2_score(y, y_pred),
        'mse': mean_squared_error(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred)
    }
    
    # Log metrics
    logger.info("Performance metrics:")
    logger.info(f"  - R² Score: {metrics['r2_score']:.4f}")
    logger.info(f"  - MSE: {metrics['mse']:.4f}")
    logger.info(f"  - RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  - MAE: {metrics['mae']:.4f}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to a file
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, 'performance_metrics.csv'), index=False)
    
    # Plot 1: Predicted vs Actual
    plt.figure(figsize=(10, 8))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', lw=2)
    plt.title('Predicted vs Actual Occupancy')
    plt.xlabel('Actual Occupancy')
    plt.ylabel('Predicted Occupancy')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'predicted_vs_actual.png'), dpi=300)
    plt.close()
    
    # Plot 2: Residuals
    residuals = y - y_pred
    plt.figure(figsize=(10, 8))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.title('Residual Plot')
    plt.xlabel('Predicted Occupancy')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'residuals.png'), dpi=300)
    plt.close()
    
    # Plot 3: Residual distribution
    plt.figure(figsize=(10, 8))
    sns.histplot(residuals, kde=True)
    plt.title('Distribution of Residuals')
    plt.xlabel('Residual Value')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'residual_distribution.png'), dpi=300)
    plt.close()
    
    # Plot 4: Error distribution by location
    if 'location_id' in X.columns or 'location_id_categorical' in X.columns:
        loc_col = 'location_id' if 'location_id' in X.columns else 'location_id_categorical'
        error_by_loc = pd.DataFrame({
            'location': X[loc_col],
            'abs_error': np.abs(residuals)
        })
        
        # Aggregate errors by location
        loc_errors = error_by_loc.groupby('location')['abs_error'].agg(['mean', 'median', 'count'])
        loc_errors = loc_errors.sort_values('mean', ascending=False)
        
        # Plot top 20 locations with highest error
        top_n = min(20, len(loc_errors))
        plt.figure(figsize=(12, 8))
        top_locs = loc_errors.head(top_n)
        top_locs['mean'].plot(kind='bar')
        plt.title(f'Top {top_n} Locations with Highest Error')
        plt.ylabel('Mean Absolute Error')
        plt.xlabel('Location')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_by_location.png'), dpi=300)
        plt.close()
        
        # Save location errors to CSV
        loc_errors.to_csv(os.path.join(output_dir, 'error_by_location.csv'))
    
    # Plot 5: Error by time of day (if hour feature is available)
    if 'hour' in X.columns:
        error_by_hour = pd.DataFrame({
            'hour': X['hour'],
            'abs_error': np.abs(residuals)
        })
        
        # Aggregate errors by hour
        hour_errors = error_by_hour.groupby('hour')['abs_error'].mean()
        
        plt.figure(figsize=(12, 6))
        hour_errors.plot(kind='line', marker='o')
        plt.title('Error by Hour of Day')
        plt.ylabel('Mean Absolute Error')
        plt.xlabel('Hour')
        plt.xticks(range(0, 24))
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'error_by_hour.png'), dpi=300)
        plt.close()
        
        # Save hour errors to CSV
        hour_errors.to_csv(os.path.join(output_dir, 'error_by_hour.csv'))
    
    logger.info("Model performance evaluation completed")
    return metrics

def analyze_model_robustness(
    model: Any, 
    X: pd.DataFrame, 
    y: np.ndarray,
    output_dir: str
) -> None:
    """
    Analyze model robustness to different data conditions.
    
    Args:
        model: The trained model
        X: Feature DataFrame
        y: Target array
        output_dir: Directory to save the output
    """
    if y is None:
        logger.warning("Cannot analyze model robustness: target variable not available")
        return
    
    logger.info("Analyzing model robustness...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Performance by day of week
    if 'day_of_week' in X.columns:
        performance_by_dow = []
        for dow in range(7):
            # Filter data for this day of week
            mask = X['day_of_week'] == dow
            if mask.sum() > 0:
                X_dow = X[mask]
                y_dow = y[mask]
                
                # Make predictions
                y_pred_dow = model.predict(X_dow)
                
                # Calculate metrics
                metrics = {
                    'day_of_week': dow,
                    'day_name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][dow],
                    'count': mask.sum(),
                    'r2_score': r2_score(y_dow, y_pred_dow),
                    'rmse': np.sqrt(mean_squared_error(y_dow, y_pred_dow)),
                    'mae': mean_absolute_error(y_dow, y_pred_dow)
                }
                performance_by_dow.append(metrics)
        
        # Create and save DataFrame
        dow_df = pd.DataFrame(performance_by_dow)
        dow_df.to_csv(os.path.join(output_dir, 'performance_by_day_of_week.csv'), index=False)
        
        # Plot performance by day of week
        plt.figure(figsize=(12, 6))
        plt.bar(dow_df['day_name'], dow_df['mae'])
        plt.title('Model Error by Day of Week')
        plt.ylabel('Mean Absolute Error')
        plt.xlabel('Day of Week')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_by_day_of_week.png'), dpi=300)
        plt.close()
        
        logger.info("Performance by day of week:")
        for _, row in dow_df.iterrows():
            logger.info(f"  - {row['day_name']}: MAE={row['mae']:.4f}, R²={row['r2_score']:.4f}")
    
    # 2. Performance by location capacity (binned)
    if 'capacity' in X.columns:
        # Create capacity bins
        X_with_y = X.copy()
        X_with_y['occupancy'] = y
        X_with_y['prediction'] = model.predict(X)
        X_with_y['abs_error'] = np.abs(X_with_y['occupancy'] - X_with_y['prediction'])
        
        # Create capacity bins
        X_with_y['capacity_bin'] = pd.cut(
            X_with_y['capacity'], 
            bins=[0, 50, 100, 200, 500, float('inf')],
            labels=['1-50', '51-100', '101-200', '201-500', '500+']
        )
        
        # Calculate metrics by capacity bin
        capacity_perf = X_with_y.groupby('capacity_bin').agg({
            'abs_error': ['mean', 'median', 'count'],
            'capacity': 'mean'
        })
        
        # Flatten the column hierarchy
        capacity_perf.columns = ['mae', 'median_ae', 'count', 'avg_capacity']
        
        # Save to CSV
        capacity_perf.reset_index().to_csv(
            os.path.join(output_dir, 'performance_by_capacity.csv'), 
            index=False
        )
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.bar(capacity_perf.index, capacity_perf['mae'])
        plt.title('Model Error by Parking Capacity')
        plt.ylabel('Mean Absolute Error')
        plt.xlabel('Capacity Bin')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_by_capacity.png'), dpi=300)
        plt.close()
        
        logger.info("Performance by capacity bin:")
        for bin_name, row in capacity_perf.iterrows():
            logger.info(f"  - {bin_name}: MAE={row['mae']:.4f}, Count={int(row['count'])}")
    
    # 3. Performance by time of year (month)
    if 'month' in X.columns:
        performance_by_month = []
        for month in range(1, 13):
            # Filter data for this month
            mask = X['month'] == month
            if mask.sum() > 0:
                X_month = X[mask]
                y_month = y[mask]
                
                # Make predictions
                y_pred_month = model.predict(X_month)
                
                # Calculate metrics
                metrics = {
                    'month': month,
                    'month_name': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1],
                    'count': mask.sum(),
                    'r2_score': r2_score(y_month, y_pred_month),
                    'rmse': np.sqrt(mean_squared_error(y_month, y_pred_month)),
                    'mae': mean_absolute_error(y_month, y_pred_month)
                }
                performance_by_month.append(metrics)
        
        # Create and save DataFrame
        month_df = pd.DataFrame(performance_by_month)
        month_df.to_csv(os.path.join(output_dir, 'performance_by_month.csv'), index=False)
        
        # Plot performance by month
        plt.figure(figsize=(12, 6))
        plt.bar(month_df['month_name'], month_df['mae'])
        plt.title('Model Error by Month')
        plt.ylabel('Mean Absolute Error')
        plt.xlabel('Month')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_by_month.png'), dpi=300)
        plt.close()
        
        logger.info("Performance by month:")
        for _, row in month_df.iterrows():
            logger.info(f"  - {row['month_name']}: MAE={row['mae']:.4f}, R²={row['r2_score']:.4f}")
    
    logger.info("Model robustness analysis completed")

def create_summary_report(
    model: Any,
    metrics: Dict[str, float],
    output_dir: str
) -> None:
    """
    Create a summary report of the model analysis.
    
    Args:
        model: The trained model
        metrics: Dictionary of performance metrics
        output_dir: Directory to save the report
    """
    logger.info("Creating summary report...")
    
    report_path = os.path.join(output_dir, 'model_analysis_summary.md')
    
    with open(report_path, 'w') as f:
        f.write("# Random Forest Model Analysis Summary\n\n")
        
        # Model information
        f.write("## Model Information\n\n")
        if hasattr(model, 'n_estimators'):
            f.write(f"- **Model Type:** Random Forest\n")
            f.write(f"- **Number of Trees:** {model.n_estimators}\n")
            f.write(f"- **Max Depth:** {model.max_depth if hasattr(model, 'max_depth') else 'None'}\n")
            f.write(f"- **Features:** {model.n_features_in_}\n\n")
        else:
            f.write(f"- **Model Type:** {type(model).__name__}\n\n")
        
        # Performance metrics
        if metrics:
            f.write("## Performance Metrics\n\n")
            f.write(f"- **R² Score:** {metrics.get('r2_score', 'N/A'):.4f}\n")
            f.write(f"- **Mean Squared Error:** {metrics.get('mse', 'N/A'):.4f}\n")
            f.write(f"- **Root Mean Squared Error:** {metrics.get('rmse', 'N/A'):.4f}\n")
            f.write(f"- **Mean Absolute Error:** {metrics.get('mae', 'N/A'):.4f}\n\n")
        
        # Feature importance
        f.write("## Feature Importance\n\n")
        f.write("See feature importance visualizations in the output directory:\n")
        f.write("- `feature_importance_mdi.png`\n")
        f.write("- `feature_importance_permutation.png`\n\n")
        
        # Visualizations
        f.write("## Visualizations\n\n")
        f.write("The analysis has generated the following visualizations:\n\n")
        f.write("### Performance Visualizations\n")
        f.write("- `predicted_vs_actual.png`: Scatter plot of predicted vs actual values\n")
        f.write("- `residuals.png`: Residual plot\n")
        f.write("- `residual_distribution.png`: Distribution of residuals\n\n")
        
        f.write("### Error Analysis\n")
        f.write("- `error_by_location.png`: Mean absolute error by location\n")
        f.write("- `error_by_hour.png`: Mean absolute error by hour of day\n")
        f.write("- `error_by_day_of_week.png`: Mean absolute error by day of week\n")
        f.write("- `error_by_month.png`: Mean absolute error by month\n")
        f.write("- `error_by_capacity.png`: Mean absolute error by capacity bin\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        f.write("Based on the analysis, consider the following next steps:\n\n")
        f.write("1. **Feature Engineering:** Focus on improving the most important features\n")
        f.write("2. **Model Tuning:** Further tune hyperparameters for better performance\n")
        f.write("3. **Data Collection:** Collect more data for underrepresented conditions\n")
        f.write("4. **Error Analysis:** Investigate high-error locations and time periods\n")
    
    logger.info(f"Summary report created at {report_path}")

def main(model_path: str, data_path: str, output_dir: str):
    """
    Main function to run the model analysis.
    
    Args:
        model_path: Path to the trained model file
        data_path: Path to the test data CSV file
        output_dir: Directory to save the analysis outputs
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Load the model
        model = load_model(model_path)
        
        # 2. Load and prepare the data
        X, y, feature_names = load_and_prepare_data(data_path)
        
        # 3. Analyze feature importance
        analyze_feature_importance(model, X, y, feature_names, output_dir)
        
        # 4. Evaluate model performance
        metrics = evaluate_model_performance(model, X, y, output_dir)
        
        # 5. Analyze model robustness
        analyze_model_robustness(model, X, y, output_dir)
        
        # 6. Create summary report
        create_summary_report(model, metrics, output_dir)
        
        logger.info(f"Model analysis completed. Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during model analysis: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Random Forest model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model pickle file"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the test data CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_analysis_results/",
        help="Directory to save the analysis outputs"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/",
        help="Default directory to look for models if --model is not a full path"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/splits_full/",
        help="Default directory to look for data if --data is not a full path"
    )
    
    args = parser.parse_args()
    
    # Handle paths
    model_path = args.model
    if not os.path.dirname(model_path):
        model_path = os.path.join(args.model_dir, model_path)
    
    data_path = args.data
    if not os.path.dirname(data_path):
        data_path = os.path.join(args.data_dir, data_path)
    
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    main(model_path, data_path, output_dir) 