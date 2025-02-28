#!/usr/bin/env python
"""
Script to analyze project structure and identify potential data leakage.
"""

import os
import sys
import importlib
import inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_module_files(directory):
    """Find all Python module files in a directory and its subdirectories."""
    module_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                module_files.append(os.path.join(root, file))
    return module_files

def search_for_feature_engineering(module_files):
    """Search for feature engineering code in module files."""
    feature_engineering_files = []
    
    for file_path in module_files:
        with open(file_path, 'r') as f:
            content = f.read()
            if 'feature' in content.lower() and ('engineer' in content.lower() or 'transform' in content.lower()):
                feature_engineering_files.append(file_path)
    
    return feature_engineering_files

def analyze_data_with_basic_features(data_path):
    """Analyze data with basic features."""
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Convert timestamp to datetime if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
    
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Check for high correlation with target
    if 'occupancy' in df.columns:
        correlations = {}
        for col in df.columns:
            if col != 'occupancy' and df[col].dtype in [np.int64, np.float64]:
                correlations[col] = df[col].corr(df['occupancy'])
        
        sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        logger.info("Top 10 features with highest correlation to target:")
        for col, corr in sorted_correlations[:10]:
            logger.info(f"{col}: {corr:.4f}")
    
    # Test for time-based performance
    if 'timestamp' in df.columns and 'occupancy' in df.columns:
        logger.info("Testing model performance with time-based split...")
        
        # Prepare features and target
        exclude_cols = ['timestamp', 'date', 'occupancy']
        X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        y = df['occupancy']
        
        # Create time-based split
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Get the last split
        for train_idx, test_idx in tscv.split(X):
            pass
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)
        
        logger.info(f"Train R²: {train_r2:.4f}")
        logger.info(f"Test R²: {test_r2:.4f}")
        logger.info(f"R² difference: {train_r2 - test_r2:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 most important features:")
        for i, row in feature_importance.head(10).iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")

def analyze_file_content(file_path):
    """Analyze the content of a file for potential data leakage patterns."""
    logger.info(f"Analyzing file: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
        
    # Look for suspicious patterns
    suspicious_patterns = [
        'groupby', 'rolling', 'expanding', 'shift(-', 
        'lead', 'future', 'next', 'ahead', 'transform'
    ]
    
    for pattern in suspicious_patterns:
        if pattern in content:
            logger.warning(f"Potential leakage pattern found: '{pattern}'")
            
            # Show the lines containing the pattern
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if pattern in line:
                    logger.warning(f"Line {i+1}: {line.strip()}")

def main():
    """Main function to analyze project and data."""
    logger.info("Starting project analysis...")
    
    # Find all Python module files
    module_files = find_module_files('.')
    logger.info(f"Found {len(module_files)} Python files in the project")
    
    # Search for feature engineering code
    feature_files = search_for_feature_engineering(module_files)
    logger.info(f"Found {len(feature_files)} files with potential feature engineering code:")
    for file in feature_files:
        logger.info(f"  - {file}")
    
    # Analyze each feature engineering file
    for file in feature_files:
        analyze_file_content(file)
    
    # Analyze data with basic features
    if os.path.exists('data/prepared_data_improved.csv'):
        analyze_data_with_basic_features('data/prepared_data_improved.csv')
    else:
        logger.warning("Could not find data/prepared_data_improved.csv")
        # Try to find other CSV files
        csv_files = []
        for root, dirs, files in os.walk('data'):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if csv_files:
            logger.info(f"Found {len(csv_files)} CSV files:")
            for file in csv_files:
                logger.info(f"  - {file}")
            
            # Analyze the first CSV file
            analyze_data_with_basic_features(csv_files[0])
        else:
            logger.warning("No CSV files found in the data directory")

if __name__ == "__main__":
    main()