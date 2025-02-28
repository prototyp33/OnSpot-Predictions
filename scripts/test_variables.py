#!/usr/bin/env python
"""
Script for testing the impact of different variables on parking occupancy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import logging
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory
output_dir = "variable_test_results"
os.makedirs(output_dir, exist_ok=True)

def load_data(file_path):
    """Load and prepare the dataset for testing."""
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract time components
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'] >= 5
    
    logger.info(f"Dataset loaded with shape: {df.shape}")
    return df

def identify_variable_types(df):
    """Identify numerical and categorical variables in the dataset."""
    # Exclude target variable and timestamp
    exclude_cols = ['occupancy', 'timestamp', 'date']
    
    # Identify numerical and categorical columns
    numerical_cols = []
    categorical_cols = []
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        
        if np.issubdtype(df[col].dtype, np.number):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
    
    logger.info(f"Identified {len(numerical_cols)} numerical variables: {numerical_cols}")
    logger.info(f"Identified {len(categorical_cols)} categorical variables: {categorical_cols}")
    
    return numerical_cols, categorical_cols

def test_individual_variables(df, numerical_cols, categorical_cols):
    """Test the impact of each variable individually."""
    logger.info("Testing individual variables...")
    
    results = []
    
    # Target variable
    y = df['occupancy']
    
    # Test each numerical variable
    for col in numerical_cols:
        X = df[[col]]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'variable': col,
            'type': 'numerical',
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'coefficient': model.coef_[0]
        })
    
    # Test each categorical variable
    for col in categorical_cols:
        # One-hot encode
        X = pd.get_dummies(df[col], prefix=col)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'variable': col,
            'type': 'categorical',
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'coefficient': 'multiple'
        })
    
    # Convert to DataFrame and sort by R²
    results_df = pd.DataFrame(results).sort_values('r2', ascending=False)
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, 'individual_variable_results.csv'), index=False)
    
    # Plot results
    plt.figure(figsize=(14, 8))
    sns.barplot(x='variable', y='r2', data=results_df)
    plt.title('R² Score by Individual Variable')
    plt.xlabel('Variable')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'individual_variable_r2.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Individual variable testing completed.")
    return results_df

def test_variable_groups(df, numerical_cols, categorical_cols):
    """Test the impact of different groups of variables."""
    logger.info("Testing variable groups...")
    
    # Target variable
    y = df['occupancy']
    
    # Define variable groups
    groups = {
        'time': ['hour', 'day_of_week', 'month', 'is_weekend'],
        'weather': [col for col in numerical_cols if col in ['temperature', 'humidity', 'wind_speed', 'precipitation']],
        'location': [col for col in df.columns if 'location' in col.lower() or 'zone' in col.lower()],
        'traffic': [col for col in numerical_cols if 'traffic' in col.lower()],
        'all_numerical': numerical_cols,
        'all_categorical': categorical_cols,
        'all_variables': numerical_cols + categorical_cols
    }
    
    # Filter out empty groups
    groups = {k: v for k, v in groups.items() if v}
    
    results = []
    
    for group_name, group_cols in groups.items():
        logger.info(f"Testing group: {group_name} with {len(group_cols)} variables")
        
        # Prepare data
        if all(col in numerical_cols for col in group_cols):
            # All numerical
            X = df[group_cols]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Feature importance
            importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': group_cols,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Save feature importance
            importance_df.to_csv(os.path.join(output_dir, f'importance_{group_name}.csv'), index=False)
            
            # Plot feature importance
            plt.figure(figsize=(12, 6))
            sns.barplot(x='feature', y='importance', data=importance_df)
            plt.title(f'Feature Importance - {group_name}')
            plt.xlabel('Feature')
            plt.ylabel('Importance')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'importance_{group_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            # Mixed or categorical
            numerical_in_group = [col for col in group_cols if col in numerical_cols]
            categorical_in_group = [col for col in group_cols if col in categorical_cols]
            
            # Create preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_in_group),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_in_group)
                ]
            )
            
            # Create pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            
            # Split data
            X = df[group_cols]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            y_pred = pipeline.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
        
        results.append({
            'group': group_name,
            'variables': ', '.join(group_cols),
            'num_variables': len(group_cols),
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2
        })
    
    # Convert to DataFrame and sort by R²
    results_df = pd.DataFrame(results).sort_values('r2', ascending=False)
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, 'variable_group_results.csv'), index=False)
    
    # Plot results
    plt.figure(figsize=(14, 8))
    sns.barplot(x='group', y='r2', data=results_df)
    plt.title('R² Score by Variable Group')
    plt.xlabel('Variable Group')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variable_group_r2.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Variable group testing completed.")
    return results_df

def test_location_specific_variables(df):
    """Test which variables are most important for each location."""
    logger.info("Testing location-specific variables...")
    
    # Check if location_id exists
    if 'location_id' not in df.columns:
        logger.warning("No location_id column found. Skipping location-specific analysis.")
        return None
    
    # Get all locations
    locations = df['location_id'].unique()
    
    # Identify variables
    numerical_cols, categorical_cols = identify_variable_types(df)
    all_vars = numerical_cols + categorical_cols
    
    # Remove location_id from variables
    if 'location_id' in all_vars:
        all_vars.remove('location_id')
    
    # Store results
    location_results = []
    
    for location in locations:
        logger.info(f"Analyzing location: {location}")
        
        # Filter data for this location
        loc_df = df[df['location_id'] == location]
        
        # Target variable
        y = loc_df['occupancy']
        
        # Features
        X = loc_df[all_vars]
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        except ValueError:
            logger.warning(f"Not enough data for location {location}. Skipping.")
            continue
        
        # Create preprocessing pipeline
        numerical_in_group = [col for col in all_vars if col in numerical_cols]
        categorical_in_group = [col for col in all_vars if col in categorical_cols]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_in_group),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_in_group)
            ]
        )
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Get feature importance
        # This is complex with pipelines, so we'll train a separate model on preprocessed data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_processed, y_train)
        
        # Get feature names after preprocessing
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat':
                for col in cols:
                    feature_names.extend([f"{col}_{cat}" for cat in loc_df[col].unique()])
        
        # Get importance
        importance = model.feature_importances_
        
        # Limit to top 10 features
        if len(feature_names) > 10:
            top_indices = np.argsort(importance)[-10:]
            top_features = [feature_names[i] for i in top_indices if i < len(feature_names)]
            top_importance = [importance[i] for i in top_indices if i < len(feature_names)]
        else:
            top_features = feature_names
            top_importance = importance
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        plt.barh(range(len(top_features)), top_importance, align='center')
        plt.yticks(range(len(top_features)), top_features)
        plt.title(f'Top Features - Location {location}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'importance_location_{location}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        location_results.append({
            'location': location,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'top_features': ', '.join(top_features[:5])
        })
    
    # Convert to DataFrame
    location_results_df = pd.DataFrame(location_results).sort_values('r2', ascending=False)
    
    # Save results
    location_results_df.to_csv(os.path.join(output_dir, 'location_specific_results.csv'), index=False)
    
    # Plot results
    plt.figure(figsize=(14, 8))
    sns.barplot(x='location', y='r2', data=location_results_df)
    plt.title('R² Score by Location')
    plt.xlabel('Location')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'location_r2.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Location-specific testing completed.")
    return location_results_df

def main(data_path):
    """Main function to run the variable testing."""
    logger.info("Starting variable testing...")
    
    # Load data
    df = load_data(data_path)
    
    # Identify variable types
    numerical_cols, categorical_cols = identify_variable_types(df)
    
    # Run tests
    individual_results = test_individual_variables(df, numerical_cols, categorical_cols)
    group_results = test_variable_groups(df, numerical_cols, categorical_cols)
    location_results = test_location_specific_variables(df)
    
    # Create summary report
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
        f.write("=== VARIABLE TESTING SUMMARY ===\n\n")
        
        f.write("Top 5 Individual Variables:\n")
        for _, row in individual_results.head(5).iterrows():
            f.write(f"- {row['variable']}: R² = {row['r2']:.4f}, RMSE = {row['rmse']:.4f}\n")
        
        f.write("\nTop Variable Groups:\n")
        for _, row in group_results.head(3).iterrows():
            f.write(f"- {row['group']}: R² = {row['r2']:.4f}, RMSE = {row['rmse']:.4f}\n")
        
        if location_results is not None:
            f.write("\nLocation-Specific Results:\n")
            for _, row in location_results.head(5).iterrows():
                f.write(f"- Location {row['location']}: R² = {row['r2']:.4f}, Top features: {row['top_features']}\n")
    
    logger.info(f"Variable testing completed. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the impact of variables on parking occupancy")
    parser.add_argument("--data", default="data/prepared_data_improved.csv", help="Path to the prepared data file")
    
    args = parser.parse_args()
    main(args.data) 