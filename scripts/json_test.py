#!/usr/bin/env python
"""
Test script to identify and fix JSON serialization issues with pandas Timestamp objects.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

def fix_categorical_stats(categorical_stats):
    """Convert any pandas Timestamp objects to strings in categorical stats."""
    fixed_stats = {}
    for col, stats in categorical_stats.items():
        fixed_col_stats = {}
        for category, value in stats.items():
            # Convert Timestamp objects to strings
            if hasattr(category, 'isoformat'):  # Check if it has isoformat method
                category_key = category.isoformat()
            else:
                category_key = str(category)
            fixed_col_stats[category_key] = value
        fixed_stats[col] = fixed_col_stats
    return fixed_stats

def main():
    """Run test for JSON serialization."""
    print("Creating test data...")
    
    # Create sample data with timestamp column
    dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
    df = pd.DataFrame({
        'timestamp': dates,
        'location_id': ['A1', 'B2', 'C3', 'D4', 'E5'],
        'temperature': [25.0, 26.5, 24.0, 22.5, 27.0],
        'humidity': [60, 65, 55, 70, 50]
    })
    
    print("Sample dataframe:")
    print(df.head())
    
    # Extract numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns
    
    print(f"Numeric columns: {list(numeric_cols)}")
    print(f"Categorical columns: {list(categorical_cols)}")
    
    # Compute basic statistics
    numeric_data = df[numeric_cols]
    stats = {
        'mean': numeric_data.mean(),
        'std': numeric_data.std(),
        'quantiles': numeric_data.quantile([0.25, 0.5, 0.75]),
        'timestamp': datetime.now().isoformat(),
        'categorical_cols': list(categorical_cols),
        'numeric_cols': list(numeric_cols)
    }
    
    # Compute category statistics
    cat_stats = {}
    for col in categorical_cols:
        try:
            value_counts = df[col].value_counts(normalize=True).to_dict()
            cat_stats[col] = value_counts
        except Exception as e:
            print(f"Error computing value counts for {col}: {e}")
    
    stats['categorical_stats'] = cat_stats
    
    print("\nAttempting to prepare data for JSON serialization...")
    
    try:
        # Prepare for JSON serialization
        stats_for_json = {
            'mean': stats['mean'].to_dict(),
            'std': stats['std'].to_dict(),
            'quantiles': {},
            'timestamp': stats['timestamp'],
            'categorical_cols': list(map(str, stats['categorical_cols'])),
            'numeric_cols': list(map(str, stats['numeric_cols']))
        }
        
        # Convert quantiles DataFrame to nested dict with string keys
        for idx_name in stats['quantiles'].index:
            idx_key = str(float(idx_name))
            stats_for_json['quantiles'][idx_key] = stats['quantiles'].loc[idx_name].to_dict()
            
        # Fix categorical stats to use string keys
        stats_for_json['categorical_stats'] = fix_categorical_stats(stats['categorical_stats'])
        
        print("\nJSON-ready dictionary created successfully.")
        print("Sample of the JSON-ready dictionary:")
        print(json.dumps(stats_for_json, indent=2)[:500] + "...")
        
        # Try serializing the whole thing
        json_str = json.dumps(stats_for_json)
        print("\nSuccessfully serialized to JSON.")
        
    except Exception as e:
        print(f"\nError during JSON serialization: {e}")
        # Print the types of keys in categorical_stats to help diagnose
        print("\nDiagnostic information about categorical_stats:")
        for col, counts in stats['categorical_stats'].items():
            print(f"Column: {col}")
            for key, value in counts.items():
                print(f"  Key: {key} (type: {type(key).__name__}), Value: {value}")

if __name__ == "__main__":
    main() 