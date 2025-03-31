#!/usr/bin/env python
"""
Test script for statistical tests module.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from .statistical_tests import StatisticalTests

# Set up logging
logger = logging.getLogger(__name__)

def generate_test_data():
    """Generate synthetic data for testing."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Continuous data
    n_samples = 1000
    continuous_data = {
        # Similar distributions
        'normal_similar': {
            'dist1': np.random.normal(0, 1, n_samples),
            'dist2': np.random.normal(0.1, 1.1, n_samples)
        },
        # Different distributions
        'normal_different': {
            'dist1': np.random.normal(0, 1, n_samples),
            'dist2': np.random.normal(2, 1.5, n_samples)
        }
    }
    
    # Categorical data
    categories = ['A', 'B', 'C']
    categorical_data = {
        # Similar distributions
        'categorical_similar': {
            'dist1': pd.Series(np.random.choice(categories, n_samples, p=[0.3, 0.4, 0.3])),
            'dist2': pd.Series(np.random.choice(categories, n_samples, p=[0.32, 0.38, 0.3]))
        },
        # Different distributions
        'categorical_different': {
            'dist1': pd.Series(np.random.choice(categories, n_samples, p=[0.8, 0.1, 0.1])),
            'dist2': pd.Series(np.random.choice(categories, n_samples, p=[0.2, 0.4, 0.4]))
        }
    }
    
    return continuous_data, categorical_data

def test_continuous_features(tests, continuous_data):
    """Test statistical tests for continuous features."""
    logger.info("\nTesting continuous features...")
    
    for feature_name, data in continuous_data.items():
        logger.info(f"\nTesting {feature_name}:")
        dist1 = pd.Series(data['dist1'])
        dist2 = pd.Series(data['dist2'])
        
        # Run all applicable tests
        results = tests.run_all_tests(dist1, dist2, feature_name, is_categorical=False)
        
        # Log results
        for test_name, result in results.items():
            logger.info(f"\n{test_name}:")
            logger.info(f"Statistic: {result.statistic:.4f}")
            if result.p_value is not None:
                logger.info(f"P-value: {result.p_value:.4f}")
            logger.info(f"Significant: {result.is_significant}")
            
            if test_name == 'psi':
                stability = result.details['interpretation']['stability_assessment']
                logger.info(f"Stability Assessment: {stability}")

def test_categorical_features(tests, categorical_data):
    """Test statistical tests for categorical features."""
    logger.info("\nTesting categorical features...")
    
    for feature_name, data in categorical_data.items():
        logger.info(f"\nTesting {feature_name}:")
        dist1 = data['dist1']
        dist2 = data['dist2']
        
        # Run all applicable tests
        results = tests.run_all_tests(dist1, dist2, feature_name, is_categorical=True)
        
        # Log results
        for test_name, result in results.items():
            logger.info(f"\n{test_name}:")
            logger.info(f"Statistic: {result.statistic:.4f}")
            if result.p_value is not None:
                logger.info(f"P-value: {result.p_value:.4f}")
            logger.info(f"Significant: {result.is_significant}")
            
            if test_name == 'psi':
                stability = result.details['interpretation']['stability_assessment']
                logger.info(f"Stability Assessment: {stability}")

def main():
    """Main function to run tests."""
    logger.info("Starting statistical tests validation...")
    
    # Initialize statistical tests
    tests = StatisticalTests()
    
    # Generate test data
    continuous_data, categorical_data = generate_test_data()
    
    # Run tests
    test_continuous_features(tests, continuous_data)
    test_categorical_features(tests, categorical_data)
    
    logger.info("\nStatistical tests validation completed!")

if __name__ == "__main__":
    main() 