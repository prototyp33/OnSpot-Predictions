"""
Test script for the distribution analyzer module.

This script verifies the functionality of the DistributionAnalyzer class
using synthetic data that simulates real-world scenarios.
"""

import numpy as np
import pandas as pd
import logging
import os
import yaml
import shutil
from pathlib import Path
from typing import Dict, List
from . import ValidationConfig
from .analyzer import DistributionAnalyzer, AnalysisResult

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_config() -> str:
    """Create a test configuration with all required settings."""
    # Create test directories
    test_dir = Path('tests/validation')
    test_dir.mkdir(parents=True, exist_ok=True)
    
    splits_dir = test_dir / 'splits'
    splits_dir.mkdir(exist_ok=True)
    
    output_dir = test_dir / 'output'
    output_dir.mkdir(exist_ok=True)
    
    config = {
        'data': {
            'splits_dir': str(splits_dir)
        },
        'analysis': {
            'statistical_tests': {
                'ks_test': {
                    'enabled': True,
                    'significance_level': 0.05
                },
                'chi_squared': {
                    'enabled': True,
                    'significance_level': 0.05
                },
                'mann_whitney': {
                    'enabled': True,
                    'significance_level': 0.05
                },
                'psi': {
                    'enabled': True,
                    'threshold': 0.2,
                    'bins': 10
                }
            }
        },
        'thresholds': {
            'missing_rate': 0.1,
            'correlation': 0.8,
            'variance_ratio': 2.0,
            'category_ratio': 1.5,
            'psi_threshold': 0.2,
            'effect_size': 0.5
        },
        'output': {
            'base_dir': str(output_dir),
            'log_level': 'INFO',
            'subdirs': {
                'analysis': 'analysis_results',
                'plots': 'plots',
                'logs': 'logs',
                'metrics': 'metrics'
            }
        },
        'visualization': {
            'style': 'seaborn',
            'figsize': [10, 6],
            'dpi': 100
        }
    }
    
    # Save config to file
    config_path = test_dir / 'test_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
        
    return str(config_path)

def cleanup_test_files():
    """Clean up test files and directories."""
    test_dir = Path('tests/validation')
    if test_dir.exists():
        shutil.rmtree(test_dir)
        logger.info("Cleaned up test files")

def generate_test_data(n_samples: int = 1000, random_seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic dataset splits for testing.
    
    Args:
        n_samples (int): Number of samples per split
        random_seed (int): Random seed for reproducibility
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of dataset splits
    """
    np.random.seed(random_seed)
    
    # Generate continuous features
    reference_data = {
        # Normal distribution (stable)
        'stable_continuous': np.random.normal(0, 1, n_samples),
        # Normal distribution (drift)
        'drift_continuous': np.random.normal(0, 1, n_samples),
        # Exponential distribution (stable)
        'stable_exponential': np.random.exponential(2, n_samples),
        # Exponential distribution (drift)
        'drift_exponential': np.random.exponential(2, n_samples)
    }
    
    comparison_data = {
        'stable_continuous': np.random.normal(0, 1, n_samples),
        'drift_continuous': np.random.normal(1, 2, n_samples),  # Different mean and std
        'stable_exponential': np.random.exponential(2, n_samples),
        'drift_exponential': np.random.exponential(4, n_samples)  # Different rate
    }
    
    # Generate categorical features
    categories = ['A', 'B', 'C', 'D']
    
    # Stable categorical (similar proportions)
    reference_data['stable_categorical'] = np.random.choice(
        categories,
        size=n_samples,
        p=[0.3, 0.3, 0.2, 0.2]
    )
    comparison_data['stable_categorical'] = np.random.choice(
        categories,
        size=n_samples,
        p=[0.28, 0.32, 0.21, 0.19]
    )
    
    # Drifting categorical (different proportions)
    reference_data['drift_categorical'] = np.random.choice(
        categories,
        size=n_samples,
        p=[0.4, 0.3, 0.2, 0.1]
    )
    comparison_data['drift_categorical'] = np.random.choice(
        categories,
        size=n_samples,
        p=[0.1, 0.2, 0.3, 0.4]  # Reversed proportions
    )
    
    # Create DataFrames
    reference_df = pd.DataFrame(reference_data)
    comparison_df = pd.DataFrame(comparison_data)
    
    # Add some missing values
    for df in [reference_df, comparison_df]:
        mask = np.random.random(df.shape) < 0.05  # 5% missing values
        df[mask] = np.nan
    
    return {
        'reference': reference_df,
        'comparison': comparison_df
    }

def test_feature_type_detection(analyzer: DistributionAnalyzer, data: pd.DataFrame):
    """
    Test the feature type detection logic.
    
    Args:
        analyzer (DistributionAnalyzer): Analyzer instance
        data (pd.DataFrame): Test data
    """
    logger.info("Testing feature type detection...")
    
    # Test continuous features
    for feature in ['stable_continuous', 'drift_continuous', 
                   'stable_exponential', 'drift_exponential']:
        is_cat = analyzer._is_categorical(data[feature])
        assert not is_cat, f"Expected {feature} to be continuous"
        logger.info(f"Correctly identified {feature} as continuous")
    
    # Test categorical features
    for feature in ['stable_categorical', 'drift_categorical']:
        is_cat = analyzer._is_categorical(data[feature])
        assert is_cat, f"Expected {feature} to be categorical"
        logger.info(f"Correctly identified {feature} as categorical")

def test_basic_stats(analyzer: DistributionAnalyzer, data_splits: Dict[str, pd.DataFrame]):
    """
    Test basic statistics calculation.
    
    Args:
        analyzer (DistributionAnalyzer): Analyzer instance
        data_splits (Dict[str, pd.DataFrame]): Test data splits
    """
    logger.info("Testing basic statistics calculation...")
    
    # Test continuous feature
    continuous_dists = {
        name: split['stable_continuous']
        for name, split in data_splits.items()
    }
    cont_stats = analyzer._get_basic_stats(continuous_dists)
    
    # Verify continuous statistics
    for split_stats in cont_stats.values():
        assert all(key in split_stats for key in ['mean', 'std', 'min', 'max', 'median'])
        logger.info("Continuous feature statistics verified")
    
    # Test categorical feature
    cat_dists = {
        name: split['stable_categorical']
        for name, split in data_splits.items()
    }
    cat_stats = analyzer._get_basic_stats(cat_dists)
    
    # Verify categorical statistics
    for split_stats in cat_stats.values():
        assert 'top_categories' in split_stats
        assert len(split_stats['top_categories']) <= 5
        logger.info("Categorical feature statistics verified")

def test_feature_analysis(analyzer: DistributionAnalyzer, data_splits: Dict[str, pd.DataFrame]):
    """
    Test single feature analysis.
    
    Args:
        analyzer (DistributionAnalyzer): Analyzer instance
        data_splits (Dict[str, pd.DataFrame]): Test data splits
    """
    logger.info("Testing single feature analysis...")
    
    # Test stable continuous feature
    stable_dists = {
        name: split['stable_continuous']
        for name, split in data_splits.items()
    }
    stable_result = analyzer.analyze_feature(stable_dists, 'stable_continuous')
    
    assert stable_result.feature_type == 'continuous'
    assert all(test.is_significant == False for test in stable_result.statistical_results.values())
    logger.info("Stable continuous feature analysis verified")
    
    # Test drifting continuous feature
    drift_dists = {
        name: split['drift_continuous']
        for name, split in data_splits.items()
    }
    drift_result = analyzer.analyze_feature(drift_dists, 'drift_continuous')
    
    assert drift_result.feature_type == 'continuous'
    assert any(test.is_significant for test in drift_result.statistical_results.values())
    logger.info("Drifting continuous feature analysis verified")

def test_batch_analysis(analyzer: DistributionAnalyzer, data_splits: Dict[str, pd.DataFrame]):
    """
    Test batch feature analysis and result serialization.
    
    Args:
        analyzer (DistributionAnalyzer): Analyzer instance
        data_splits (Dict[str, pd.DataFrame]): Test data splits
    """
    logger.info("Testing batch feature analysis...")
    
    # Analyze all features
    results = analyzer.analyze_features(data_splits)
    
    # Verify results
    assert len(results) == len(data_splits['reference'].columns)
    
    # Test result serialization
    try:
        # Test JSON serialization
        analyzer.save_analysis_results(results, output_format='json')
        json_files = list(analyzer.output_dir.glob('*.json'))
        assert len(json_files) > 0, "No JSON results file created"
        
        # Test YAML serialization
        analyzer.save_analysis_results(results, output_format='yaml')
        yaml_files = list(analyzer.output_dir.glob('*.yaml'))
        assert len(yaml_files) > 0, "No YAML results file created"
        
        logger.info("Result serialization tests passed")
    except Exception as e:
        logger.error(f"Result serialization failed: {str(e)}")
        raise

def main():
    """Run all tests."""
    try:
        # Clean up any existing test files
        cleanup_test_files()
        
        # Create test configuration
        config_path = create_test_config()
        config = ValidationConfig(config_path)
        
        # Generate test data
        data_splits = generate_test_data()
        
        # Save test data to splits directory
        splits_dir = Path(config.config['data']['splits_dir'])
        for split_name, df in data_splits.items():
            df.to_csv(splits_dir / f'{split_name}.csv', index=False)
        
        # Initialize analyzer
        analyzer = DistributionAnalyzer(config)
        
        # Run tests
        test_feature_type_detection(analyzer, data_splits['reference'])
        test_basic_stats(analyzer, data_splits)
        test_feature_analysis(analyzer, data_splits)
        test_batch_analysis(analyzer, data_splits)
        
        logger.info("All tests passed successfully!")
    except AssertionError as e:
        logger.error(f"Test failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during testing: {str(e)}")
        raise
    finally:
        # Clean up test files
        cleanup_test_files()

if __name__ == '__main__':
    main() 