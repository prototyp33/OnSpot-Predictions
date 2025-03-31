"""
Test script for the visualization module.

This script generates synthetic data to test the visualization capabilities
for both continuous and categorical distributions.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from .visualization import DistributionVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_test_data(n_samples: int = 1000, random_seed: int = 42) -> dict:
    """
    Generate synthetic data for testing visualizations.
    
    Args:
        n_samples (int): Number of samples to generate
        random_seed (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing test data distributions
    """
    np.random.seed(random_seed)
    
    # 1. Continuous distributions
    # Similar normal distributions
    dist1_similar = np.random.normal(loc=0, scale=1, size=n_samples)
    dist2_similar = np.random.normal(loc=0.1, scale=1.1, size=n_samples)
    
    # Different normal distributions
    dist1_different = np.random.normal(loc=0, scale=1, size=n_samples)
    dist2_different = np.random.normal(loc=2, scale=2, size=n_samples)
    
    # 2. Categorical distributions
    categories = ['A', 'B', 'C']
    
    # Similar categorical distributions
    probs_similar1 = [0.3, 0.4, 0.3]
    probs_similar2 = [0.32, 0.38, 0.3]
    cat_dist1_similar = np.random.choice(categories, size=n_samples, p=probs_similar1)
    cat_dist2_similar = np.random.choice(categories, size=n_samples, p=probs_similar2)
    
    # Different categorical distributions
    probs_different1 = [0.2, 0.3, 0.5]
    probs_different2 = [0.6, 0.2, 0.2]
    cat_dist1_different = np.random.choice(categories, size=n_samples, p=probs_different1)
    cat_dist2_different = np.random.choice(categories, size=n_samples, p=probs_different2)
    
    return {
        'continuous': {
            'similar': {
                'dist1': pd.Series(dist1_similar, name='normal_similar_1'),
                'dist2': pd.Series(dist2_similar, name='normal_similar_2')
            },
            'different': {
                'dist1': pd.Series(dist1_different, name='normal_different_1'),
                'dist2': pd.Series(dist2_different, name='normal_different_2')
            }
        },
        'categorical': {
            'similar': {
                'dist1': pd.Series(cat_dist1_similar, name='cat_similar_1'),
                'dist2': pd.Series(cat_dist2_similar, name='cat_similar_2')
            },
            'different': {
                'dist1': pd.Series(cat_dist1_different, name='cat_different_1'),
                'dist2': pd.Series(cat_dist2_different, name='cat_different_2')
            }
        }
    }

def test_continuous_visualization(visualizer: DistributionVisualizer, test_data: dict):
    """
    Test visualization of continuous distributions.
    
    Args:
        visualizer (DistributionVisualizer): Instance of DistributionVisualizer
        test_data (dict): Dictionary containing test data
    """
    logger.info("Testing continuous distribution visualizations...")
    
    # Test similar distributions
    similar_dists = test_data['continuous']['similar']
    plot_paths = visualizer.plot_continuous_distribution(
        similar_dists,
        'Similar Normal Distributions'
    )
    logger.info("Generated plots for similar continuous distributions:")
    for plot_type, path in plot_paths.items():
        logger.info(f"- {plot_type}: {path}")
    
    # Test different distributions
    different_dists = test_data['continuous']['different']
    plot_paths = visualizer.plot_continuous_distribution(
        different_dists,
        'Different Normal Distributions'
    )
    logger.info("Generated plots for different continuous distributions:")
    for plot_type, path in plot_paths.items():
        logger.info(f"- {plot_type}: {path}")

def test_categorical_visualization(visualizer: DistributionVisualizer, test_data: dict):
    """
    Test visualization of categorical distributions.
    
    Args:
        visualizer (DistributionVisualizer): Instance of DistributionVisualizer
        test_data (dict): Dictionary containing test data
    """
    logger.info("Testing categorical distribution visualizations...")
    
    # Test similar distributions
    similar_dists = test_data['categorical']['similar']
    plot_paths = visualizer.plot_categorical_distribution(
        similar_dists,
        'Similar Categorical Distributions'
    )
    logger.info("Generated plots for similar categorical distributions:")
    for plot_type, path in plot_paths.items():
        logger.info(f"- {plot_type}: {path}")
    
    # Test different distributions
    different_dists = test_data['categorical']['different']
    plot_paths = visualizer.plot_categorical_distribution(
        different_dists,
        'Different Categorical Distributions'
    )
    logger.info("Generated plots for different categorical distributions:")
    for plot_type, path in plot_paths.items():
        logger.info(f"- {plot_type}: {path}")

def verify_plot_files(output_dir: Path):
    """
    Verify that all expected plot files were created.
    
    Args:
        output_dir (Path): Directory where plots should be saved
    """
    logger.info("Verifying plot files...")
    
    expected_continuous_plots = [
        'Similar Normal Distributions_histogram.png',
        'Similar Normal Distributions_boxplot.png',
        'Similar Normal Distributions_qq.png',
        'Different Normal Distributions_histogram.png',
        'Different Normal Distributions_boxplot.png',
        'Different Normal Distributions_qq.png'
    ]
    
    expected_categorical_plots = [
        'Similar Categorical Distributions_barplot.png',
        'Similar Categorical Distributions_piechart.png',
        'Different Categorical Distributions_barplot.png',
        'Different Categorical Distributions_piechart.png'
    ]
    
    all_expected_plots = expected_continuous_plots + expected_categorical_plots
    missing_plots = []
    
    for plot_file in all_expected_plots:
        plot_path = output_dir / plot_file
        if not plot_path.exists():
            missing_plots.append(plot_file)
    
    if missing_plots:
        logger.error(f"Missing plot files: {missing_plots}")
        raise FileNotFoundError(f"Some expected plot files were not created: {missing_plots}")
    else:
        logger.info("All expected plot files were created successfully")

def main():
    """Main function to run visualization tests."""
    logger.info("Starting visualization tests...")
    
    # Initialize visualizer
    visualizer = DistributionVisualizer()
    
    # Generate test data
    test_data = generate_test_data()
    
    try:
        # Test continuous distributions
        test_continuous_visualization(visualizer, test_data)
        
        # Test categorical distributions
        test_categorical_visualization(visualizer, test_data)
        
        # Verify all plots were created
        verify_plot_files(visualizer.output_dir)
        
        logger.info("All visualization tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during visualization testing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 