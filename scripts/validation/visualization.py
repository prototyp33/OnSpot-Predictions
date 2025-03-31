"""
Visualization module for distribution comparison and drift detection.

This module provides visualization capabilities to complement statistical tests,
helping to visually identify and understand distribution differences between
dataset splits (train, validation, test).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import List, Dict, Optional, Union, Tuple

# Set up logging
logger = logging.getLogger(__name__)

class DistributionVisualizer:
    """
    Class for creating and saving distribution visualizations.
    """
    
    def __init__(self, config: 'ValidationConfig'):  # Forward reference
        """
        Initialize with configuration settings.
        
        Args:
            config: Validation configuration object
        """
        self.config = config
        self.viz_config = config.get_visualization_config()
        
        # Log available styles and current configuration
        logger.info(f"Available matplotlib styles: {plt.style.available}")
        logger.info(f"Current visualization config: {self.viz_config}")
        
        try:
            # First set seaborn defaults
            logger.info("Setting seaborn defaults...")
            sns.set_theme(style='whitegrid')
            
            # Then try to get the configured style
            style = self.viz_config.get('style', 'default')
            logger.info(f"Attempting to apply style: {style}")
            
            if style != 'default':
                plt.style.use(style)
            
            logger.info("Style application successful")
            
        except Exception as e:
            logger.error(f"Error applying style: {str(e)}")
            logger.warning("Falling back to default style")
            plt.style.use('default')
        
        # Apply custom style settings
        logger.info("Applying custom style settings...")
        plt.rcParams.update({
            'figure.figsize': self.viz_config.get('figsize', (10, 6)),
            'figure.dpi': self.viz_config.get('dpi', 100),
            'font.family': self.viz_config.get('font', {}).get('family', 'sans-serif'),
            'font.size': self.viz_config.get('font', {}).get('size', 10),
            'axes.titlesize': self.viz_config.get('font', {}).get('title_size', 12),
            'axes.grid': self.viz_config.get('grid', {}).get('show', True),
            'grid.alpha': self.viz_config.get('grid', {}).get('alpha', 0.3),
            'grid.linestyle': self.viz_config.get('grid', {}).get('linestyle', '--')
        })
        
        # Set up colors
        self.colors = self.viz_config.get('colors', {
            'reference': '#2ecc71',  # Green
            'comparison': '#3498db'   # Blue
        })
        
        # Create output directory if it doesn't exist
        self.output_dir = Path(config.get_output_dir()) / 'visualizations'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("DistributionVisualizer initialization complete")
    
    def plot_distributions(
        self,
        distributions: Dict[str, pd.Series],
        feature_name: str,
        is_categorical: bool,
        plot_types: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Create visualizations for feature distributions, handling both continuous and categorical features.
        
        Args:
            distributions: Dictionary of distributions to compare
            feature_name: Name of the feature being visualized
            is_categorical: Whether the feature is categorical
            plot_types: Optional list of plot types to generate
            
        Returns:
            Dictionary mapping plot types to their saved file paths (as strings)
        """
        if is_categorical:
            paths = self.plot_categorical_distribution(distributions, feature_name, plot_types)
        else:
            paths = self.plot_continuous_distribution(distributions, feature_name, plot_types)
        
        # Convert Path objects to strings for JSON serialization
        return {k: str(v) for k, v in paths.items()}
    
    def plot_continuous_distribution(
        self,
        distributions: Dict[str, pd.Series],
        feature_name: str,
        plot_types: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """
        Create visualizations for continuous feature distributions.
        
        Args:
            distributions (Dict[str, pd.Series]): Dictionary of distributions to compare
                (e.g., {'train': train_dist, 'validation': val_dist})
            feature_name (str): Name of the feature being visualized
            plot_types (Optional[List[str]]): Types of plots to generate
                (default: ['histogram', 'kde', 'box', 'qq'])
        
        Returns:
            Dict[str, Path]: Dictionary mapping plot types to their saved file paths
        """
        logger.info(f"Creating visualizations for continuous feature: {feature_name}")
        
        # Get plot configuration
        plot_types = plot_types or self.viz_config.get('continuous_plots', 
                                                      ['histogram', 'kde', 'box', 'qq'])
        figsize = self.viz_config.get('figsize', (10, 6))
        dpi = self.viz_config.get('dpi', 100)
        
        plot_paths = {}
        
        # 1. Histogram with KDE
        if 'histogram' in plot_types:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            for name, dist in distributions.items():
                sns.histplot(
                    data=dist,
                    label=name,
                    alpha=0.5,
                    stat='density',
                    common_norm=True,
                    ax=ax
                )
                
                # Add KDE if enabled
                if 'kde' in plot_types:
                    sns.kdeplot(
                        data=dist,
                        label=f"{name} (KDE)",
                        ax=ax
                    )
            
            ax.set_title(f'Distribution Comparison - {feature_name}')
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Density')
            ax.legend()
            
            # Save plot
            plot_path = self.output_dir / f'{feature_name}_histogram.png'
            fig.savefig(plot_path)
            plt.close(fig)
            plot_paths['histogram'] = plot_path
        
        # 2. Box Plot
        if 'box' in plot_types:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            # Prepare data for box plot
            box_data = pd.DataFrame({
                name: dist.values for name, dist in distributions.items()
            })
            
            sns.boxplot(data=box_data, ax=ax)
            ax.set_title(f'Box Plot Comparison - {feature_name}')
            ax.set_ylabel(feature_name)
            
            # Save plot
            plot_path = self.output_dir / f'{feature_name}_boxplot.png'
            fig.savefig(plot_path)
            plt.close(fig)
            plot_paths['box'] = plot_path
        
        # 3. Q-Q Plot
        if 'qq' in plot_types:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            # Use first distribution as reference
            reference_name = list(distributions.keys())[0]
            reference_dist = distributions[reference_name]
            
            for name, dist in distributions.items():
                if name != reference_name:
                    # Calculate theoretical quantiles
                    quantiles = np.linspace(0, 100, len(dist))
                    ref_percentiles = np.percentile(reference_dist, quantiles)
                    dist_percentiles = np.percentile(dist, quantiles)
                    
                    # Plot Q-Q line
                    ax.scatter(
                        ref_percentiles,
                        dist_percentiles,
                        alpha=0.5,
                        label=f'{name} vs {reference_name}'
                    )
                    
                    # Add reference line
                    min_val = min(ref_percentiles.min(), dist_percentiles.min())
                    max_val = max(ref_percentiles.max(), dist_percentiles.max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax.set_title(f'Q-Q Plot - {feature_name}')
            ax.set_xlabel(f'{reference_name} Distribution')
            ax.set_ylabel('Comparison Distribution')
            ax.legend()
            
            # Save plot
            plot_path = self.output_dir / f'{feature_name}_qq.png'
            fig.savefig(plot_path)
            plt.close(fig)
            plot_paths['qq'] = plot_path
        
        logger.info(f"Created {len(plot_paths)} visualizations for {feature_name}")
        return plot_paths
    
    def plot_categorical_distribution(
        self,
        distributions: Dict[str, pd.Series],
        feature_name: str,
        plot_types: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """
        Create visualizations for categorical feature distributions.
        
        Args:
            distributions (Dict[str, pd.Series]): Dictionary of distributions to compare
            feature_name (str): Name of the feature being visualized
            plot_types (Optional[List[str]]): Types of plots to generate
                (default: ['bar', 'pie'])
        
        Returns:
            Dict[str, Path]: Dictionary mapping plot types to their saved file paths
        """
        logger.info(f"Creating visualizations for categorical feature: {feature_name}")
        
        # Get plot configuration
        plot_types = plot_types or self.viz_config.get('categorical_plots', ['bar', 'pie'])
        figsize = self.viz_config.get('figsize', (10, 6))
        dpi = self.viz_config.get('dpi', 100)
        
        plot_paths = {}
        
        # Calculate proportions for each distribution
        proportions = {
            name: dist.value_counts(normalize=True)
            for name, dist in distributions.items()
        }
        
        # Ensure all categories are present in all distributions
        all_categories = pd.Index(
            set().union(*[prop.index for prop in proportions.values()])
        )
        proportions = {
            name: prop.reindex(all_categories, fill_value=0)
            for name, prop in proportions.items()
        }
        
        # 1. Bar Plot
        if 'bar' in plot_types:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            # Set up bar positions
            n_categories = len(all_categories)
            n_distributions = len(distributions)
            width = 0.8 / n_distributions
            
            for i, (name, prop) in enumerate(proportions.items()):
                positions = np.arange(n_categories) + i * width
                ax.bar(positions, prop, width, label=name, alpha=0.7)
            
            ax.set_title(f'Category Distribution Comparison - {feature_name}')
            ax.set_xlabel('Categories')
            ax.set_ylabel('Proportion')
            ax.set_xticks(np.arange(n_categories) + width * (n_distributions - 1) / 2)
            ax.set_xticklabels(all_categories, rotation=45 if n_categories > 5 else 0)
            ax.legend()
            
            # Save plot
            plot_path = self.output_dir / f'{feature_name}_barplot.png'
            fig.savefig(plot_path, bbox_inches='tight')
            plt.close(fig)
            plot_paths['bar'] = plot_path
        
        # 2. Pie Charts
        if 'pie' in plot_types:
            n_distributions = len(distributions)
            fig, axes = plt.subplots(
                1, n_distributions,
                figsize=(figsize[0] * n_distributions, figsize[1]),
                dpi=dpi
            )
            
            if n_distributions == 1:
                axes = [axes]
            
            for ax, (name, prop) in zip(axes, proportions.items()):
                ax.pie(
                    prop,
                    labels=prop.index,
                    autopct='%1.1f%%',
                    startangle=90
                )
                ax.set_title(f'{name} Distribution')
            
            plt.suptitle(f'Category Distributions - {feature_name}')
            
            # Save plot
            plot_path = self.output_dir / f'{feature_name}_piechart.png'
            fig.savefig(plot_path, bbox_inches='tight')
            plt.close(fig)
            plot_paths['pie'] = plot_path
        
        logger.info(f"Created {len(plot_paths)} visualizations for {feature_name}")
        return plot_paths 