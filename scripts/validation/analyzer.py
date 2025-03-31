"""
Distribution analyzer module for comparing feature distributions across dataset splits.

This module combines statistical tests and visualizations to provide a comprehensive
analysis of distribution differences between dataset splits.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from .statistical_tests import StatisticalTests, TestResult
from .visualization import DistributionVisualizer
import json
import yaml
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Container for analysis results of a single feature."""
    feature_name: str
    feature_type: str  # 'continuous' or 'categorical'
    statistical_results: Dict[str, TestResult]
    visualization_paths: Dict[str, str]
    summary: Dict[str, Any]  # Summary statistics and metadata

class DistributionAnalyzer:
    """
    Class for analyzing distributions of features across dataset splits.
    Orchestrates statistical tests and visualizations.
    """
    
    def __init__(
        self,
        config: 'ValidationConfig',  # Forward reference
        stats: Optional[StatisticalTests] = None,
        visualizer: Optional[DistributionVisualizer] = None
    ):
        """
        Initialize the analyzer with configuration and optional components.
        
        Args:
            config: Validation configuration
            stats: Optional StatisticalTests instance
            visualizer: Optional DistributionVisualizer instance
        """
        self.config = config
        self.stats = stats or StatisticalTests(config)
        self.visualizer = visualizer or DistributionVisualizer(config)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        self.output_dir = Path(self.config.get_output_dir()) / 'analysis_results'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized DistributionAnalyzer")
    
    def _is_categorical(self, series: pd.Series, threshold: int = 10) -> bool:
        """
        Determine if a feature should be treated as categorical.
        
        Args:
            series (pd.Series): Feature values
            threshold (int): Maximum number of unique values for categorical features
            
        Returns:
            bool: True if feature should be treated as categorical
        """
        if pd.api.types.is_categorical_dtype(series):
            return True
        if pd.api.types.is_bool_dtype(series):
            return True
        if pd.api.types.is_object_dtype(series):
            return True
        if pd.api.types.is_numeric_dtype(series):
            return series.nunique() <= threshold
        return False
    
    def _get_basic_stats(self, distributions: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Calculate basic statistics for a feature across distributions.
        
        Args:
            distributions (Dict[str, pd.Series]): Dictionary of distributions
            
        Returns:
            Dict[str, Any]: Dictionary of basic statistics
        """
        stats = {}
        for name, dist in distributions.items():
            stats[name] = {
                'count': len(dist),
                'missing': dist.isna().sum(),
                'unique': dist.nunique()
            }
            
            if pd.api.types.is_numeric_dtype(dist):
                stats[name].update({
                    'mean': dist.mean(),
                    'std': dist.std(),
                    'min': dist.min(),
                    'max': dist.max(),
                    'median': dist.median()
                })
            else:
                value_counts = dist.value_counts(normalize=True)
                stats[name]['top_categories'] = value_counts.head(5).to_dict()
        
        return stats
    
    def analyze_feature(self, distributions: Dict[str, pd.Series], feature_name: str) -> AnalysisResult:
        """
        Analyze a single feature's distributions.
        
        Args:
            distributions: Dictionary of distributions to compare
            feature_name: Name of the feature being analyzed
            
        Returns:
            AnalysisResult containing test results and visualization paths
        """
        self.logger.info(f"Analyzing feature: {feature_name}")
        
        # Determine feature type
        is_categorical = self._is_categorical(distributions['reference'])
        feature_type = 'categorical' if is_categorical else 'continuous'
        
        # Run statistical tests
        test_results = (
            self.stats.run_categorical_tests(distributions, feature_name)
            if is_categorical
            else self.stats.run_continuous_tests(distributions, feature_name)
        )
        
        # Generate visualizations
        viz_paths = self.visualizer.plot_distributions(distributions, feature_name, is_categorical)
        
        # Calculate basic statistics
        stats_summary = self._get_basic_stats(distributions)
        
        # Create summary
        summary = {
            'feature_type': feature_type,
            'basic_stats': stats_summary,
            'test_results': test_results,
            'visualization_paths': viz_paths
        }
        
        return AnalysisResult(
            feature_name=feature_name,
            feature_type=feature_type,
            statistical_results=test_results,
            visualization_paths=viz_paths,
            summary=summary
        )
    
    def analyze_features(
        self,
        data_splits: Dict[str, pd.DataFrame],
        features: Optional[List[str]] = None,
        feature_types: Optional[Dict[str, str]] = None
    ) -> Dict[str, AnalysisResult]:
        """
        Analyze multiple features across dataset splits.
        
        Args:
            data_splits (Dict[str, pd.DataFrame]): Dictionary of dataset splits
            features (Optional[List[str]]): List of features to analyze. If None, analyzes all
            feature_types (Optional[Dict[str, str]]): Dictionary mapping features to their types
            
        Returns:
            Dict[str, AnalysisResult]: Dictionary mapping features to their analysis results
        """
        logger.info("Starting feature distribution analysis")
        
        # Get features to analyze
        if features is None:
            features = data_splits[next(iter(data_splits.keys()))].columns
        
        results = {}
        for feature in features:
            try:
                # Extract feature distributions
                distributions = {
                    name: split[feature] for name, split in data_splits.items()
                }
                
                # Get forced type if specified
                force_type = feature_types.get(feature) if feature_types else None
                
                # Analyze feature
                results[feature] = self.analyze_feature(
                    distributions,
                    feature
                )
                
                logger.info(f"Successfully analyzed feature: {feature}")
                
            except Exception as e:
                logger.error(f"Error analyzing feature {feature}: {str(e)}")
                continue
        
        return results
    
    def _convert_test_result_to_dict(self, test_result: TestResult) -> Dict[str, Any]:
        """Convert a TestResult object to a serializable dictionary."""
        result_dict = {
            'statistic': float(test_result.statistic) if test_result.statistic is not None else None,
            'p_value': float(test_result.p_value) if test_result.p_value is not None else None,
            'is_significant': bool(test_result.is_significant),
            'details': self._ensure_serializable(test_result.details),
            'interpretation': str(test_result.interpretation) if test_result.interpretation else None
        }
        return result_dict

    def _ensure_serializable(self, obj: Any) -> Any:
        """Ensure an object is JSON-serializable."""
        if isinstance(obj, (str, int, float, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._ensure_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(k): self._ensure_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, bool):
            return bool(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._ensure_serializable(obj.tolist())
        elif pd.isna(obj):
            return None
        else:
            return str(obj)

    def _convert_analysis_result_to_dict(self, result: AnalysisResult) -> Dict[str, Any]:
        """Convert an AnalysisResult object to a serializable dictionary."""
        return {
            'feature_name': str(result.feature_name),
            'feature_type': str(result.feature_type),
            'statistical_results': {
                str(test_name): self._convert_test_result_to_dict(test_result)
                for test_name, test_result in result.statistical_results.items()
            },
            'visualization_paths': [str(path) for path in result.visualization_paths],
            'summary': self._ensure_serializable(result.summary)
        }

    def save_analysis_results(self, results: Union[Dict[str, AnalysisResult], List[AnalysisResult]], output_format: str = 'json') -> None:
        """Save analysis results to a file.
        
        Args:
            results: Dictionary or List of AnalysisResult objects to save
            output_format: Format to save results in ('json' or 'yaml')
        """
        if not results:
            logger.warning("No results to save")
            return

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Convert results to list if it's a dictionary
        results_list = list(results.values()) if isinstance(results, dict) else results

        # Convert results to serializable format
        serializable_results = [
            self._convert_analysis_result_to_dict(result)
            for result in results_list
        ]

        # Determine output path and format
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = self.output_dir / f'analysis_results_{timestamp}.{output_format}'

        try:
            if output_format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
            elif output_format == 'yaml':
                with open(output_path, 'w') as f:
                    yaml.dump(serializable_results, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            logger.info(f"Analysis results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save analysis results: {str(e)}")
            raise 