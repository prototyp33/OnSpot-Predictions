"""
Statistical tests for distribution comparison and drift detection.

This module provides a collection of statistical tests to compare distributions
between different dataset splits (train, validation, test) and detect potential
distribution drift.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional, Union, List, Any
import logging
from dataclasses import dataclass
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: Optional[float] = None
    threshold: Optional[float] = None
    is_significant: Optional[bool] = None
    details: Optional[Dict[str, Any]] = None
    
    @property
    def interpretation(self) -> Optional[str]:
        """Get test interpretation from details if available."""
        if self.details and 'interpretation' in self.details:
            if isinstance(self.details['interpretation'], str):
                return self.details['interpretation']
            elif isinstance(self.details['interpretation'], dict):
                if 'stability_assessment' in self.details['interpretation']:
                    return self.details['interpretation']['stability_assessment']
        return None

class StatisticalTests:
    """
    Collection of statistical tests for distribution comparison.
    """
    
    def __init__(self, config: 'ValidationConfig'):  # Forward reference
        """
        Initialize with configuration settings.
        
        Args:
            config: Optional ValidationConfig instance. If not provided, will use default config.
        """
        self.config = config or ValidationConfig()
        self.analysis_config = self.config.get_analysis_config()
        self.thresholds = self.config.get_thresholds()
        
        logger.info("Initialized StatisticalTests")
    
    def ks_test(
        self,
        distribution1: pd.Series,
        distribution2: pd.Series,
        feature_name: str
    ) -> TestResult:
        """
        Perform Kolmogorov-Smirnov test between two distributions.
        
        Args:
            distribution1 (pd.Series): First distribution
            distribution2 (pd.Series): Second distribution
            feature_name (str): Name of the feature being tested
        
        Returns:
            TestResult: Container with test results
        """
        # Get configuration
        significance_level = self.analysis_config['statistical_tests']['ks_test']['significance_level']
        
        # Remove NaN values
        dist1 = distribution1.dropna()
        dist2 = distribution2.dropna()
        
        # Log sample sizes
        logger.debug(f"KS test for {feature_name}: n1={len(dist1)}, n2={len(dist2)}")
        
        # Perform test
        statistic, p_value = stats.ks_2samp(dist1, dist2)
        
        # Calculate additional metrics
        details = {
            'sample_sizes': {
                'distribution1': len(dist1),
                'distribution2': len(dist2)
            },
            'missing_values': {
                'distribution1': len(distribution1) - len(dist1),
                'distribution2': len(distribution2) - len(dist2)
            },
            'basic_stats': {
                'distribution1': {
                    'mean': dist1.mean(),
                    'std': dist1.std(),
                    'median': dist1.median()
                },
                'distribution2': {
                    'mean': dist2.mean(),
                    'std': dist2.std(),
                    'median': dist2.median()
                }
            }
        }
        
        return TestResult(
            test_name='ks_test',
            statistic=statistic,
            p_value=p_value,
            threshold=significance_level,
            is_significant=p_value < significance_level,
            details=details
        )
    
    def chi_squared_test(
        self,
        distribution1: pd.Series,
        distribution2: pd.Series,
        feature_name: str
    ) -> TestResult:
        """
        Perform Chi-squared test for categorical distributions.
        
        Args:
            distribution1 (pd.Series): First distribution
            distribution2 (pd.Series): Second distribution
            feature_name (str): Name of the feature being tested
        
        Returns:
            TestResult: Container with test results
        """
        # Get configuration
        significance_level = self.analysis_config['statistical_tests']['chi_squared']['significance_level']
        
        # Calculate value counts
        counts1 = distribution1.value_counts()
        counts2 = distribution2.value_counts()
        
        # Ensure both distributions have the same categories
        all_categories = pd.Index(set(counts1.index) | set(counts2.index))
        counts1 = counts1.reindex(all_categories, fill_value=0)
        counts2 = counts2.reindex(all_categories, fill_value=0)
        
        # Perform test
        contingency_table = pd.DataFrame({
            'distribution1': counts1,
            'distribution2': counts2
        })
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Calculate category-wise contributions to chi-square
        observed = contingency_table.values
        chi2_contributions = (observed - expected) ** 2 / expected
        
        details = {
            'degrees_of_freedom': dof,
            'contingency_table': contingency_table.to_dict(),
            'expected_frequencies': pd.DataFrame(
                expected,
                index=contingency_table.index,
                columns=contingency_table.columns
            ).to_dict(),
            'chi2_contributions': pd.DataFrame(
                chi2_contributions,
                index=contingency_table.index,
                columns=contingency_table.columns
            ).to_dict(),
            'category_counts': {
                'distribution1': counts1.to_dict(),
                'distribution2': counts2.to_dict()
            }
        }
        
        return TestResult(
            test_name='chi_squared',
            statistic=chi2,
            p_value=p_value,
            threshold=significance_level,
            is_significant=p_value < significance_level,
            details=details
        )
    
    def calculate_psi(
        self,
        distribution1: pd.Series,
        distribution2: pd.Series,
        feature_name: str,
        bins: Optional[int] = None,
        is_categorical: bool = False
    ) -> TestResult:
        """
        Calculate Population Stability Index (PSI).
        
        PSI = Σ (Actual% - Expected%) * ln(Actual% / Expected%)
        
        Args:
            distribution1 (pd.Series): Expected distribution (usually train)
            distribution2 (pd.Series): Actual distribution (usually val/test)
            feature_name (str): Name of the feature
            bins (Optional[int]): Number of bins for continuous variables
            is_categorical (bool): Whether the feature is categorical
        
        Returns:
            TestResult: Container with PSI results
        """
        # Get configuration
        psi_config = self.analysis_config['statistical_tests']['psi']
        threshold = psi_config['threshold']
        bins = bins or psi_config['bins']
        
        if is_categorical:
            # For categorical variables, use value counts
            dist1_counts = distribution1.value_counts(normalize=True)
            dist2_counts = distribution2.value_counts(normalize=True)
            
            # Ensure both distributions have the same categories
            all_categories = pd.Index(set(dist1_counts.index) | set(dist2_counts.index))
            dist1_counts = dist1_counts.reindex(all_categories, fill_value=0)
            dist2_counts = dist2_counts.reindex(all_categories, fill_value=0)
            
            bin_details = {
                'type': 'categorical',
                'categories': list(all_categories)
            }
        else:
            # For continuous variables, bin the data
            min_val = min(distribution1.min(), distribution2.min())
            max_val = max(distribution1.max(), distribution2.max())
            
            # Create bins
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            dist1_counts, _ = np.histogram(distribution1, bins=bin_edges, density=True)
            dist2_counts, _ = np.histogram(distribution2, bins=bin_edges, density=True)
            
            # Normalize
            dist1_counts = dist1_counts / dist1_counts.sum()
            dist2_counts = dist2_counts / dist2_counts.sum()
            
            bin_details = {
                'type': 'continuous',
                'bin_edges': bin_edges.tolist(),
                'bin_centers': ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
            }
        
        # Calculate PSI
        # Add small epsilon to avoid division by zero or log of zero
        epsilon = 1e-10
        dist1_counts = np.array(dist1_counts) + epsilon
        dist2_counts = np.array(dist2_counts) + epsilon
        
        psi_components = (dist2_counts - dist1_counts) * np.log(dist2_counts / dist1_counts)
        psi = np.sum(psi_components)
        
        details = {
            'bin_details': bin_details,
            'distributions': {
                'expected': dist1_counts.tolist(),
                'actual': dist2_counts.tolist()
            },
            'psi_components': psi_components.tolist(),
            'interpretation': {
                'psi_threshold': threshold,
                'stability_assessment': 'unstable' if psi > threshold else 'stable',
                'guidelines': {
                    'PSI < 0.1': 'No significant change',
                    '0.1 ≤ PSI < 0.2': 'Moderate change',
                    'PSI ≥ 0.2': 'Significant change'
                }
            }
        }
        
        return TestResult(
            test_name='psi',
            statistic=psi,
            p_value=None,  # PSI doesn't have a p-value
            threshold=threshold,
            is_significant=psi > threshold,
            details=details
        )
    
    def mann_whitney_test(
        self,
        distribution1: pd.Series,
        distribution2: pd.Series,
        feature_name: str
    ) -> TestResult:
        """
        Perform Mann-Whitney U test between two distributions.
        
        Args:
            distribution1 (pd.Series): First distribution
            distribution2 (pd.Series): Second distribution
            feature_name (str): Name of the feature being tested
        
        Returns:
            TestResult: Container with test results
        """
        # Get configuration
        significance_level = self.analysis_config['statistical_tests']['mann_whitney']['significance_level']
        
        # Remove NaN values
        dist1 = distribution1.dropna()
        dist2 = distribution2.dropna()
        
        # Perform test
        statistic, p_value = stats.mannwhitneyu(
            dist1,
            dist2,
            alternative='two-sided',
            use_continuity=True
        )
        
        # Calculate effect size (r = Z / sqrt(N))
        n1, n2 = len(dist1), len(dist2)
        z_score = (statistic - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        effect_size = abs(z_score) / np.sqrt(n1 + n2)
        
        details = {
            'sample_sizes': {
                'distribution1': n1,
                'distribution2': n2
            },
            'effect_size': {
                'value': effect_size,
                'interpretation': {
                    'small': '0.1 - 0.3',
                    'medium': '0.3 - 0.5',
                    'large': '> 0.5'
                }
            },
            'test_details': {
                'z_score': z_score,
                'alternative': 'two-sided'
            }
        }
        
        return TestResult(
            test_name='mann_whitney',
            statistic=statistic,
            p_value=p_value,
            threshold=significance_level,
            is_significant=p_value < significance_level,
            details=details
        )
    
    def run_all_tests(
        self,
        distribution1: pd.Series,
        distribution2: pd.Series,
        feature_name: str,
        is_categorical: bool = False
    ) -> Dict[str, TestResult]:
        """
        Run all applicable statistical tests for a given feature.
        
        Args:
            distribution1 (pd.Series): First distribution
            distribution2 (pd.Series): Second distribution
            feature_name (str): Name of the feature being tested
            is_categorical (bool): Whether the feature is categorical
        
        Returns:
            Dict[str, TestResult]: Dictionary of test results
        """
        results = {}
        
        if is_categorical:
            # Tests for categorical variables
            if self.analysis_config['statistical_tests']['chi_squared']['enabled']:
                results['chi_squared'] = self.chi_squared_test(
                    distribution1, distribution2, feature_name
                )
        else:
            # Tests for continuous variables
            if self.analysis_config['statistical_tests']['ks_test']['enabled']:
                results['ks_test'] = self.ks_test(
                    distribution1, distribution2, feature_name
                )
            
            if self.analysis_config['statistical_tests']['mann_whitney']['enabled']:
                results['mann_whitney'] = self.mann_whitney_test(
                    distribution1, distribution2, feature_name
                )
        
        # PSI can be calculated for both types
        if self.analysis_config['statistical_tests']['psi']['enabled']:
            results['psi'] = self.calculate_psi(
                distribution1, distribution2, feature_name,
                is_categorical=is_categorical
            )
        
        return results 

    def run_continuous_tests(self, distributions: Dict[str, pd.Series], feature_name: str) -> Dict[str, TestResult]:
        """
        Run all applicable tests for continuous features.
        
        Args:
            distributions: Dictionary of distributions to compare (with string keys and pd.Series values)
            feature_name: Name of the feature being tested
            
        Returns:
            Dictionary of test results
        """
        results = {}
        ref_dist = distributions['reference']
        
        for name, comp_dist in distributions.items():
            if name == 'reference':
                continue
                
            results[f'ks_test_{name}'] = self.ks_test(ref_dist, comp_dist, feature_name)
            results[f'mann_whitney_{name}'] = self.mann_whitney_test(ref_dist, comp_dist, feature_name)
            results[f'psi_{name}'] = self.calculate_psi(ref_dist, comp_dist, feature_name, is_categorical=False)
            
        return results
        
    def run_categorical_tests(self, distributions: Dict[str, pd.Series], feature_name: str) -> Dict[str, TestResult]:
        """
        Run all applicable tests for categorical features.
        
        Args:
            distributions: Dictionary of distributions to compare (with string keys and pd.Series values)
            feature_name: Name of the feature being tested
            
        Returns:
            Dictionary of test results
        """
        results = {}
        ref_dist = distributions['reference']
        
        for name, comp_dist in distributions.items():
            if name == 'reference':
                continue
                
            results[f'chi_squared_{name}'] = self.chi_squared_test(ref_dist, comp_dist, feature_name)
            results[f'psi_{name}'] = self.calculate_psi(ref_dist, comp_dist, feature_name, is_categorical=True)
            
        return results 