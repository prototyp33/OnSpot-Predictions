"""
Data Quality Monitoring Module
Tracks and analyzes data quality metrics for model inputs and predictions
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """Container for data quality metrics"""
    missing_rate: float
    out_of_range_rate: float
    correlation_changes: Dict[str, float]
    distribution_metrics: Dict[str, Dict[str, float]]
    timestamp: datetime
    sample_size: int
    feature_names: List[str]

class DataQualityMonitor:
    """Monitors data quality metrics"""
    
    def __init__(
        self,
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        correlation_threshold: float = 0.1,
        distribution_threshold: float = 0.05
    ):
        self.feature_ranges = feature_ranges or {}
        self.correlation_threshold = correlation_threshold
        self.distribution_threshold = distribution_threshold
        self.reference_stats: Dict[str, Dict] = {}
    
    def set_reference_data(
        self,
        reference_data: np.ndarray,
        feature_names: List[str]
    ) -> None:
        """Set reference data for comparison"""
        if len(feature_names) != reference_data.shape[1]:
            raise ValueError("Number of feature names must match data dimensions")
        
        # Calculate reference statistics
        for i, feature in enumerate(feature_names):
            feature_data = reference_data[:, i]
            
            self.reference_stats[feature] = {
                'mean': np.mean(feature_data),
                'std': np.std(feature_data),
                'quantiles': np.percentile(feature_data, [25, 50, 75]),
                'range': (np.min(feature_data), np.max(feature_data))
            }
            
            if feature not in self.feature_ranges:
                # Set default range as mean ± 3*std
                mean, std = self.reference_stats[feature]['mean'], self.reference_stats[feature]['std']
                self.feature_ranges[feature] = (mean - 3*std, mean + 3*std)
        
        # Calculate reference correlations
        self.reference_correlations = pd.DataFrame(reference_data, columns=feature_names).corr()
    
    def check_data_quality(
        self,
        data: np.ndarray,
        feature_names: List[str],
        timestamp: Optional[datetime] = None
    ) -> DataQualityMetrics:
        """
        Check data quality metrics
        
        Args:
            data: Input data to check
            feature_names: List of feature names
            timestamp: Optional timestamp for the check
            
        Returns:
            DataQualityMetrics object containing quality metrics
        """
        if not self.reference_stats:
            raise ValueError("Reference data must be set before checking quality")
        
        if len(feature_names) != data.shape[1]:
            raise ValueError("Number of feature names must match data dimensions")
        
        timestamp = timestamp or datetime.now()
        
        # Check missing values
        missing_rates = np.isnan(data).mean(axis=0)
        overall_missing_rate = missing_rates.mean()
        
        # Check out of range values
        out_of_range_counts = np.zeros(data.shape[1])
        for i, feature in enumerate(feature_names):
            if feature in self.feature_ranges:
                min_val, max_val = self.feature_ranges[feature]
                out_of_range_counts[i] = np.sum(
                    (data[:, i] < min_val) | (data[:, i] > max_val)
                )
        
        overall_out_of_range_rate = out_of_range_counts.sum() / data.size
        
        # Check correlation changes
        current_correlations = pd.DataFrame(data, columns=feature_names).corr()
        correlation_changes = {}
        
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                feat1, feat2 = feature_names[i], feature_names[j]
                ref_corr = self.reference_correlations.loc[feat1, feat2]
                curr_corr = current_correlations.loc[feat1, feat2]
                
                if abs(ref_corr - curr_corr) > self.correlation_threshold:
                    correlation_changes[f"{feat1}__{feat2}"] = curr_corr - ref_corr
        
        # Check distribution metrics
        distribution_metrics = {}
        for i, feature in enumerate(feature_names):
            ref_stats = self.reference_stats[feature]
            curr_data = data[:, i]
            
            # Calculate current statistics
            curr_mean = np.mean(curr_data)
            curr_std = np.std(curr_data)
            curr_quantiles = np.percentile(curr_data, [25, 50, 75])
            
            # Calculate distribution metrics
            ks_statistic, p_value = stats.ks_2samp(
                curr_data,
                np.random.normal(ref_stats['mean'], ref_stats['std'], len(curr_data))
            )
            
            distribution_metrics[feature] = {
                'mean_shift': (curr_mean - ref_stats['mean']) / ref_stats['std'],
                'std_ratio': curr_std / ref_stats['std'],
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                'is_significant': p_value < self.distribution_threshold
            }
        
        return DataQualityMetrics(
            missing_rate=overall_missing_rate,
            out_of_range_rate=overall_out_of_range_rate,
            correlation_changes=correlation_changes,
            distribution_metrics=distribution_metrics,
            timestamp=timestamp,
            sample_size=len(data),
            feature_names=feature_names
        )
    
    def get_quality_score(
        self,
        metrics: DataQualityMetrics
    ) -> float:
        """Calculate overall quality score from metrics"""
        # Start with perfect score
        score = 1.0
        
        # Penalize for missing values
        score *= (1 - metrics.missing_rate)
        
        # Penalize for out of range values
        score *= (1 - metrics.out_of_range_rate)
        
        # Penalize for correlation changes
        if metrics.correlation_changes:
            avg_corr_change = np.mean([abs(c) for c in metrics.correlation_changes.values()])
            score *= max(0, 1 - avg_corr_change)
        
        # Penalize for distribution shifts
        distribution_penalties = []
        for feature_metrics in metrics.distribution_metrics.values():
            if feature_metrics['is_significant']:
                distribution_penalties.append(feature_metrics['ks_statistic'])
        
        if distribution_penalties:
            score *= max(0, 1 - np.mean(distribution_penalties))
        
        return score
    
    def get_feature_importance(
        self,
        metrics: DataQualityMetrics
    ) -> Dict[str, float]:
        """Calculate feature importance based on quality issues"""
        importance_scores = {feature: 0.0 for feature in metrics.feature_names}
        
        # Add importance based on distribution shifts
        for feature, dist_metrics in metrics.distribution_metrics.items():
            if dist_metrics['is_significant']:
                importance_scores[feature] += dist_metrics['ks_statistic']
        
        # Add importance based on correlation changes
        for corr_pair, change in metrics.correlation_changes.items():
            feat1, feat2 = corr_pair.split('__')
            importance_scores[feat1] += abs(change) / 2
            importance_scores[feat2] += abs(change) / 2
        
        # Normalize scores
        max_score = max(importance_scores.values()) if importance_scores else 1.0
        return {
            feature: score / max_score
            for feature, score in importance_scores.items()
        } 