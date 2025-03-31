"""Drift detection module for monitoring data and model drift."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from dataclasses import dataclass

@dataclass
class DriftResult:
    """Container for drift detection results."""
    feature_name: str
    p_value: float
    statistic: float
    is_drift: bool
    drift_type: str
    threshold: float

class DriftDetector:
    """Detects data drift in features and predictions."""
    
    def __init__(
        self,
        threshold: float = 0.05,
        config: Optional[Dict[str, Any]] = None
    ):
        self.threshold = threshold
        self.reference_stats: Dict[str, Dict[str, float]] = {}
        self.numerical_features: List[str] = []
        self.categorical_features: List[str] = []
    
    def fit(self, reference_data: pd.DataFrame) -> 'DriftDetector':
        """Compute reference statistics from training data."""
        # Identify feature types
        for col in reference_data.columns:
            if reference_data[col].dtype in ['int64', 'float64']:
                self.numerical_features.append(col)
            else:
                self.categorical_features.append(col)
        
        # Compute reference statistics
        for col in self.numerical_features:
            self.reference_stats[col] = {
                'mean': reference_data[col].mean(),
                'std': reference_data[col].std(),
                'q25': reference_data[col].quantile(0.25),
                'q75': reference_data[col].quantile(0.75)
            }
        
        for col in self.categorical_features:
            self.reference_stats[col] = {
                'distribution': reference_data[col].value_counts(normalize=True).to_dict()
            }
        
        return self
    
    def detect_drift(
        self, current_data: pd.DataFrame
    ) -> Dict[str, DriftResult]:
        """Detect drift in current data compared to reference data."""
        results = {}
        
        # Check numerical features
        for feature in self.numerical_features:
            ref_stats = self.reference_stats[feature]
            current_values = current_data[feature].dropna()
            
            # Perform Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(
                current_values,
                np.random.normal(
                    ref_stats['mean'],
                    ref_stats['std'],
                    len(current_values)
                )
            )
            
            results[feature] = DriftResult(
                feature_name=feature,
                p_value=p_value,
                statistic=statistic,
                is_drift=p_value < self.threshold,
                drift_type='distribution',
                threshold=self.threshold
            )
        
        # Check categorical features
        for feature in self.categorical_features:
            ref_dist = self.reference_stats[feature]['distribution']
            current_dist = current_data[feature].value_counts(normalize=True).to_dict()
            
            # Perform chi-square test
            ref_counts = pd.Series(ref_dist)
            current_counts = pd.Series(current_dist)
            
            # Align categories
            all_categories = sorted(set(ref_counts.index) | set(current_counts.index))
            ref_aligned = pd.Series(0, index=all_categories)
            current_aligned = pd.Series(0, index=all_categories)
            
            ref_aligned.update(ref_counts)
            current_aligned.update(current_counts)
            
            statistic, p_value = stats.chisquare(
                current_aligned * len(current_data),
                ref_aligned * len(current_data)
            )
            
            results[feature] = DriftResult(
                feature_name=feature,
                p_value=p_value,
                statistic=statistic,
                is_drift=p_value < self.threshold,
                drift_type='categorical',
                threshold=self.threshold
            )
        
        return results
    
    def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> DriftResult:
        """Detect drift in model predictions."""
        # Perform Kolmogorov-Smirnov test on predictions
        statistic, p_value = stats.ks_2samp(
            reference_predictions,
            current_predictions
        )
        
        return DriftResult(
            feature_name='predictions',
            p_value=p_value,
            statistic=statistic,
            is_drift=p_value < self.threshold,
            drift_type='prediction',
            threshold=self.threshold
        )
    
    def get_drift_report(
        self, drift_results: Dict[str, DriftResult]
    ) -> Dict[str, Any]:
        """Generate a summary report of drift detection results."""
        drifted_features = [
            f for f, r in drift_results.items() if r.is_drift
        ]
        
        return {
            'total_features': len(drift_results),
            'drifted_features': len(drifted_features),
            'drift_ratio': len(drifted_features) / len(drift_results),
            'drifted_feature_names': drifted_features,
            'results': {
                f: {
                    'p_value': r.p_value,
                    'statistic': r.statistic,
                    'drift_type': r.drift_type
                }
                for f, r in drift_results.items()
            }
        } 