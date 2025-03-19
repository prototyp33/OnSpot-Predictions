"""
Drift Detection Module for Model Monitoring
Implements statistical tests and seasonal adjustments for detecting data and concept drift
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime
import logging
from statsmodels.tsa.seasonal import seasonal_decompose
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DriftResult:
    """Container for drift detection results"""
    feature_name: str
    drift_score: float
    p_value: float
    is_drift_detected: bool
    test_name: str
    timestamp: datetime
    sample_size: int
    reference_stats: Dict
    current_stats: Dict
    seasonal_adjusted: bool = False

class DriftDetector:
    def __init__(
        self,
        drift_threshold: float = 0.05,
        min_samples: int = 1000,
        seasonal_period: Optional[int] = None
    ):
        """
        Initialize drift detector
        
        Args:
            drift_threshold: Significance level for drift detection tests
            min_samples: Minimum number of samples required for drift detection
            seasonal_period: Number of periods for seasonal decomposition (e.g., 24 for hourly data with daily seasonality)
        """
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples
        self.seasonal_period = seasonal_period

    def _calculate_distribution_stats(self, data: np.ndarray) -> Dict:
        """Calculate basic distribution statistics"""
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'median': float(np.median(data)),
            'q1': float(np.percentile(data, 25)),
            'q3': float(np.percentile(data, 75)),
            'min': float(np.min(data)),
            'max': float(np.max(data))
        }

    def _apply_seasonal_adjustment(
        self,
        data: pd.Series,
        timestamps: pd.Series
    ) -> pd.Series:
        """
        Apply seasonal adjustment to time series data
        
        Args:
            data: Time series data
            timestamps: Corresponding timestamps
        
        Returns:
            Seasonally adjusted data
        """
        if not self.seasonal_period:
            return data

        try:
            # Create time series with regular frequency
            ts = pd.Series(data.values, index=pd.DatetimeIndex(timestamps))
            
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                ts,
                period=self.seasonal_period,
                extrapolate_trend='freq'
            )
            
            # Return seasonally adjusted data
            return decomposition.resid + decomposition.trend
        except Exception as e:
            logger.warning(f"Failed to apply seasonal adjustment: {str(e)}")
            return data

    def detect_feature_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_name: str,
        timestamps: Optional[pd.Series] = None
    ) -> DriftResult:
        """
        Detect drift in a single feature using KS test
        
        Args:
            reference_data: Historical reference data
            current_data: Current data to test for drift
            feature_name: Name of the feature being tested
            timestamps: Timestamps for seasonal adjustment (if enabled)
        
        Returns:
            DriftResult object containing test results
        """
        if len(reference_data) < self.min_samples or len(current_data) < self.min_samples:
            logger.warning(f"Insufficient samples for drift detection in {feature_name}")
            return None

        # Apply seasonal adjustment if timestamps are provided and seasonal_period is set
        seasonal_adjusted = False
        if timestamps is not None and self.seasonal_period:
            reference_data = self._apply_seasonal_adjustment(
                pd.Series(reference_data),
                timestamps[:len(reference_data)]
            )
            current_data = self._apply_seasonal_adjustment(
                pd.Series(current_data),
                timestamps[len(reference_data):]
            )
            seasonal_adjusted = True

        # Perform Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.ks_2samp(reference_data, current_data)
        
        return DriftResult(
            feature_name=feature_name,
            drift_score=ks_statistic,
            p_value=p_value,
            is_drift_detected=p_value < self.drift_threshold,
            test_name='Kolmogorov-Smirnov',
            timestamp=datetime.now(),
            sample_size=min(len(reference_data), len(current_data)),
            reference_stats=self._calculate_distribution_stats(reference_data),
            current_stats=self._calculate_distribution_stats(current_data),
            seasonal_adjusted=seasonal_adjusted
        )

    def detect_concept_drift(
        self,
        reference_predictions: np.ndarray,
        reference_actuals: np.ndarray,
        current_predictions: np.ndarray,
        current_actuals: np.ndarray
    ) -> DriftResult:
        """
        Detect concept drift by comparing error distributions
        
        Args:
            reference_predictions: Model predictions on reference data
            reference_actuals: Actual values for reference data
            current_predictions: Model predictions on current data
            current_actuals: Actual values for current data
        
        Returns:
            DriftResult object containing test results
        """
        if (len(reference_predictions) < self.min_samples or 
            len(current_predictions) < self.min_samples):
            logger.warning("Insufficient samples for concept drift detection")
            return None

        # Calculate prediction errors
        reference_errors = np.abs(reference_predictions - reference_actuals)
        current_errors = np.abs(current_predictions - current_actuals)

        # Perform KS test on error distributions
        ks_statistic, p_value = stats.ks_2samp(reference_errors, current_errors)
        
        return DriftResult(
            feature_name='prediction_error',
            drift_score=ks_statistic,
            p_value=p_value,
            is_drift_detected=p_value < self.drift_threshold,
            test_name='Kolmogorov-Smirnov',
            timestamp=datetime.now(),
            sample_size=min(len(reference_errors), len(current_errors)),
            reference_stats=self._calculate_distribution_stats(reference_errors),
            current_stats=self._calculate_distribution_stats(current_errors),
            seasonal_adjusted=False
        )

    def detect_all_feature_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        timestamp_column: Optional[str] = None
    ) -> Dict[str, DriftResult]:
        """
        Detect drift for all features in the dataset
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset to test for drift
            timestamp_column: Name of the timestamp column for seasonal adjustment
        
        Returns:
            Dictionary mapping feature names to their drift detection results
        """
        results = {}
        timestamps = None
        
        if timestamp_column and timestamp_column in reference_data.columns:
            timestamps = pd.concat([
                reference_data[timestamp_column],
                current_data[timestamp_column]
            ])

        for column in reference_data.columns:
            if column == timestamp_column:
                continue
                
            try:
                result = self.detect_feature_drift(
                    reference_data[column].values,
                    current_data[column].values,
                    column,
                    timestamps
                )
                if result:
                    results[column] = result
            except Exception as e:
                logger.error(f"Error detecting drift for feature {column}: {str(e)}")

        return results 