"""
A/B Testing module for OnSpot Predictive Model.
Handles test configuration, variant assignment, and statistical analysis.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
import time
import hashlib

class ABTest:
    def __init__(
        self,
        name: str,
        variants: List[str],
        traffic_split: Optional[List[float]] = None,
        min_sample_size: int = 1000,
        confidence_level: float = 0.95
    ):
        """
        Initialize an A/B test.
        
        Args:
            name: Name of the test
            variants: List of variant names (e.g., ["control", "treatment"])
            traffic_split: Optional list of traffic proportions (must sum to 1)
            min_sample_size: Minimum sample size required for statistical significance
            confidence_level: Confidence level for statistical tests (default: 0.95)
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.variants = variants
        self.traffic_split = traffic_split or [1.0 / len(variants)] * len(variants)
        self.min_sample_size = min_sample_size
        self.confidence_level = confidence_level
        self.start_time = datetime.utcnow()
        self.end_time = None
        
        # Validation
        if len(self.traffic_split) != len(variants):
            raise ValueError("Traffic split must match number of variants")
        if abs(sum(self.traffic_split) - 1.0) > 1e-6:
            raise ValueError("Traffic split must sum to 1")

        # Metrics storage
        self.metrics: Dict[str, Dict[str, List[float]]] = {
            variant: {
                "predictions": [],
                "actuals": [],
                "response_times": []
            } for variant in variants
        }

    def assign_variant(self, user_id: str) -> str:
        """
        Assign a user to a variant using consistent hashing.
        
        Args:
            user_id: Unique identifier for the user/request
            
        Returns:
            Assigned variant name
        """
        # Use hash of user_id and test_id for consistent assignment
        hash_input = f"{self.name}:{user_id}".encode()
        hash_value = int(hashlib.sha256(hash_input).hexdigest(), 16)
        
        # Use the hash to consistently assign a variant
        random_value = hash_value / 2**256  # Normalize to [0,1]
        cumulative_split = 0
        
        for variant, split in zip(self.variants, self.traffic_split):
            cumulative_split += split
            if random_value < cumulative_split:
                return variant
        return self.variants[-1]

    def record_observation(
        self,
        variant: str,
        predicted: float,
        actual: Optional[float] = None,
        response_time: Optional[float] = None
    ) -> None:
        """
        Record an observation for a variant.
        
        Args:
            variant: Variant name
            predicted: Predicted value
            actual: Actual value (if available)
            response_time: Response time in milliseconds
        """
        if variant not in self.metrics:
            raise ValueError(f"Unknown variant: {variant}")
        
        self.metrics[variant]["predictions"].append(predicted)
        if actual is not None:
            self.metrics[variant]["actuals"].append(actual)
        if response_time is not None:
            self.metrics[variant]["response_times"].append(response_time)

    def get_statistics(self) -> Dict[str, Dict[str, Union[float, int, bool]]]:
        """
        Calculate statistics for each variant.
        
        Returns:
            Dictionary containing statistics for each variant
        """
        stats_dict = {}
        
        for variant in self.variants:
            metrics = self.metrics[variant]
            n_samples = len(metrics["predictions"])
            
            # Skip if not enough samples
            if n_samples < self.min_sample_size:
                stats_dict[variant] = {
                    "sample_size": n_samples,
                    "has_sufficient_data": False
                }
                continue
                
            # Calculate basic statistics
            stats_dict[variant] = {
                "sample_size": n_samples,
                "has_sufficient_data": True,
                "mean_prediction": np.mean(metrics["predictions"]),
                "std_prediction": np.std(metrics["predictions"]),
            }
            
            # Add response time statistics if available
            if metrics["response_times"]:
                stats_dict[variant].update({
                    "mean_response_time": np.mean(metrics["response_times"]),
                    "p95_response_time": np.percentile(metrics["response_times"], 95),
                })
            
            # Add accuracy metrics if actuals available
            if metrics["actuals"]:
                mse = np.mean([(p - a) ** 2 for p, a in 
                             zip(metrics["predictions"], metrics["actuals"])])
                mae = np.mean([abs(p - a) for p, a in 
                             zip(metrics["predictions"], metrics["actuals"])])
                stats_dict[variant].update({
                    "mse": mse,
                    "mae": mae,
                    "rmse": np.sqrt(mse)
                })
        
        return stats_dict

    def calculate_significance(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Calculate statistical significance between variants.
        
        Returns:
            Dictionary containing p-values for various metrics between variants
        """
        if len(self.variants) != 2:
            raise ValueError("Significance testing only supported for two variants")
            
        variant_a, variant_b = self.variants
        metrics_a = self.metrics[variant_a]
        metrics_b = self.metrics[variant_b]
        
        # Skip if insufficient data
        if (len(metrics_a["predictions"]) < self.min_sample_size or
            len(metrics_b["predictions"]) < self.min_sample_size):
            return None
            
        results = {}
        
        # Test prediction distributions
        t_stat, p_val = stats.ttest_ind(
            metrics_a["predictions"],
            metrics_b["predictions"]
        )
        results["predictions"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "is_significant": p_val < (1 - self.confidence_level)
        }
        
        # Test response times if available
        if metrics_a["response_times"] and metrics_b["response_times"]:
            t_stat, p_val = stats.ttest_ind(
                metrics_a["response_times"],
                metrics_b["response_times"]
            )
            results["response_times"] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "is_significant": p_val < (1 - self.confidence_level)
            }
            
        # Test accuracy metrics if actuals available
        if metrics_a["actuals"] and metrics_b["actuals"]:
            errors_a = [abs(p - a) for p, a in 
                       zip(metrics_a["predictions"], metrics_a["actuals"])]
            errors_b = [abs(p - a) for p, a in 
                       zip(metrics_b["predictions"], metrics_b["actuals"])]
            t_stat, p_val = stats.ttest_ind(errors_a, errors_b)
            results["prediction_errors"] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "is_significant": p_val < (1 - self.confidence_level)
            }
            
        return results

    def end_test(self) -> None:
        """End the A/B test and record the end time."""
        self.end_time = datetime.utcnow()

    def get_winner(self) -> Optional[Tuple[str, Dict[str, float]]]:
        """
        Determine the winning variant based on statistical significance.
        
        Returns:
            Tuple of (winning_variant, improvement_metrics) or None if no clear winner
        """
        if not self.end_time:
            return None
            
        significance = self.calculate_significance()
        if not significance:
            return None
            
        stats_dict = self.get_statistics()
        variant_a, variant_b = self.variants
        
        # Check if differences are significant
        improvements = {}
        
        # Check prediction error improvement if available
        if ("prediction_errors" in significance and 
            significance["prediction_errors"]["is_significant"]):
            mae_a = stats_dict[variant_a].get("mae")
            mae_b = stats_dict[variant_b].get("mae")
            if mae_a is not None and mae_b is not None:
                improvement = (mae_a - mae_b) / mae_a * 100
                improvements["mae_improvement"] = improvement
                
        # Check response time improvement
        if ("response_times" in significance and 
            significance["response_times"]["is_significant"]):
            rt_a = stats_dict[variant_a].get("mean_response_time")
            rt_b = stats_dict[variant_b].get("mean_response_time")
            if rt_a is not None and rt_b is not None:
                improvement = (rt_a - rt_b) / rt_a * 100
                improvements["response_time_improvement"] = improvement
                
        if not improvements:
            return None
            
        # Determine winner based on improvements
        total_improvement = sum(improvements.values())
        winner = variant_b if total_improvement > 0 else variant_a
        
        return winner, improvements 