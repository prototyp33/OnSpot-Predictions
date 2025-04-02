"""
Model Comparison Module
Handles comparison of multiple models' performance metrics and characteristics
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from src.features.model_performance.data_access.metrics_repository import MetricsRepository

@dataclass
class ComparisonResult:
    """Container for model comparison results"""
    metric_name: str
    values: Dict[str, float]
    best_model: str
    improvement: float  # Percentage improvement over second best
    timestamp: datetime
    comparison_window: timedelta

class ModelComparator:
    """Handles comparison of multiple models"""
    
    def __init__(self, metrics_repository: MetricsRepository):
        self.metrics_repository = metrics_repository
    
    async def compare_models(
        self,
        model_ids: List[str],
        metrics: List[str],
        time_window: timedelta = timedelta(days=7)
    ) -> Dict[str, ComparisonResult]:
        """
        Compare multiple models across specified metrics
        
        Args:
            model_ids: List of model IDs to compare
            metrics: List of metric names to compare
            time_window: Time window for comparison
            
        Returns:
            Dictionary of comparison results by metric
        """
        results = {}
        
        for metric_name in metrics:
            metric_values = {}
            
            for model_id in model_ids:
                # Get metric history for each model
                history = await self.metrics_repository.get_metric_history(
                    model_id=model_id,
                    metric_name=metric_name,
                    time_window=time_window
                )
                
                # Calculate average metric value
                if history:
                    values = [entry.value for entry in history]
                    metric_values[model_id] = np.mean(values)
            
            if metric_values:
                # Determine best model and improvement
                sorted_models = sorted(
                    metric_values.items(),
                    key=lambda x: x[1],
                    reverse=metric_name == "r2"  # Higher is better for R²
                )
                
                best_model = sorted_models[0][0]
                best_value = sorted_models[0][1]
                
                # Calculate improvement over second best
                if len(sorted_models) > 1:
                    second_best = sorted_models[1][1]
                    improvement = abs((best_value - second_best) / second_best * 100)
                else:
                    improvement = 0.0
                
                results[metric_name] = ComparisonResult(
                    metric_name=metric_name,
                    values=metric_values,
                    best_model=best_model,
                    improvement=improvement,
                    timestamp=datetime.now(),
                    comparison_window=time_window
                )
        
        return results
    
    async def get_model_rankings(
        self,
        model_ids: List[str],
        metrics: Dict[str, float],  # metric_name: weight
        time_window: timedelta = timedelta(days=7)
    ) -> List[Tuple[str, float]]:
        """
        Rank models based on weighted combination of metrics
        
        Args:
            model_ids: List of model IDs to rank
            metrics: Dictionary of metric names and their weights
            time_window: Time window for comparison
            
        Returns:
            List of (model_id, score) tuples, sorted by score
        """
        metric_comparisons = await self.compare_models(
            model_ids=model_ids,
            metrics=list(metrics.keys()),
            time_window=time_window
        )
        
        scores = {model_id: 0.0 for model_id in model_ids}
        
        for metric_name, weight in metrics.items():
            if metric_name in metric_comparisons:
                comparison = metric_comparisons[metric_name]
                
                # Normalize values to [0, 1] range
                values = comparison.values
                if values:
                    min_val = min(values.values())
                    max_val = max(values.values())
                    range_val = max_val - min_val if max_val > min_val else 1.0
                    
                    for model_id, value in values.items():
                        normalized_value = (value - min_val) / range_val
                        if metric_name != "r2":  # Lower is better for error metrics
                            normalized_value = 1 - normalized_value
                        scores[model_id] += normalized_value * weight
        
        # Normalize final scores
        max_score = max(scores.values()) if scores else 1.0
        normalized_scores = {
            model_id: score / max_score
            for model_id, score in scores.items()
        }
        
        # Sort by score in descending order
        return sorted(
            normalized_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
    
    async def analyze_performance_trends(
        self,
        model_ids: List[str],
        metric_name: str,
        time_window: timedelta = timedelta(days=30),
        aggregation: str = 'daily'
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze performance trends for multiple models
        
        Args:
            model_ids: List of model IDs to analyze
            metric_name: Metric to analyze
            time_window: Time window for analysis
            aggregation: Aggregation period ('hourly', 'daily', 'weekly')
            
        Returns:
            Dictionary of DataFrames containing trend analysis by model
        """
        trends = {}
        
        for model_id in model_ids:
            # Get metric history
            history = await self.metrics_repository.get_metric_history(
                model_id=model_id,
                metric_name=metric_name,
                time_window=time_window,
                aggregation=aggregation
            )
            
            if history:
                # Convert to DataFrame
                df = pd.DataFrame([
                    {
                        'timestamp': entry.timestamp,
                        'value': entry.value,
                        'sample_size': entry.sample_size
                    }
                    for entry in history
                ])
                
                # Calculate rolling statistics
                df['rolling_mean'] = df['value'].rolling(window=7).mean()
                df['rolling_std'] = df['value'].rolling(window=7).std()
                
                # Calculate trend
                df['trend'] = df['rolling_mean'].diff().fillna(0)
                
                trends[model_id] = df
        
        return trends 