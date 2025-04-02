"""
Model Performance Metrics Module
Handles calculation and tracking of model performance metrics
"""

from typing import Dict, List, Optional
import numpy as np
from datetime import datetime, timedelta
from src.core.database import MetricsStorage
from src.shared.utils.validation import validate_metrics

class ModelPerformanceMetrics:
    def __init__(self, model_id: str, storage: Optional[MetricsStorage] = None):
        self.model_id = model_id
        self.storage = storage or MetricsStorage()
        
    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate model accuracy over a given period"""
        return np.mean(y_true == y_pred)
    
    def calculate_latency(self, prediction_times: List[float]) -> Dict[str, float]:
        """Calculate model inference latency statistics"""
        times = np.array(prediction_times)
        return {
            "mean_latency": np.mean(times),
            "p95_latency": np.percentile(times, 95),
            "p99_latency": np.percentile(times, 99)
        }
    
    def track_performance(self, 
                         metric_name: str, 
                         metric_value: float,
                         timestamp: Optional[datetime] = None) -> None:
        """Track a performance metric over time"""
        timestamp = timestamp or datetime.now()
        validated_metric = validate_metrics({metric_name: metric_value})
        self.storage.store_metric(
            model_id=self.model_id,
            metric_name=metric_name,
            metric_value=validated_metric[metric_name],
            timestamp=timestamp
        )
    
    def get_performance_history(self,
                              metric_name: str,
                              start_time: datetime,
                              end_time: Optional[datetime] = None) -> Dict[str, List]:
        """Retrieve historical performance metrics"""
        end_time = end_time or datetime.now()
        return self.storage.get_metric_history(
            model_id=self.model_id,
            metric_name=metric_name,
            start_time=start_time,
            end_time=end_time
        )
    
    def analyze_performance_trend(self,
                                metric_name: str,
                                window_size: timedelta = timedelta(hours=24)) -> Dict:
        """Analyze performance trends over time"""
        end_time = datetime.now()
        start_time = end_time - window_size
        
        history = self.get_performance_history(metric_name, start_time, end_time)
        values = np.array(history["values"])
        
        return {
            "current_value": values[-1] if len(values) > 0 else None,
            "mean": np.mean(values),
            "std": np.std(values),
            "trend": np.polyfit(range(len(values)), values, deg=1)[0] if len(values) > 1 else 0
        } 