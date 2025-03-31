"""Metrics tracking module for monitoring model performance."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime

class MetricsTracker:
    """Tracks and stores model performance metrics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = {}
        self.export_path = Path(config['monitoring']['metrics']['export_path']
                              if config else 'monitoring/metrics/')
        self.export_path.mkdir(parents=True, exist_ok=True)
    
    def log_metrics(
        self,
        model_name: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log metrics for a model."""
        if model_name not in self.metrics_history:
            self.metrics_history[model_name] = []
        
        # Add timestamp and metadata
        metrics_entry = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'metadata': metadata or {}
        }
        
        self.metrics_history[model_name].append(metrics_entry)
        self._export_metrics(model_name)
    
    def get_metrics_history(
        self, model_name: str
    ) -> List[Dict[str, Any]]:
        """Get metrics history for a model."""
        return self.metrics_history.get(model_name, [])
    
    def get_latest_metrics(
        self, model_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get the most recent metrics for a model."""
        history = self.get_metrics_history(model_name)
        return history[-1] if history else None
    
    def _export_metrics(self, model_name: str) -> None:
        """Export metrics to disk."""
        metrics_file = self.export_path / f"{model_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history[model_name], f, indent=2)
    
    def load_metrics(self, model_name: str) -> None:
        """Load metrics from disk."""
        metrics_file = self.export_path / f"{model_name}_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                self.metrics_history[model_name] = json.load(f)
    
    def analyze_performance_trend(
        self,
        model_name: str,
        metric_name: str,
        window: int = 10
    ) -> Dict[str, Any]:
        """Analyze trend in a specific metric."""
        history = self.get_metrics_history(model_name)
        if not history:
            return {}
        
        # Extract metric values
        values = [
            entry['metrics'].get(metric_name)
            for entry in history
            if metric_name in entry['metrics']
        ]
        
        if not values:
            return {}
        
        # Calculate trend statistics
        trend_stats = {
            'current': values[-1],
            'mean': np.mean(values[-window:]),
            'std': np.std(values[-window:]),
            'min': np.min(values[-window:]),
            'max': np.max(values[-window:]),
            'trend': 'stable'
        }
        
        # Determine trend direction
        if len(values) >= window:
            slope = np.polyfit(range(window), values[-window:], 1)[0]
            if slope > 0.1:
                trend_stats['trend'] = 'improving'
            elif slope < -0.1:
                trend_stats['trend'] = 'degrading'
        
        return trend_stats
    
    def get_performance_summary(
        self, model_name: str
    ) -> Dict[str, Any]:
        """Generate a performance summary for a model."""
        history = self.get_metrics_history(model_name)
        if not history:
            return {}
        
        # Get all metric names
        metric_names = set()
        for entry in history:
            metric_names.update(entry['metrics'].keys())
        
        # Analyze trends for each metric
        summary = {
            'model_name': model_name,
            'metrics_tracked': len(metric_names),
            'history_length': len(history),
            'last_updated': history[-1]['timestamp'],
            'trends': {}
        }
        
        for metric in metric_names:
            summary['trends'][metric] = self.analyze_performance_trend(
                model_name, metric
            )
        
        return summary 