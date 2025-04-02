"""
Model Performance Visualizations Module
Provides visualization components for model performance metrics
"""

from typing import Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from src.shared.visualization_components.base import BaseVisualization
from src.shared.utils.time_utils import format_timestamp

class PerformanceVisualizations(BaseVisualization):
    def __init__(self, model_id: str):
        super().__init__()
        self.model_id = model_id
        
    def create_metric_timeline(self,
                             timestamps: List[datetime],
                             values: List[float],
                             metric_name: str,
                             threshold: float = None) -> go.Figure:
        """Create a timeline visualization for a performance metric"""
        fig = go.Figure()
        
        # Add main metric line
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            name=metric_name,
            line=dict(color='blue'),
            mode='lines+markers'
        ))
        
        # Add threshold line if specified
        if threshold is not None:
            fig.add_trace(go.Scatter(
                x=[timestamps[0], timestamps[-1]],
                y=[threshold, threshold],
                name='Threshold',
                line=dict(color='red', dash='dash')
            ))
        
        fig.update_layout(
            title=f"{metric_name.title()} Over Time",
            xaxis_title="Timestamp",
            yaxis_title=metric_name.replace('_', ' ').title(),
            hovermode='x unified'
        )
        
        return fig
    
    def create_performance_dashboard(self,
                                  metrics_data: Dict[str, Dict],
                                  time_window: timedelta = timedelta(hours=24)) -> go.Figure:
        """Create a comprehensive performance dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Accuracy", "Latency", "Predictions", "Errors")
        )
        
        # Add accuracy timeline
        if "accuracy" in metrics_data:
            fig.add_trace(
                go.Scatter(
                    x=metrics_data["accuracy"]["timestamps"],
                    y=metrics_data["accuracy"]["values"],
                    name="Accuracy"
                ),
                row=1, col=1
            )
        
        # Add latency timeline
        if "latency" in metrics_data:
            fig.add_trace(
                go.Scatter(
                    x=metrics_data["latency"]["timestamps"],
                    y=metrics_data["latency"]["values"],
                    name="Latency (ms)"
                ),
                row=1, col=2
            )
        
        # Add prediction volume
        if "prediction_count" in metrics_data:
            fig.add_trace(
                go.Bar(
                    x=metrics_data["prediction_count"]["timestamps"],
                    y=metrics_data["prediction_count"]["values"],
                    name="Predictions"
                ),
                row=2, col=1
            )
        
        # Add error timeline
        if "error_rate" in metrics_data:
            fig.add_trace(
                go.Scatter(
                    x=metrics_data["error_rate"]["timestamps"],
                    y=metrics_data["error_rate"]["values"],
                    name="Error Rate",
                    line=dict(color='red')
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title=f"Model Performance Dashboard - Last {time_window.total_seconds()/3600:.0f} hours",
            showlegend=True
        )
        
        return fig
    
    def create_performance_summary(self, performance_stats: Dict) -> go.Figure:
        """Create a summary visualization of key performance metrics"""
        fig = go.Figure()
        
        metrics = list(performance_stats.keys())
        values = [performance_stats[m]["current_value"] for m in metrics]
        trends = [performance_stats[m]["trend"] for m in metrics]
        
        # Create gauge charts for each metric
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=values[i],
                delta={'reference': values[i] - trends[i]},
                gauge={
                    'axis': {'range': [None, max(values[i] * 1.5, 1)]},
                    'steps': [
                        {'range': [0, values[i]], 'color': "lightgray"},
                        {'range': [values[i], values[i] * 1.5], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': values[i]
                    }
                },
                title={'text': metric.replace('_', ' ').title()},
                domain={'row': i // 2, 'column': i % 2}
            ))
        
        fig.update_layout(
            grid={'rows': (len(metrics) + 1) // 2, 'columns': 2, 'pattern': "independent"},
            height=200 * ((len(metrics) + 1) // 2)
        )
        
        return fig 