"""
Data Quality Visualization Module
Provides visualization components for data quality monitoring
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class DataQualityVisualizer:
    """Visualization components for data quality monitoring"""
    
    def create_quality_overview(
        self,
        quality_metrics: Dict[str, float],
        feature_importance: Dict[str, float]
    ) -> go.Figure:
        """Create overview of data quality metrics"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Overall Quality Score",
                "Feature Importance",
                "Quality Metrics",
                "Issues Distribution"
            )
        )
        
        # Overall quality gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=quality_metrics.get("overall_score", 0) * 100,
                title={"text": "Quality Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "red"},
                        {"range": [50, 80], "color": "yellow"},
                        {"range": [80, 100], "color": "green"}
                    ]
                }
            ),
            row=1, col=1
        )
        
        # Feature importance bar chart
        features = list(feature_importance.keys())
        importance_values = list(feature_importance.values())
        
        fig.add_trace(
            go.Bar(
                x=features,
                y=importance_values,
                name="Feature Importance",
                marker_color="lightblue"
            ),
            row=1, col=2
        )
        
        # Quality metrics breakdown
        metric_names = [k for k in quality_metrics.keys() if k != "overall_score"]
        metric_values = [quality_metrics[k] for k in metric_names]
        
        fig.add_trace(
            go.Bar(
                x=metric_names,
                y=metric_values,
                name="Quality Metrics",
                marker_color="lightgreen"
            ),
            row=2, col=1
        )
        
        # Issues distribution pie chart
        issues = {
            "Missing Values": quality_metrics.get("missing_rate", 0),
            "Out of Range": quality_metrics.get("out_of_range_rate", 0),
            "Distribution Shift": quality_metrics.get("distribution_shift_rate", 0),
            "Correlation Change": quality_metrics.get("correlation_change_rate", 0)
        }
        
        fig.add_trace(
            go.Pie(
                labels=list(issues.keys()),
                values=list(issues.values()),
                hole=0.3
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Data Quality Overview"
        )
        
        return fig
    
    def create_missing_values_heatmap(
        self,
        missing_rates: Dict[str, List[float]],
        feature_names: List[str],
        timestamps: List[datetime]
    ) -> go.Figure:
        """Create heatmap of missing values over time"""
        fig = go.Figure(data=go.Heatmap(
            z=list(missing_rates.values()),
            x=timestamps,
            y=feature_names,
            colorscale="RdYlBu_r",
            colorbar={"title": "Missing Rate"}
        ))
        
        fig.update_layout(
            title="Missing Values Over Time",
            xaxis_title="Time",
            yaxis_title="Features",
            height=400
        )
        
        return fig
    
    def create_correlation_changes(
        self,
        correlation_changes: Dict[str, float],
        reference_correlations: pd.DataFrame,
        current_correlations: pd.DataFrame
    ) -> go.Figure:
        """Create correlation changes visualization"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Reference Correlations", "Current Correlations")
        )
        
        # Reference correlation heatmap
        fig.add_trace(
            go.Heatmap(
                z=reference_correlations.values,
                x=reference_correlations.columns,
                y=reference_correlations.index,
                colorscale="RdBu",
                zmin=-1,
                zmax=1
            ),
            row=1, col=1
        )
        
        # Current correlation heatmap
        fig.add_trace(
            go.Heatmap(
                z=current_correlations.values,
                x=current_correlations.columns,
                y=current_correlations.index,
                colorscale="RdBu",
                zmin=-1,
                zmax=1
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Feature Correlation Changes",
            height=500
        )
        
        return fig
    
    def create_distribution_shifts(
        self,
        distribution_metrics: Dict[str, Dict[str, float]],
        feature_names: List[str]
    ) -> go.Figure:
        """Create distribution shift visualization"""
        fig = make_subplots(rows=2, cols=1)
        
        # KS statistics
        ks_stats = [
            distribution_metrics[f].get("ks_statistic", 0)
            for f in feature_names
        ]
        
        fig.add_trace(
            go.Bar(
                x=feature_names,
                y=ks_stats,
                name="KS Statistic",
                marker_color="lightblue"
            ),
            row=1, col=1
        )
        
        # P-values
        p_values = [
            distribution_metrics[f].get("p_value", 1)
            for f in feature_names
        ]
        
        fig.add_trace(
            go.Scatter(
                x=feature_names,
                y=p_values,
                name="P-value",
                mode="lines+markers",
                line={"color": "red"}
            ),
            row=2, col=1
        )
        
        # Add threshold line
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(feature_names) - 0.5,
            y0=0.05,  # Common significance level
            y1=0.05,
            line={"color": "red", "dash": "dash"},
            row=2, col=1
        )
        
        fig.update_layout(
            title="Distribution Shifts by Feature",
            xaxis2_title="Features",
            yaxis_title="KS Statistic",
            yaxis2_title="P-value",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_feature_distributions(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_names: List[str],
        max_features: int = 6
    ) -> go.Figure:
        """Create feature distribution comparison plots"""
        n_features = min(len(feature_names), max_features)
        n_rows = (n_features + 1) // 2
        
        fig = make_subplots(
            rows=n_rows,
            cols=2,
            subplot_titles=feature_names[:n_features]
        )
        
        for i in range(n_features):
            row = i // 2 + 1
            col = i % 2 + 1
            
            # Add reference distribution
            fig.add_trace(
                go.Histogram(
                    x=reference_data[:, i],
                    name="Reference",
                    opacity=0.7,
                    nbinsx=30,
                    histnorm="probability"
                ),
                row=row, col=col
            )
            
            # Add current distribution
            fig.add_trace(
                go.Histogram(
                    x=current_data[:, i],
                    name="Current",
                    opacity=0.7,
                    nbinsx=30,
                    histnorm="probability"
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=300 * n_rows,
            showlegend=True,
            title_text="Feature Distribution Comparison"
        )
        
        return fig 