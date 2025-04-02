"""
Drift Visualization Module
Provides visualization components for data and concept drift
"""

from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from dash import html, dcc
from src.features.model_performance.business_logic.metrics_calculator import MetricResult

class DriftVisualization:
    """Visualization components for drift analysis"""
    
    def create_drift_summary(
        self,
        drift_metrics: Dict[str, MetricResult],
        threshold: float = 0.05
    ) -> go.Figure:
        """
        Create a summary visualization of drift metrics
        
        Args:
            drift_metrics: Dictionary of drift metrics by feature
            threshold: Significance threshold for drift detection
        """
        # Extract feature names and drift statistics
        features = [k for k in drift_metrics.keys() if k != "overall_drift"]
        drift_values = [drift_metrics[f].value for f in features]
        p_values = [drift_metrics[f].metadata["p_value"] for f in features]
        is_significant = [v <= threshold for v in p_values]
        
        # Create heatmap
        fig = go.Figure()
        
        # Add drift magnitude bars
        fig.add_trace(go.Bar(
            x=features,
            y=drift_values,
            name="Drift Magnitude",
            marker_color=['red' if sig else 'blue' for sig in is_significant]
        ))
        
        # Add threshold line
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(features) - 0.5,
            y0=threshold,
            y1=threshold,
            line=dict(
                color="red",
                width=2,
                dash="dash"
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Feature Drift Analysis",
            xaxis_title="Features",
            yaxis_title="Drift Magnitude",
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_feature_distribution_comparison(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_names: List[str],
        feature_types: Dict[str, str],
        max_features: int = 6
    ) -> go.Figure:
        """
        Create distribution comparison plots for features
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            feature_names: List of feature names
            feature_types: Dictionary of feature types
            max_features: Maximum number of features to display
        """
        n_features = min(len(feature_names), max_features)
        n_rows = (n_features + 1) // 2
        
        fig = make_subplots(
            rows=n_rows,
            cols=2,
            subplot_titles=feature_names[:n_features]
        )
        
        for i, feature in enumerate(feature_names[:n_features]):
            row = i // 2 + 1
            col = i % 2 + 1
            
            if feature_types[feature] == 'numerical':
                # Add KDE plots for numerical features
                fig.add_trace(
                    go.Histogram(
                        x=reference_data[:, i],
                        name="Reference",
                        opacity=0.7,
                        nbinsx=30,
                        histnorm='probability density'
                    ),
                    row=row, col=col
                )
                
                fig.add_trace(
                    go.Histogram(
                        x=current_data[:, i],
                        name="Current",
                        opacity=0.7,
                        nbinsx=30,
                        histnorm='probability density'
                    ),
                    row=row, col=col
                )
            else:
                # Add bar plots for categorical features
                categories = np.unique(np.concatenate([reference_data[:, i], current_data[:, i]]))
                ref_counts = np.array([np.sum(reference_data[:, i] == cat) for cat in categories])
                curr_counts = np.array([np.sum(current_data[:, i] == cat) for cat in categories])
                
                # Convert to proportions
                ref_props = ref_counts / len(reference_data)
                curr_props = curr_counts / len(current_data)
                
                fig.add_trace(
                    go.Bar(
                        x=categories,
                        y=ref_props,
                        name="Reference",
                        opacity=0.7
                    ),
                    row=row, col=col
                )
                
                fig.add_trace(
                    go.Bar(
                        x=categories,
                        y=curr_props,
                        name="Current",
                        opacity=0.7
                    ),
                    row=row, col=col
                )
        
        # Update layout
        fig.update_layout(
            height=300 * n_rows,
            showlegend=True,
            title_text="Feature Distribution Comparison"
        )
        
        return fig
    
    def create_concept_drift_visualization(
        self,
        reference_errors: np.ndarray,
        current_errors: np.ndarray,
        drift_result: MetricResult
    ) -> go.Figure:
        """
        Create visualization for concept drift analysis
        
        Args:
            reference_errors: Prediction errors from reference period
            current_errors: Prediction errors from current period
            drift_result: Concept drift detection results
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Error Distribution Comparison", "Error Ratio Over Time"]
        )
        
        # Add error distribution comparison
        fig.add_trace(
            go.Histogram(
                x=reference_errors,
                name="Reference Errors",
                opacity=0.7,
                nbinsx=30,
                histnorm='probability density'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=current_errors,
                name="Current Errors",
                opacity=0.7,
                nbinsx=30,
                histnorm='probability density'
            ),
            row=1, col=1
        )
        
        # Add error ratio indicator
        error_ratio = drift_result.metadata["error_ratio"]
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=error_ratio,
                delta={'reference': 1},
                gauge={
                    'axis': {'range': [0, max(2, error_ratio * 1.2)]},
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 1
                    }
                },
                title={'text': "Error Ratio (Current/Reference)"}
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Concept Drift Analysis"
        )
        
        return fig
    
    def create_drift_monitoring_dashboard(
        self,
        drift_metrics: Dict[str, MetricResult],
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_names: List[str],
        feature_types: Dict[str, str]
    ) -> html.Div:
        """Create a complete drift monitoring dashboard"""
        return html.Div([
            # Summary section
            html.Div([
                html.H2("Drift Analysis Dashboard"),
                html.Div([
                    html.Div([
                        html.H3("Overall Drift Score"),
                        html.H4(f"{drift_metrics['overall_drift'].value:.4f}"),
                        html.P(f"Significant Features: "
                              f"{drift_metrics['overall_drift'].metadata['n_significant_features']} / "
                              f"{drift_metrics['overall_drift'].metadata['total_features']}")
                    ], className="summary-stats")
                ], className="summary-container")
            ], className="dashboard-header"),
            
            # Drift magnitude plot
            html.Div([
                html.H3("Feature Drift Analysis"),
                dcc.Graph(figure=self.create_drift_summary(drift_metrics))
            ], className="drift-summary-section"),
            
            # Distribution comparisons
            html.Div([
                html.H3("Feature Distribution Comparison"),
                dcc.Graph(figure=self.create_feature_distribution_comparison(
                    reference_data,
                    current_data,
                    feature_names,
                    feature_types
                ))
            ], className="distribution-section")
        ], className="drift-dashboard") 