"""
Monitoring Dashboard Module
Provides comprehensive visualization components for model monitoring
"""

from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import json

class MonitoringDashboard:
    """Comprehensive monitoring dashboard"""
    
    def __init__(
        self,
        model_ids: List[str],
        metrics_repository,
        alert_manager,
        health_monitor,
        model_comparator,
        data_quality_monitor,
        auto_response_manager,
        update_interval: int = 60  # seconds
    ):
        self.model_ids = model_ids
        self.metrics_repository = metrics_repository
        self.alert_manager = alert_manager
        self.health_monitor = health_monitor
        self.model_comparator = model_comparator
        self.data_quality_monitor = data_quality_monitor
        self.auto_response_manager = auto_response_manager
        self.update_interval = update_interval
    
    def create_layout(self) -> html.Div:
        """Create the main dashboard layout"""
        return html.Div([
            # Header
            html.Div([
                html.H1("Model Monitoring Dashboard"),
                self._create_model_selector(),
                self._create_time_range_selector()
            ], className="dashboard-header"),
            
            # Main content
            html.Div([
                # Left column - Overview and Alerts
                html.Div([
                    self._create_overview_section(),
                    self._create_alerts_section(),
                    self._create_automated_actions_section()
                ], className="column left-column"),
                
                # Middle column - Performance and Health
                html.Div([
                    self._create_performance_section(),
                    self._create_health_section()
                ], className="column middle-column"),
                
                # Right column - Data Quality and Drift
                html.Div([
                    self._create_data_quality_section(),
                    self._create_drift_section()
                ], className="column right-column")
            ], className="dashboard-content"),
            
            # Model Comparison Tab
            dcc.Tab(label="Model Comparison", children=[
                self._create_model_comparison_section()
            ])
        ], className="monitoring-dashboard")
    
    def _create_model_selector(self) -> html.Div:
        """Create model selection dropdown"""
        return html.Div([
            html.Label("Select Model:"),
            dcc.Dropdown(
                id="model-selector",
                options=[{"label": mid, "value": mid} for mid in self.model_ids],
                value=self.model_ids[0],
                clearable=False
            )
        ], className="selector-container")
    
    def _create_time_range_selector(self) -> html.Div:
        """Create time range selection controls"""
        return html.Div([
            html.Label("Time Range:"),
            dcc.Dropdown(
                id="time-range",
                options=[
                    {"label": "Last Hour", "value": "1H"},
                    {"label": "Last Day", "value": "1D"},
                    {"label": "Last Week", "value": "1W"},
                    {"label": "Last Month", "value": "1M"},
                    {"label": "Custom", "value": "custom"}
                ],
                value="1D",
                clearable=False
            ),
            html.Div(
                dcc.DateTimePickerRange(
                    id="custom-date-range",
                    className="date-range-picker"
                ),
                id="custom-range-container",
                style={"display": "none"}
            )
        ], className="selector-container")
    
    def _create_overview_section(self) -> html.Div:
        """Create overview section with key metrics"""
        return html.Div([
            html.H2("Overview"),
            dcc.Loading(
                id="overview-content",
                children=[
                    html.Div([
                        html.Div([
                            html.H4("Performance Score"),
                            html.Div(id="performance-score")
                        ], className="metric-card"),
                        html.Div([
                            html.H4("Health Score"),
                            html.Div(id="health-score")
                        ], className="metric-card"),
                        html.Div([
                            html.H4("Data Quality Score"),
                            html.Div(id="quality-score")
                        ], className="metric-card")
                    ], className="metrics-container")
                ]
            )
        ], className="section overview-section")
    
    def _create_alerts_section(self) -> html.Div:
        """Create alerts section"""
        return html.Div([
            html.H2("Active Alerts"),
            dcc.Loading(
                id="alerts-content",
                children=[
                    html.Div(id="alerts-list"),
                    dcc.Graph(id="alerts-timeline")
                ]
            )
        ], className="section alerts-section")
    
    def _create_automated_actions_section(self) -> html.Div:
        """Create automated actions section"""
        return html.Div([
            html.H2("Automated Actions"),
            dcc.Loading(
                id="actions-content",
                children=[
                    html.Div([
                        html.H4("Recommended Actions"),
                        html.Div(id="recommended-actions-list"),
                        html.H4("Action History"),
                        dcc.Graph(id="action-history-timeline"),
                        html.Div([
                            html.H4("Action Statistics"),
                            dcc.Graph(id="action-stats")
                        ])
                    ])
                ]
            )
        ], className="section actions-section")
    
    def _create_performance_section(self) -> html.Div:
        """Create performance metrics section"""
        return html.Div([
            html.H2("Performance Metrics"),
            dcc.Loading(
                id="performance-content",
                children=[
                    dcc.Graph(id="performance-trends"),
                    html.Div([
                        dcc.Graph(id="metric-distribution"),
                        dcc.Graph(id="prediction-scatter")
                    ], className="performance-charts")
                ]
            )
        ], className="section performance-section")
    
    def _create_health_section(self) -> html.Div:
        """Create health monitoring section"""
        return html.Div([
            html.H2("Model Health"),
            dcc.Loading(
                id="health-content",
                children=[
                    dcc.Graph(id="health-metrics"),
                    html.Div([
                        dcc.Graph(id="latency-trend"),
                        dcc.Graph(id="error-rate-trend")
                    ], className="health-charts"),
                    dcc.Graph(id="resource-usage")
                ]
            )
        ], className="section health-section")
    
    def _create_data_quality_section(self) -> html.Div:
        """Create data quality section"""
        return html.Div([
            html.H2("Data Quality"),
            dcc.Loading(
                id="quality-content",
                children=[
                    dcc.Graph(id="quality-metrics"),
                    html.Div([
                        dcc.Graph(id="missing-values-heatmap"),
                        dcc.Graph(id="correlation-changes")
                    ], className="quality-charts"),
                    dcc.Graph(id="distribution-shifts")
                ]
            )
        ], className="section quality-section")
    
    def _create_drift_section(self) -> html.Div:
        """Create drift monitoring section"""
        return html.Div([
            html.H2("Drift Analysis"),
            dcc.Loading(
                id="drift-content",
                children=[
                    dcc.Graph(id="drift-summary"),
                    html.Div([
                        dcc.Graph(id="feature-distributions"),
                        dcc.Graph(id="concept-drift")
                    ], className="drift-charts")
                ]
            )
        ], className="section drift-section")
    
    def _create_model_comparison_section(self) -> html.Div:
        """Create model comparison section"""
        return html.Div([
            html.H2("Model Comparison"),
            dcc.Loading(
                id="comparison-content",
                children=[
                    dcc.Graph(id="model-rankings"),
                    html.Div([
                        dcc.Graph(id="metric-comparison"),
                        dcc.Graph(id="performance-matrix")
                    ], className="comparison-charts"),
                    dcc.Graph(id="trend-comparison")
                ]
            )
        ], className="section comparison-section")
    
    def create_performance_trend(self, model_id: str, metric: str, time_window: timedelta) -> go.Figure:
        """Create performance trend visualization"""
        fig = go.Figure()
        # Add performance trend line
        # Add confidence intervals
        # Add threshold lines
        return fig
    
    def create_health_metrics(self, model_id: str, time_window: timedelta) -> go.Figure:
        """Create health metrics visualization"""
        fig = make_subplots(rows=2, cols=2)
        # Add latency plot
        # Add error rate plot
        # Add resource usage
        # Add request volume
        return fig
    
    def create_quality_metrics(self, model_id: str, time_window: timedelta) -> go.Figure:
        """Create data quality visualization"""
        fig = make_subplots(rows=2, cols=2)
        # Add missing values plot
        # Add distribution shifts
        # Add correlation changes
        # Add feature importance
        return fig
    
    def create_drift_summary(self, model_id: str, time_window: timedelta) -> go.Figure:
        """Create drift summary visualization"""
        fig = make_subplots(rows=2, cols=1)
        # Add feature drift plot
        # Add concept drift plot
        return fig
    
    def create_model_comparison(self, metrics: List[str], time_window: timedelta) -> go.Figure:
        """Create model comparison visualization"""
        fig = make_subplots(rows=2, cols=2)
        # Add performance comparison
        # Add ranking plot
        # Add trend comparison
        # Add correlation matrix
        return fig
    
    def create_alerts_timeline(self, time_window: timedelta) -> go.Figure:
        """Create alerts timeline visualization"""
        fig = go.Figure()
        # Add alert markers
        # Add severity colors
        # Add hover information
        return fig
    
    def create_recommended_actions_list(self, actions: List[RecommendedAction]) -> html.Div:
        """Create list of recommended actions"""
        return html.Div([
            html.Div([
                html.Div([
                    html.H5(action.description),
                    html.P(f"Priority: {action.priority}"),
                    html.P(f"Confidence: {action.confidence:.2%}"),
                    html.P(f"Source: {action.issue_source}"),
                    html.Button(
                        "Execute Action",
                        id={"type": "action-button", "index": i},
                        className="action-button"
                    )
                ], className=f"action-card priority-{action.priority}")
            for i, action in enumerate(actions)
            ])
        ], className="recommended-actions")

    def create_action_history_timeline(self, history: List[RecommendedAction]) -> go.Figure:
        """Create timeline visualization of action history"""
        fig = go.Figure()
        
        # Create timeline
        for action_type in ActionType:
            type_actions = [a for a in history if a.action_type == action_type]
            if type_actions:
                fig.add_trace(go.Scatter(
                    x=[a.timestamp for a in type_actions],
                    y=[action_type.value for _ in type_actions],
                    mode="markers",
                    name=action_type.value,
                    marker=dict(
                        size=[a.priority * 5 for a in type_actions],
                        color=[a.confidence * 100 for a in type_actions],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Confidence %")
                    ),
                    text=[a.description for a in type_actions],
                    hovertemplate="<b>%{text}</b><br>" +
                                "Time: %{x}<br>" +
                                "Confidence: %{marker.color:.1f}%<br>" +
                                "<extra></extra>"
                ))
        
        fig.update_layout(
            title="Action History Timeline",
            xaxis_title="Time",
            yaxis_title="Action Type",
            showlegend=True,
            height=400
        )
        
        return fig

    def create_action_statistics(self, history: List[RecommendedAction]) -> go.Figure:
        """Create statistics visualization for actions"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Actions by Type",
                "Actions by Priority",
                "Average Confidence by Type",
                "Actions Over Time"
            )
        )
        
        # Actions by type
        action_counts = {}
        for action in history:
            action_counts[action.action_type.value] = action_counts.get(action.action_type.value, 0) + 1
        
        fig.add_trace(
            go.Bar(
                x=list(action_counts.keys()),
                y=list(action_counts.values()),
                name="Count"
            ),
            row=1, col=1
        )
        
        # Actions by priority
        priority_counts = {}
        for action in history:
            priority_counts[action.priority] = priority_counts.get(action.priority, 0) + 1
        
        fig.add_trace(
            go.Bar(
                x=list(priority_counts.keys()),
                y=list(priority_counts.values()),
                name="Count",
                marker_color="orange"
            ),
            row=1, col=2
        )
        
        # Average confidence by type
        confidence_by_type = {}
        count_by_type = {}
        for action in history:
            if action.action_type.value not in confidence_by_type:
                confidence_by_type[action.action_type.value] = 0
                count_by_type[action.action_type.value] = 0
            confidence_by_type[action.action_type.value] += action.confidence
            count_by_type[action.action_type.value] += 1
        
        avg_confidence = {
            action_type: conf / count_by_type[action_type]
            for action_type, conf in confidence_by_type.items()
        }
        
        fig.add_trace(
            go.Bar(
                x=list(avg_confidence.keys()),
                y=list(avg_confidence.values()),
                name="Avg Confidence",
                marker_color="green"
            ),
            row=2, col=1
        )
        
        # Actions over time
        timestamps = [action.timestamp for action in history]
        fig.add_trace(
            go.Histogram(
                x=timestamps,
                nbinsx=20,
                name="Actions",
                marker_color="purple"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=False
        )
        
        return fig
    
    def register_callbacks(self, app) -> None:
        """Register all dashboard callbacks"""
        
        @app.callback(
            [Output("performance-score", "children"),
             Output("health-score", "children"),
             Output("quality-score", "children"),
             Output("recommended-actions-list", "children"),
             Output("action-history-timeline", "figure"),
             Output("action-stats", "figure")],
            [Input("model-selector", "value"),
             Input("time-range", "value")]
        )
        def update_overview(model_id: str, time_range: str) -> tuple:
            # Update overview metrics
            current_metrics = self.metrics_repository.get_latest_metrics(model_id)
            quality_metrics = self.data_quality_monitor.get_latest_metrics(model_id)
            health_metrics = self.health_monitor.get_latest_metrics(model_id)
            
            recommended_actions = self.auto_response_manager.analyze_and_respond(
                model_id,
                current_metrics,
                quality_metrics,
                health_metrics
            )
            
            # Get action history
            action_history = self.auto_response_manager.get_action_history()
            
            return (
                "85%",
                "92%",
                "78%",
                self.create_recommended_actions_list(recommended_actions),
                self.create_action_history_timeline(action_history),
                self.create_action_statistics(action_history)
            )
        
        @app.callback(
            Output("action-status", "children"),
            [Input({"type": "action-button", "index": ALL}, "n_clicks")],
            [State("recommended-actions-list", "children")]
        )
        def execute_action(n_clicks, actions_list):
            ctx = dash.callback_context
            if not ctx.triggered:
                return ""
            
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            action_index = json.loads(button_id)["index"]
            action = self.auto_response_manager.action_history[action_index]
            
            success = self.auto_response_manager.execute_action(action)
            return f"Action {'executed successfully' if success else 'failed'}"
        
        # Add more callbacks for other components
        
    def run(self, host: str = "0.0.0.0", port: int = 8050) -> None:
        """Run the dashboard"""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        app.layout = self.create_layout()
        self.register_callbacks(app)
        app.run_server(host=host, port=port) 