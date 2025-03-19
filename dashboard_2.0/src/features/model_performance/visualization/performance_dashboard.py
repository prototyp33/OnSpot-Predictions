"""
Performance Dashboard Module
Provides interactive dashboard components for model performance visualization
"""

from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas as pd
from dash import html, dcc
from dash.dependencies import Input, Output, State
from src.shared.visualization_components.base import BaseDashboard
from src.features.model_performance.business_logic.metrics_calculator import MetricResult
from src.features.model_performance.data_access.metrics_repository import MetricsRepository

class ModelPerformanceDashboard(BaseDashboard):
    """Interactive dashboard for model performance monitoring"""
    
    def __init__(
        self,
        model_id: str,
        metrics_repository: MetricsRepository,
        update_interval: int = 60000  # 1 minute
    ):
        super().__init__()
        self.model_id = model_id
        self.repository = metrics_repository
        self.update_interval = update_interval
        
    def create_layout(self) -> html.Div:
        """Create the dashboard layout"""
        return html.Div([
            # Header
            html.Div([
                html.H1(f"Model Performance Dashboard - {self.model_id}"),
                html.Div([
                    dcc.Dropdown(
                        id='time-range-selector',
                        options=[
                            {'label': 'Last Hour', 'value': '1H'},
                            {'label': 'Last 24 Hours', 'value': '24H'},
                            {'label': 'Last Week', 'value': '7D'},
                            {'label': 'Last Month', 'value': '30D'},
                            {'label': 'Custom', 'value': 'custom'}
                        ],
                        value='24H'
                    ),
                    dcc.DatePickerRange(
                        id='custom-date-range',
                        display_format='YYYY-MM-DD',
                        start_date=datetime.now() - timedelta(days=7),
                        end_date=datetime.now(),
                        style={'display': 'none'}
                    )
                ], className='dashboard-controls')
            ], className='dashboard-header'),
            
            # Main content
            html.Div([
                # Summary metrics
                html.Div([
                    html.H2("Current Performance"),
                    dcc.Graph(id='performance-summary')
                ], className='summary-section'),
                
                # Detailed metrics
                html.Div([
                    html.H2("Performance Metrics"),
                    dcc.Graph(id='metrics-timeline')
                ], className='metrics-section'),
                
                # Threshold violations
                html.Div([
                    html.H2("Threshold Violations"),
                    html.Div(id='violations-table')
                ], className='violations-section')
            ], className='dashboard-content'),
            
            # Update interval
            dcc.Interval(
                id='update-interval',
                interval=self.update_interval,
                n_intervals=0
            )
        ], className='performance-dashboard')
    
    def register_callbacks(self) -> None:
        """Register dashboard callbacks"""
        
        @self.app.callback(
            Output('custom-date-range', 'style'),
            Input('time-range-selector', 'value')
        )
        def toggle_custom_date_range(selected_range):
            return {'display': 'block'} if selected_range == 'custom' else {'display': 'none'}
        
        @self.app.callback(
            [Output('performance-summary', 'figure'),
             Output('metrics-timeline', 'figure'),
             Output('violations-table', 'children')],
            [Input('update-interval', 'n_intervals'),
             Input('time-range-selector', 'value')],
            [State('custom-date-range', 'start_date'),
             State('custom-date-range', 'end_date')]
        )
        async def update_dashboard(n_intervals, time_range, custom_start, custom_end):
            # Calculate time range
            end_time = datetime.now()
            if time_range == 'custom':
                start_time = datetime.fromisoformat(custom_start)
                end_time = datetime.fromisoformat(custom_end)
            else:
                duration = {
                    '1H': timedelta(hours=1),
                    '24H': timedelta(days=1),
                    '7D': timedelta(days=7),
                    '30D': timedelta(days=30)
                }
                start_time = end_time - duration[time_range]
            
            # Fetch metrics data
            metrics_data = await self.repository.get_metric_history(
                self.model_id,
                ['rmse', 'mae', 'r2', 'mape'],
                start_time,
                end_time
            )
            
            # Create visualizations
            summary_fig = self._create_summary_figure(metrics_data)
            timeline_fig = self._create_timeline_figure(metrics_data)
            violations_table = self._create_violations_table(metrics_data)
            
            return summary_fig, timeline_fig, violations_table
    
    def _create_summary_figure(self, metrics_data: pd.DataFrame) -> go.Figure:
        """Create summary visualization with current metrics"""
        latest_metrics = metrics_data.groupby('metric_name').last()
        
        fig = go.Figure()
        
        for metric in latest_metrics.index:
            value = latest_metrics.loc[metric, 'value']
            ci_lower = latest_metrics.loc[metric, 'confidence_interval_lower']
            ci_upper = latest_metrics.loc[metric, 'confidence_interval_upper']
            
            fig.add_trace(go.Indicator(
                mode="number+delta+gauge",
                value=value,
                delta={'reference': value * 0.9},  # Example reference
                gauge={
                    'axis': {'range': [None, max(value * 2, 1)]},
                    'bar': {'color': "darkblue"},
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': value
                    }
                },
                title={'text': metric},
                domain={'row': 0, 'column': len(fig.data)}
            ))
        
        fig.update_layout(
            grid={'rows': 1, 'columns': len(latest_metrics)},
            height=250
        )
        
        return fig
    
    def _create_timeline_figure(self, metrics_data: pd.DataFrame) -> go.Figure:
        """Create detailed timeline visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["RMSE", "MAE", "R²", "MAPE"]
        )
        
        metrics_mapping = {
            'rmse': (1, 1),
            'mae': (1, 2),
            'r2': (2, 1),
            'mape': (2, 2)
        }
        
        for metric, (row, col) in metrics_mapping.items():
            metric_data = metrics_data[metrics_data['metric_name'] == metric]
            
            # Main line
            fig.add_trace(
                go.Scatter(
                    x=metric_data['timestamp'],
                    y=metric_data['value'],
                    name=metric.upper(),
                    mode='lines+markers'
                ),
                row=row, col=col
            )
            
            # Confidence interval
            if 'confidence_interval_lower' in metric_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=metric_data['timestamp'],
                        y=metric_data['confidence_interval_upper'],
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,100,80,0.2)',
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=metric_data['timestamp'],
                        y=metric_data['confidence_interval_lower'],
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,100,80,0.2)',
                        name=f'{metric.upper()} Confidence Interval'
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Performance Metrics Over Time"
        )
        
        return fig
    
    def _create_violations_table(self, metrics_data: pd.DataFrame) -> html.Table:
        """Create table of threshold violations"""
        violations = []
        
        for metric in metrics_data['metric_name'].unique():
            metric_data = metrics_data[metrics_data['metric_name'] == metric]
            threshold = self.get_threshold(metric)
            
            if metric_data['value'].iloc[-1] > threshold:
                violations.append({
                    'metric': metric,
                    'current_value': metric_data['value'].iloc[-1],
                    'threshold': threshold,
                    'violation_time': metric_data['timestamp'].iloc[-1]
                })
        
        if not violations:
            return html.Div("No threshold violations detected", className='no-violations')
        
        return html.Table(
            [html.Tr([html.Th(col) for col in ['Metric', 'Current Value', 'Threshold', 'Time']])] +
            [html.Tr([
                html.Td(v['metric']),
                html.Td(f"{v['current_value']:.4f}"),
                html.Td(f"{v['threshold']:.4f}"),
                html.Td(v['violation_time'].strftime('%Y-%m-%d %H:%M:%S'))
            ]) for v in violations],
            className='violations-table'
        )
    
    def get_threshold(self, metric_name: str) -> float:
        """Get threshold value for a metric"""
        # Example thresholds - in practice, these would come from configuration
        thresholds = {
            'rmse': 0.1,
            'mae': 0.08,
            'mape': 10.0,
            'r2': 0.95
        }
        return thresholds.get(metric_name, float('inf')) 