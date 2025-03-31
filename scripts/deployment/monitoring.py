#!/usr/bin/env python3
"""
Monitoring Service for OnSpot Predictive Model.

This service is responsible for:
- Collecting metrics from all deployments
- Detecting model drift
- Alerting on abnormal behavior
- Generating performance reports
- Monitoring system health

It connects to Prometheus for metrics collection and provides a dashboard
for visualizing model performance and deployment stats.
"""

import os
import sys
import logging
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import threading
from flask import Flask, jsonify, request, render_template
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import io
import base64
from prometheus_client.parser import text_string_to_metric_families

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from model versioning system
from scripts.models import ModelVersioning, ModelRegistry

# Set up logging
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
MODEL_REGISTRY_PATH = os.environ.get('MODEL_REGISTRY_PATH', './model_registry')
PRODUCTION_MODELS_PATH = os.environ.get('PRODUCTION_MODELS_PATH', './production_models')
PROMETHEUS_URL = os.environ.get('PROMETHEUS_URL', 'http://localhost:9090')
PREDICTION_API = os.environ.get('PREDICTION_API', 'http://prediction-api:5000')
GATEWAY_API = os.environ.get('GATEWAY_API', 'http://deployment-gateway:8080')
MONITORING_INTERVAL = int(os.environ.get('MONITORING_INTERVAL', '300'))  # 5 minutes
DRIFT_THRESHOLD = float(os.environ.get('DRIFT_THRESHOLD', '0.1'))  # 10% drift
ALERT_WEBHOOK_URL = os.environ.get('ALERT_WEBHOOK_URL', '')

# Initialize model versioning
versioning = ModelVersioning(
    registry_path=MODEL_REGISTRY_PATH,
    production_models_path=PRODUCTION_MODELS_PATH
)

# Storage for collected metrics
performance_metrics = {}
drift_metrics = {}
health_status = {}
alerts = []

# Template directory
os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
template_path = os.path.join(os.path.dirname(__file__), 'templates', 'dashboard.html')

# Create default dashboard template if it doesn't exist
if not os.path.exists(template_path):
    default_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>OnSpot Model Monitoring Dashboard</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
            .card { background-color: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
                    padding: 20px; margin-bottom: 20px; }
            .row { display: flex; flex-wrap: wrap; margin: 0 -10px; }
            .col { flex: 1; padding: 0 10px; min-width: 300px; }
            h1, h2, h3 { margin-top: 0; }
            .metric { font-size: 24px; font-weight: bold; margin: 10px 0; }
            .chart { width: 100%; max-width: 100%; height: auto; margin: 15px 0; }
            table { width: 100%; border-collapse: collapse; }
            table th, table td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
            table th { background-color: #f2f2f2; }
            .alert { background-color: #f8d7da; color: #721c24; padding: 10px; margin-bottom: 10px; border-radius: 3px; }
            .status-healthy { color: #28a745; }
            .status-warning { color: #ffc107; }
            .status-error { color: #dc3545; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>OnSpot Model Monitoring Dashboard</h1>
                <p>Last updated: {{ last_updated }}</p>
            </div>
            
            <div class="row">
                <div class="col">
                    <div class="card">
                        <h2>System Health</h2>
                        <table>
                            <tr>
                                <th>Service</th>
                                <th>Status</th>
                                <th>Last Checked</th>
                            </tr>
                            {% for service, data in health_status.items() %}
                            <tr>
                                <td>{{ service }}</td>
                                <td class="status-{{ data.status }}">{{ data.status }}</td>
                                <td>{{ data.last_checked }}</td>
                            </tr>
                            {% endfor %}
                        </table>
                    </div>
                </div>
                
                <div class="col">
                    <div class="card">
                        <h2>Active Alerts</h2>
                        {% if alerts %}
                            {% for alert in alerts %}
                            <div class="alert">
                                <strong>{{ alert.type }}</strong> - {{ alert.message }} ({{ alert.timestamp }})
                            </div>
                            {% endfor %}
                        {% else %}
                            <p>No active alerts</p>
                        {% endif %}
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col">
                    <div class="card">
                        <h2>Model Performance</h2>
                        {% if performance_charts %}
                            {% for chart in performance_charts %}
                            <div class="chart">
                                <h3>{{ chart.title }}</h3>
                                <img src="data:image/png;base64,{{ chart.data }}" alt="{{ chart.title }}">
                            </div>
                            {% endfor %}
                        {% else %}
                            <p>No performance data available</p>
                        {% endif %}
                    </div>
                </div>
                
                <div class="col">
                    <div class="card">
                        <h2>Model Drift</h2>
                        {% if drift_charts %}
                            {% for chart in drift_charts %}
                            <div class="chart">
                                <h3>{{ chart.title }}</h3>
                                <img src="data:image/png;base64,{{ chart.data }}" alt="{{ chart.title }}">
                            </div>
                            {% endfor %}
                        {% else %}
                            <p>No drift data available</p>
                        {% endif %}
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col">
                    <div class="card">
                        <h2>Traffic Distribution</h2>
                        {% if traffic_chart %}
                            <div class="chart">
                                <img src="data:image/png;base64,{{ traffic_chart }}" alt="Traffic Distribution">
                            </div>
                        {% else %}
                            <p>No traffic data available</p>
                        {% endif %}
                    </div>
                </div>
                
                <div class="col">
                    <div class="card">
                        <h2>Latency</h2>
                        {% if latency_chart %}
                            <div class="chart">
                                <img src="data:image/png;base64,{{ latency_chart }}" alt="Latency Distribution">
                            </div>
                        {% else %}
                            <p>No latency data available</p>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    with open(template_path, 'w') as f:
        f.write(default_template)

def generate_chart(data, title, chart_type='line', x_label='Time', y_label='Value'):
    """Generate a chart image as base64 string."""
    plt.figure(figsize=(10, 6))
    
    if chart_type == 'line':
        if isinstance(data, dict):
            for label, values in data.items():
                if isinstance(values, list) and len(values) > 0:
                    plt.plot(values, label=label)
        else:
            plt.plot(data)
    elif chart_type == 'bar':
        if isinstance(data, dict):
            plt.bar(list(data.keys()), list(data.values()))
        else:
            plt.bar(range(len(data)), data)
    elif chart_type == 'pie':
        if isinstance(data, dict):
            plt.pie(list(data.values()), labels=list(data.keys()), autopct='%1.1f%%')
        
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    if chart_type == 'line' and isinstance(data, dict) and len(data) > 1:
        plt.legend()
    
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

def fetch_prometheus_metrics(query):
    """Fetch metrics from Prometheus using PromQL."""
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={'query': query},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('data', {}).get('result', [])
        else:
            logger.error(f"Failed to fetch metrics from Prometheus: {response.status_code}")
            return []
    
    except Exception as e:
        logger.error(f"Error fetching Prometheus metrics: {str(e)}")
        return []

def check_model_drift():
    """Check for model drift by comparing metrics."""
    # Get models from registry
    models = versioning.registry.list_models()
    drift_results = {}
    
    for model_info in models:
        model_name = model_info.get('name')
        if not model_name:
            continue
        
        # Get metrics for this model
        prediction_metrics = fetch_prometheus_metrics(
            f'ground_truth_metrics{{model_name="{model_name}"}}'
        )
        
        # Skip if no metrics available
        if not prediction_metrics:
            continue
        
        # Get model metadata to compare with expected metrics
        model_id = model_info.get('model_id')
        if not model_id:
            continue
            
        try:
            _, metadata = versioning.registry.load_model(model_id, with_metadata=True)
            expected_metrics = metadata.get('performance_metrics', {})
            
            if not expected_metrics:
                continue
                
            # Calculate drift for each metric
            metric_drift = {}
            for metric in prediction_metrics:
                metric_name = metric.get('metric', {}).get('metric_name')
                if not metric_name or metric_name not in expected_metrics:
                    continue
                    
                expected_value = expected_metrics.get(metric_name, 0)
                actual_value = float(metric.get('value', [0, 0])[1])
                
                if expected_value != 0:
                    drift_percentage = abs((actual_value - expected_value) / expected_value)
                    metric_drift[metric_name] = drift_percentage
                    
                    # Alert if drift exceeds threshold
                    if drift_percentage > DRIFT_THRESHOLD:
                        create_alert(
                            'model_drift',
                            f"Model {model_name} has {metric_name} drift of {drift_percentage:.2%}, " +
                            f"exceeding threshold of {DRIFT_THRESHOLD:.2%}"
                        )
            
            drift_results[model_name] = metric_drift
            
        except Exception as e:
            logger.error(f"Error checking drift for model {model_name}: {str(e)}")
    
    # Store drift results
    drift_metrics.update(drift_results)
    return drift_results

def check_system_health():
    """Check health of all services."""
    health_results = {}
    
    # Check Prediction API
    try:
        response = requests.get(f"{PREDICTION_API}/health", timeout=5)
        if response.status_code == 200:
            health_results['prediction_api'] = {
                'status': 'healthy',
                'last_checked': datetime.now().isoformat(),
                'details': response.json()
            }
        else:
            health_results['prediction_api'] = {
                'status': 'error',
                'last_checked': datetime.now().isoformat(),
                'details': {'error': f"Status code {response.status_code}"}
            }
    except Exception as e:
        health_results['prediction_api'] = {
            'status': 'error',
            'last_checked': datetime.now().isoformat(),
            'details': {'error': str(e)}
        }
        create_alert('service_down', f"Prediction API is unreachable: {str(e)}")
    
    # Check Gateway API
    try:
        response = requests.get(f"{GATEWAY_API}/health", timeout=5)
        if response.status_code == 200:
            health_results['gateway_api'] = {
                'status': 'healthy',
                'last_checked': datetime.now().isoformat(),
                'details': response.json()
            }
        else:
            health_results['gateway_api'] = {
                'status': 'error',
                'last_checked': datetime.now().isoformat(),
                'details': {'error': f"Status code {response.status_code}"}
            }
    except Exception as e:
        health_results['gateway_api'] = {
            'status': 'error',
            'last_checked': datetime.now().isoformat(),
            'details': {'error': str(e)}
        }
        create_alert('service_down', f"Gateway API is unreachable: {str(e)}")
    
    # Check Prometheus
    try:
        response = requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=5)
        if response.status_code == 200:
            health_results['prometheus'] = {
                'status': 'healthy',
                'last_checked': datetime.now().isoformat(),
                'details': {}
            }
        else:
            health_results['prometheus'] = {
                'status': 'error',
                'last_checked': datetime.now().isoformat(),
                'details': {'error': f"Status code {response.status_code}"}
            }
    except Exception as e:
        health_results['prometheus'] = {
            'status': 'error',
            'last_checked': datetime.now().isoformat(),
            'details': {'error': str(e)}
        }
        create_alert('service_down', f"Prometheus is unreachable: {str(e)}")
    
    # Update global health status
    health_status.update(health_results)
    return health_results

def collect_performance_metrics():
    """Collect performance metrics from Prometheus."""
    
    # Get prediction request counts
    prediction_counts = fetch_prometheus_metrics(
        'sum by (model_name, model_version) (prediction_requests_total)'
    )
    
    # Get prediction latency
    prediction_latency = fetch_prometheus_metrics(
        'histogram_quantile(0.95, sum by (model_name, model_version, le) (rate(prediction_latency_seconds_bucket[5m])))'
    )
    
    # Get prediction errors
    prediction_errors = fetch_prometheus_metrics(
        'sum by (model_name, model_version) (prediction_errors_total)'
    )
    
    # Get gateway metrics
    gateway_requests = fetch_prometheus_metrics(
        'sum by (strategy, destination) (gateway_requests_total)'
    )
    
    # Get ground truth metrics
    ground_truth = fetch_prometheus_metrics(
        'sum by (model_name, model_version, metric_name) (ground_truth_metrics)'
    )
    
    # Process and store metrics
    performance_data = {
        'prediction_counts': {},
        'prediction_latency': {},
        'prediction_errors': {},
        'gateway_requests': {},
        'ground_truth': {}
    }
    
    for item in prediction_counts:
        model_name = item.get('metric', {}).get('model_name', 'unknown')
        model_version = item.get('metric', {}).get('model_version', 'unknown')
        count = float(item.get('value', [0, 0])[1])
        key = f"{model_name}_{model_version}"
        performance_data['prediction_counts'][key] = count
    
    for item in prediction_latency:
        model_name = item.get('metric', {}).get('model_name', 'unknown')
        model_version = item.get('metric', {}).get('model_version', 'unknown')
        latency = float(item.get('value', [0, 0])[1])
        key = f"{model_name}_{model_version}"
        performance_data['prediction_latency'][key] = latency
        
        # Alert on high latency
        if latency > 0.5:  # 500ms
            create_alert(
                'high_latency',
                f"Model {model_name} (version {model_version}) has high P95 latency: {latency*1000:.2f}ms"
            )
    
    for item in prediction_errors:
        model_name = item.get('metric', {}).get('model_name', 'unknown')
        model_version = item.get('metric', {}).get('model_version', 'unknown')
        errors = float(item.get('value', [0, 0])[1])
        key = f"{model_name}_{model_version}"
        performance_data['prediction_errors'][key] = errors
        
        # Alert on high error rate
        if key in performance_data['prediction_counts']:
            total_requests = performance_data['prediction_counts'][key]
            if total_requests > 0:
                error_rate = errors / total_requests
                if error_rate > 0.05:  # 5% error rate
                    create_alert(
                        'high_error_rate',
                        f"Model {model_name} (version {model_version}) has high error rate: {error_rate:.2%}"
                    )
    
    for item in gateway_requests:
        strategy = item.get('metric', {}).get('strategy', 'unknown')
        destination = item.get('metric', {}).get('destination', 'unknown')
        count = float(item.get('value', [0, 0])[1])
        key = f"{strategy}_{destination}"
        performance_data['gateway_requests'][key] = count
    
    for item in ground_truth:
        model_name = item.get('metric', {}).get('model_name', 'unknown')
        model_version = item.get('metric', {}).get('model_version', 'unknown')
        metric_name = item.get('metric', {}).get('metric_name', 'unknown')
        value = float(item.get('value', [0, 0])[1])
        
        if 'ground_truth' not in performance_data:
            performance_data['ground_truth'] = {}
            
        key = f"{model_name}_{model_version}"
        if key not in performance_data['ground_truth']:
            performance_data['ground_truth'][key] = {}
            
        performance_data['ground_truth'][key][metric_name] = value
    
    # Update global performance metrics
    performance_metrics.update(performance_data)
    return performance_data

def create_alert(alert_type, message):
    """Create a new alert and send notification if configured."""
    alert = {
        'type': alert_type,
        'message': message,
        'timestamp': datetime.now().isoformat()
    }
    
    # Add to alerts list
    alerts.append(alert)
    
    # Keep only the most recent 100 alerts
    if len(alerts) > 100:
        alerts.pop(0)
    
    # Send notification
    if ALERT_WEBHOOK_URL:
        try:
            requests.post(
                ALERT_WEBHOOK_URL,
                json={
                    'text': f"ALERT: {message}",
                    'alert': alert
                },
                timeout=5
            )
        except Exception as e:
            logger.error(f"Failed to send alert notification: {str(e)}")
    
    logger.warning(f"Alert: {message}")
    return alert

def monitoring_loop():
    """Main monitoring loop that runs periodically."""
    while True:
        try:
            logger.info("Running monitoring checks...")
            
            # Check system health
            check_system_health()
            
            # Collect performance metrics
            collect_performance_metrics()
            
            # Check for model drift
            check_model_drift()
            
            logger.info("Monitoring checks completed")
            
        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}")
        
        # Sleep until next check
        time.sleep(MONITORING_INTERVAL)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'monitoring_interval': MONITORING_INTERVAL,
        'last_check': health_status.get('last_update', 'never'),
        'services_monitored': list(health_status.keys())
    })

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Return collected metrics."""
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'performance': performance_metrics,
        'drift': drift_metrics,
        'health': health_status,
        'alerts': alerts
    })

@app.route('/alerts', methods=['GET'])
def get_alerts():
    """Return current alerts."""
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'alerts': alerts,
        'count': len(alerts)
    })

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Render monitoring dashboard."""
    # Generate performance charts
    performance_charts = []
    
    if 'prediction_counts' in performance_metrics and performance_metrics['prediction_counts']:
        chart = {
            'title': 'Prediction Request Counts',
            'data': generate_chart(
                performance_metrics['prediction_counts'], 
                'Prediction Counts by Model', 
                chart_type='bar'
            )
        }
        performance_charts.append(chart)
    
    if 'prediction_latency' in performance_metrics and performance_metrics['prediction_latency']:
        chart = {
            'title': 'Prediction Latency (P95, seconds)',
            'data': generate_chart(
                performance_metrics['prediction_latency'], 
                'P95 Latency by Model', 
                chart_type='bar'
            )
        }
        performance_charts.append(chart)
    
    if 'prediction_errors' in performance_metrics and performance_metrics['prediction_errors']:
        chart = {
            'title': 'Prediction Errors',
            'data': generate_chart(
                performance_metrics['prediction_errors'], 
                'Error Count by Model', 
                chart_type='bar'
            )
        }
        performance_charts.append(chart)
    
    # Generate drift charts
    drift_charts = []
    
    for model_name, metrics in drift_metrics.items():
        if metrics:
            chart = {
                'title': f'Model Drift: {model_name}',
                'data': generate_chart(
                    metrics, 
                    f'Metric Drift for {model_name}', 
                    chart_type='bar',
                    y_label='Drift Percentage'
                )
            }
            drift_charts.append(chart)
    
    # Generate traffic chart
    traffic_chart = None
    if 'gateway_requests' in performance_metrics and performance_metrics['gateway_requests']:
        traffic_chart = generate_chart(
            performance_metrics['gateway_requests'], 
            'Traffic Distribution by Deployment Strategy', 
            chart_type='pie'
        )
    
    # Generate latency chart
    latency_chart = None
    if 'prediction_latency' in performance_metrics and performance_metrics['prediction_latency']:
        latency_chart = generate_chart(
            performance_metrics['prediction_latency'], 
            'Prediction Latency by Model', 
            chart_type='bar',
            y_label='Seconds (P95)'
        )
    
    # Render template
    return render_template(
        'dashboard.html',
        last_updated=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        health_status=health_status,
        alerts=alerts,
        performance_charts=performance_charts,
        drift_charts=drift_charts,
        traffic_chart=traffic_chart,
        latency_chart=latency_chart
    )

@app.route('/', methods=['GET'])
def index():
    """Redirect to dashboard."""
    return dashboard()

if __name__ == '__main__':
    # Start monitoring thread
    monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
    monitoring_thread.start()
    
    # Log startup info
    logger.info(f"Starting monitoring service with interval: {MONITORING_INTERVAL} seconds")
    logger.info(f"Model registry path: {MODEL_REGISTRY_PATH}")
    logger.info(f"Production models path: {PRODUCTION_MODELS_PATH}")
    logger.info(f"Prometheus URL: {PROMETHEUS_URL}")
    logger.info(f"Prediction API: {PREDICTION_API}")
    logger.info(f"Gateway API: {GATEWAY_API}")
    
    # Start the Flask app
    app.run(debug=False, host='0.0.0.0', port=8000) 