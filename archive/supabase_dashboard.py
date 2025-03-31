#!/usr/bin/env python
"""
Supabase Dashboard Module

This module provides a web-based dashboard for viewing Supabase
monitoring data, including performance metrics, health status,
and alerts.
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import threading
import webbrowser
from pathlib import Path

try:
    # Optional web dependencies - don't fail if not available
    import fastapi
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.templating import Jinja2Templates
    import uvicorn
    WEB_DEPS_AVAILABLE = True
except ImportError:
    WEB_DEPS_AVAILABLE = False
    logging.warning("Web dependencies not available. Dashboard will not be accessible. Install with: pip install fastapi uvicorn jinja2")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from scripts.supabase_monitor import get_monitor
    from scripts.supabase_metrics_extension import integrate_supabase_monitoring
except ImportError:
    logger.warning("Failed to import supabase_monitor or metrics_extension. Some functions may not work.")
    def get_monitor():
        return None
    def integrate_supabase_monitoring():
        return None

class SupabaseDashboard:
    """
    Web-based dashboard for Supabase monitoring.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8088):
        """
        Initialize the dashboard.
        
        Args:
            host: Host to bind the dashboard server to
            port: Port to bind the dashboard server to
        """
        self.host = host
        self.port = port
        self.running = False
        self.app = None
        self.server_thread = None
        self.monitor = get_monitor()
        self.metrics_extension = integrate_supabase_monitoring()
        
        # Create directories for dashboard assets
        self.dashboard_dir = Path(__file__).parent / "dashboard"
        self.static_dir = self.dashboard_dir / "static"
        self.templates_dir = self.dashboard_dir / "templates"
        
        self._ensure_directories()
        self._create_default_assets()
        
        logger.info("Supabase dashboard initialized")
    
    def _ensure_directories(self):
        """Ensure dashboard directories exist."""
        os.makedirs(self.dashboard_dir, exist_ok=True)
        os.makedirs(self.static_dir, exist_ok=True)
        os.makedirs(self.templates_dir, exist_ok=True)
    
    def _create_default_assets(self):
        """Create default dashboard assets if they don't exist."""
        # Create default index.html
        index_html_path = self.templates_dir / "index.html"
        if not index_html_path.exists():
            index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Supabase Monitoring Dashboard</title>
    <link rel="stylesheet" href="{{ url_for('static', path='/styles.css') }}">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <header>
        <h1>Supabase Monitoring Dashboard</h1>
        <div class="status-indicator {{ health_class }}">
            <span>Status: {{ health_status }}</span>
        </div>
    </header>
    
    <main>
        <section class="metrics-overview">
            <h2>Overview</h2>
            <div class="metrics-cards">
                <div class="metric-card">
                    <h3>Query Count</h3>
                    <p class="metric-value">{{ metrics.query_count }}</p>
                </div>
                <div class="metric-card">
                    <h3>Error Rate</h3>
                    <p class="metric-value {{ 'metric-danger' if metrics.error_rate > 0.05 else 'metric-success' }}">
                        {{ "{:.2%}".format(metrics.error_rate) }}
                    </p>
                </div>
                <div class="metric-card">
                    <h3>Avg Query Time</h3>
                    <p class="metric-value">
                        {{ "{:.2f}".format(metrics.performance.avg_query_time * 1000) }}ms
                    </p>
                </div>
                <div class="metric-card">
                    <h3>P95 Query Time</h3>
                    <p class="metric-value {{ 'metric-warning' if metrics.performance.p95_query_time > 0.5 else 'metric-success' }}">
                        {{ "{:.2f}".format(metrics.performance.p95_query_time * 1000) }}ms
                    </p>
                </div>
            </div>
        </section>
        
        <section class="performance-charts">
            <h2>Performance</h2>
            <div class="chart-container">
                <canvas id="queryTimeChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="operationCountChart"></canvas>
            </div>
        </section>
        
        <section class="recent-queries">
            <h2>Recent Queries</h2>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Operation</th>
                        <th>Duration</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for query in recent_queries %}
                    <tr class="{{ 'error-row' if query.error else '' }}">
                        <td>{{ query.timestamp | format_timestamp }}</td>
                        <td>{{ query.operation }}</td>
                        <td>{{ "{:.2f}".format(query.duration * 1000) }}ms</td>
                        <td>{{ "Error" if query.error else "Success" }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </section>
        
        <section class="alerts">
            <h2>Recent Alerts</h2>
            {% if alerts %}
            <ul class="alert-list">
                {% for alert in alerts %}
                <li class="alert-item {{ alert.level }}">
                    <span class="alert-time">{{ alert.timestamp | format_timestamp }}</span>
                    <span class="alert-level">{{ alert.level | title }}</span>
                    <span class="alert-message">{{ alert.issues | join(', ') }}</span>
                </li>
                {% endfor %}
            </ul>
            {% else %}
            <p>No alerts in the last 24 hours.</p>
            {% endif %}
        </section>
    </main>
    
    <footer>
        <p>OnSpot Predictive Model - Supabase Monitoring Dashboard</p>
        <p>Last updated: {{ now | format_timestamp }}</p>
    </footer>
    
    <script>
        // Fetch dashboard data every 30 seconds
        function refreshData() {
            fetch('/api/dashboard-data')
                .then(response => response.json())
                .then(data => {
                    updateCharts(data);
                    setTimeout(refreshData, 30000);
                })
                .catch(error => {
                    console.error('Error fetching dashboard data:', error);
                    setTimeout(refreshData, 60000);
                });
        }
        
        // Initialize and update charts
        function updateCharts(data) {
            // Query time chart
            const timeLabels = data.time_series.map(d => new Date(d.timestamp * 1000).toLocaleTimeString());
            const avgTimes = data.time_series.map(d => d.avg_time * 1000);
            const p95Times = data.time_series.map(d => d.p95_time * 1000);
            
            if (window.queryTimeChart) {
                window.queryTimeChart.data.labels = timeLabels;
                window.queryTimeChart.data.datasets[0].data = avgTimes;
                window.queryTimeChart.data.datasets[1].data = p95Times;
                window.queryTimeChart.update();
            } else {
                const ctxTime = document.getElementById('queryTimeChart').getContext('2d');
                window.queryTimeChart = new Chart(ctxTime, {
                    type: 'line',
                    data: {
                        labels: timeLabels,
                        datasets: [{
                            label: 'Avg Query Time (ms)',
                            data: avgTimes,
                            borderColor: 'rgba(75, 192, 192, 1)',
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            tension: 0.1
                        }, {
                            label: 'P95 Query Time (ms)',
                            data: p95Times,
                            borderColor: 'rgba(255, 159, 64, 1)',
                            backgroundColor: 'rgba(255, 159, 64, 0.2)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Query Performance'
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Time (ms)'
                                }
                            }
                        }
                    }
                });
            }
            
            // Operation count chart
            const operations = Object.keys(data.operation_counts);
            const counts = Object.values(data.operation_counts);
            
            if (window.operationChart) {
                window.operationChart.data.labels = operations;
                window.operationChart.data.datasets[0].data = counts;
                window.operationChart.update();
            } else {
                const ctxOps = document.getElementById('operationCountChart').getContext('2d');
                window.operationChart = new Chart(ctxOps, {
                    type: 'bar',
                    data: {
                        labels: operations,
                        datasets: [{
                            label: 'Operation Counts',
                            data: counts,
                            backgroundColor: [
                                'rgba(54, 162, 235, 0.5)',
                                'rgba(75, 192, 192, 0.5)',
                                'rgba(255, 206, 86, 0.5)',
                                'rgba(255, 99, 132, 0.5)',
                                'rgba(153, 102, 255, 0.5)',
                            ],
                            borderColor: [
                                'rgba(54, 162, 235, 1)',
                                'rgba(75, 192, 192, 1)',
                                'rgba(255, 206, 86, 1)',
                                'rgba(255, 99, 132, 1)',
                                'rgba(153, 102, 255, 1)',
                            ],
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Operation Distribution'
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Count'
                                }
                            }
                        }
                    }
                });
            }
        }
        
        // Start the refresh cycle when the page loads
        document.addEventListener('DOMContentLoaded', refreshData);
    </script>
</body>
</html>
"""
            with open(index_html_path, "w") as f:
                f.write(index_html)
        
        # Create default CSS
        css_path = self.static_dir / "styles.css"
        if not css_path.exists():
            css = """/* Dashboard Styles */
:root {
    --primary-color: #3ecf8e;
    --secondary-color: #1e293b;
    --text-color: #333;
    --background-color: #f8fafc;
    --card-bg-color: #fff;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --danger-color: #ef4444;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    color: var(--text-color);
    background-color: var(--background-color);
    line-height: 1.6;
}

header {
    background-color: var(--secondary-color);
    color: white;
    padding: 1rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

h1, h2, h3 {
    margin-bottom: 1rem;
}

main {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

section {
    margin-bottom: 3rem;
    background-color: var(--card-bg-color);
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    padding: 1.5rem;
}

.status-indicator {
    padding: 0.5rem 1rem;
    border-radius: 4px;
    font-weight: bold;
}

.healthy {
    background-color: var(--success-color);
}

.degraded {
    background-color: var(--warning-color);
}

.critical {
    background-color: var(--danger-color);
}

.metrics-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
}

.metric-card {
    padding: 1.5rem;
    border-radius: 8px;
    background-color: #f1f5f9;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    margin-top: 0.5rem;
}

.metric-success {
    color: var(--success-color);
}

.metric-warning {
    color: var(--warning-color);
}

.metric-danger {
    color: var(--danger-color);
}

.chart-container {
    height: 300px;
    margin-bottom: 2rem;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1rem;
}

thead {
    background-color: #f1f5f9;
}

th, td {
    padding: 0.75rem;
    text-align: left;
    border-bottom: 1px solid #e2e8f0;
}

.error-row {
    background-color: rgba(239, 68, 68, 0.1);
}

.alert-list {
    list-style: none;
}

.alert-item {
    padding: 1rem;
    margin-bottom: 0.5rem;
    border-radius: 4px;
    display: flex;
    align-items: center;
}

.alert-item.warning {
    background-color: rgba(245, 158, 11, 0.1);
    border-left: 4px solid var(--warning-color);
}

.alert-item.critical {
    background-color: rgba(239, 68, 68, 0.1);
    border-left: 4px solid var(--danger-color);
}

.alert-time {
    font-size: 0.875rem;
    color: #64748b;
    margin-right: 1rem;
    min-width: 150px;
}

.alert-level {
    font-weight: bold;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    margin-right: 1rem;
    text-transform: uppercase;
    font-size: 0.75rem;
}

.alert-item.warning .alert-level {
    background-color: var(--warning-color);
    color: white;
}

.alert-item.critical .alert-level {
    background-color: var(--danger-color);
    color: white;
}

footer {
    text-align: center;
    padding: 1rem;
    background-color: var(--secondary-color);
    color: white;
    margin-top: 2rem;
}

@media (max-width: 768px) {
    header {
        flex-direction: column;
        align-items: flex-start;
    }
    
    .status-indicator {
        margin-top: 1rem;
    }
    
    .metrics-cards {
        grid-template-columns: 1fr;
    }
}
"""
            with open(css_path, "w") as f:
                f.write(css)
    
    def start(self, open_browser: bool = False) -> bool:
        """
        Start the dashboard server.
        
        Args:
            open_browser: Whether to open a browser window
            
        Returns:
            True if server started successfully, False otherwise
        """
        if self.running:
            logger.warning("Dashboard server is already running")
            return True
        
        if not WEB_DEPS_AVAILABLE:
            logger.error("Cannot start dashboard: web dependencies not available")
            logger.error("Install with: pip install fastapi uvicorn jinja2")
            return False
        
        try:
            # Create FastAPI app
            app = FastAPI(title="Supabase Monitoring Dashboard")
            self.app = app
            
            # Set up templates
            templates = Jinja2Templates(directory=str(self.templates_dir))
            
            # Add template filters
            @templates.jinja2_env.filter
            def format_timestamp(timestamp):
                if isinstance(timestamp, (int, float)):
                    dt = datetime.fromtimestamp(timestamp)
                else:
                    dt = datetime.fromisoformat(timestamp)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # Serve static files
            app.mount("/static", StaticFiles(directory=str(self.static_dir)), name="static")
            
            # Define routes
            @app.get("/", response_class=HTMLResponse)
            async def index(request: Request):
                # Get monitor data
                metrics = self.monitor.get_metrics() if self.monitor else {
                    "query_count": 0,
                    "error_count": 0,
                    "error_rate": 0,
                    "operation_counts": {},
                    "performance": {
                        "avg_query_time": 0,
                        "p95_query_time": 0,
                        "p99_query_time": 0
                    }
                }
                
                # Calculate metrics
                metrics["error_rate"] = metrics["error_count"] / max(metrics["query_count"], 1)
                
                # Get recent queries
                recent_queries = self.monitor.get_recent_queries(minutes=15) if self.monitor else []
                
                # Get health status
                health = self.metrics_extension.get_latest_supabase_health() if self.metrics_extension else {
                    "status": "unknown",
                    "alert_level": "unknown",
                    "issues": []
                }
                
                # Get alerts
                alerts = self.metrics_extension.get_recent_alerts() if self.metrics_extension else []
                
                return templates.TemplateResponse("index.html", {
                    "request": request,
                    "metrics": metrics,
                    "recent_queries": recent_queries,
                    "health_status": health["status"],
                    "health_class": health["status"],
                    "alerts": alerts,
                    "now": time.time()
                })
            
            @app.get("/api/dashboard-data")
            async def dashboard_data():
                # Get current data
                metrics = self.monitor.get_metrics() if self.monitor else {
                    "query_count": 0, 
                    "operation_counts": {},
                    "performance": {"avg_query_time": 0, "p95_query_time": 0}
                }
                
                # Create time series data (mock - in real implementation would come from stored metrics)
                time_series = []
                now = time.time()
                for i in range(12):  # Last hour in 5-minute intervals
                    time_point = now - (i * 300)  # 5 minutes in seconds
                    time_series.append({
                        "timestamp": time_point,
                        "avg_time": metrics["performance"]["avg_query_time"],
                        "p95_time": metrics["performance"]["p95_query_time"]
                    })
                
                time_series.reverse()
                
                return {
                    "time_series": time_series,
                    "operation_counts": metrics["operation_counts"]
                }
            
            # Start server in a separate thread
            def run_server():
                uvicorn.run(app, host=self.host, port=self.port)
            
            self.server_thread = threading.Thread(target=run_server)
            self.server_thread.daemon = True
            self.server_thread.start()
            self.running = True
            
            logger.info(f"Dashboard server started at http://{self.host}:{self.port}")
            
            if open_browser:
                webbrowser.open(f"http://{self.host}:{self.port}")
            
            return True
        except Exception as e:
            logger.error(f"Error starting dashboard server: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the dashboard server.
        
        Returns:
            True if server stopped successfully, False otherwise
        """
        if not self.running:
            logger.warning("Dashboard server is not running")
            return True
        
        try:
            # In a real implementation, would need to properly shut down uvicorn
            self.running = False
            logger.info("Dashboard server stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping dashboard server: {e}")
            return False


if __name__ == "__main__":
    # Example usage
    dashboard = SupabaseDashboard()
    dashboard.start(open_browser=True)
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        dashboard.stop()
        print("Dashboard stopped.") 