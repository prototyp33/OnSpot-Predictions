"""
FastAPI implementation for the monitoring API.
"""
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import os
import sys
import logging
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import from monitoring modules
from monitoring.drift_detection import detect_drift, get_drift_history, DriftSummary, DriftResult

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="OnSpot Monitoring API",
    description="API for monitoring the OnSpot Predictive Model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
METRICS_DIR = Path("monitoring/metrics")
ALERTS_DIR = Path("monitoring/alerts")

# Ensure directories exist
METRICS_DIR.mkdir(parents=True, exist_ok=True)
ALERTS_DIR.mkdir(parents=True, exist_ok=True)

# Pydantic models
class Metric(BaseModel):
    """Model performance metric."""
    timestamp: datetime = Field(default_factory=datetime.now)
    name: str
    value: float
    previous_value: Optional[float] = None
    change_percentage: Optional[float] = None
    is_improvement: Optional[bool] = None
    meta_data: Optional[Dict[str, Any]] = None

class MetricsResponse(BaseModel):
    """Response for metrics endpoint."""
    metrics: List[Metric]
    timestamp: datetime = Field(default_factory=datetime.now)

class Alert(BaseModel):
    """Alert for model issues."""
    id: int
    timestamp: datetime = Field(default_factory=datetime.now)
    title: str
    message: str
    severity: str  # "low", "medium", "high"
    type: str  # "performance", "drift", "system"
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    meta_data: Optional[Dict[str, Any]] = None

class AlertsResponse(BaseModel):
    """Response for alerts endpoint."""
    alerts: List[Alert]
    timestamp: datetime = Field(default_factory=datetime.now)

class DriftResponse(BaseModel):
    """Response for drift endpoint."""
    drift_checks: List[DriftResult]
    timestamp: datetime = Field(default_factory=datetime.now)

class TimeRangeParams:
    """Parameters for time range filtering."""
    def __init__(
        self,
        days: Optional[int] = Query(7, description="Number of days to include"),
        start_date: Optional[datetime] = Query(None, description="Start date (ISO format)"),
        end_date: Optional[datetime] = Query(None, description="End date (ISO format)")
    ):
        self.days = days
        self.start_date = start_date
        self.end_date = end_date or datetime.now()
        
        if start_date is None and days is not None:
            self.start_date = self.end_date - timedelta(days=days)

# Mock data for development
def get_mock_metrics() -> List[Metric]:
    """Get mock metrics for development."""
    now = datetime.now()
    return [
        Metric(
            timestamp=now,
            name="rmse",
            value=0.15,
            previous_value=0.16,
            change_percentage=-6.25,
            is_improvement=True
        ),
        Metric(
            timestamp=now,
            name="mae",
            value=0.12,
            previous_value=0.13,
            change_percentage=-7.69,
            is_improvement=True
        ),
        Metric(
            timestamp=now,
            name="r2",
            value=0.85,
            previous_value=0.83,
            change_percentage=2.41,
            is_improvement=True
        ),
        Metric(
            timestamp=now,
            name="inference_time",
            value=45.2,
            previous_value=48.7,
            change_percentage=-7.19,
            is_improvement=True
        )
    ]

def get_mock_alerts() -> List[Alert]:
    """Get mock alerts for development."""
    now = datetime.now()
    return [
        Alert(
            id=1,
            timestamp=now - timedelta(hours=3),
            title="High Error Rate Detected",
            message="RMSE has increased by 15% in the last 24 hours",
            severity="high",
            type="performance"
        ),
        Alert(
            id=2,
            timestamp=now - timedelta(hours=6),
            title="Drift Detected in Temperature Feature",
            message="Statistical difference detected in the temperature feature distribution",
            severity="medium",
            type="drift"
        ),
        Alert(
            id=3,
            timestamp=now - timedelta(hours=12),
            title="Inference Time Increased",
            message="Average inference time increased by 12% in the last hour",
            severity="low",
            type="system",
            acknowledged=True,
            acknowledged_at=now - timedelta(hours=10)
        )
    ]

def get_mock_drift_checks() -> List[DriftResult]:
    """Get mock drift checks for development."""
    now = datetime.now()
    return [
        DriftResult(
            feature="temperature",
            p_value=0.02,
            statistic=0.85,
            drift_detected=True,
            timestamp=now
        ),
        DriftResult(
            feature="humidity",
            p_value=0.12,
            statistic=0.65,
            drift_detected=False,
            timestamp=now
        ),
        DriftResult(
            feature="precipitation",
            p_value=0.28,
            statistic=0.42,
            drift_detected=False,
            timestamp=now
        )
    ]

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint."""
    return {
        "name": "OnSpot Monitoring API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now()
    }

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(time_range: TimeRangeParams = Depends()):
    """
    Get model performance metrics.
    
    Args:
        time_range: Time range parameters
        
    Returns:
        MetricsResponse: Model performance metrics
    """
    # In a real implementation, we would load metrics from storage
    # For now, return mock data
    metrics = get_mock_metrics()
    
    return {
        "metrics": metrics,
        "timestamp": datetime.now()
    }

@app.get("/drift", response_model=DriftResponse)
async def get_drift(
    time_range: TimeRangeParams = Depends(),
    feature: Optional[str] = Query(None, description="Filter by feature name")
):
    """
    Get data drift detection results.
    
    Args:
        time_range: Time range parameters
        feature: Filter by feature name
        
    Returns:
        DriftResponse: Data drift detection results
    """
    # In a real implementation, we would load drift results from storage
    # For now, return mock data
    drift_checks = get_mock_drift_checks()
    
    # Filter by feature if specified
    if feature is not None:
        drift_checks = [check for check in drift_checks if check.feature == feature]
    
    return {
        "drift_checks": drift_checks,
        "timestamp": datetime.now()
    }

@app.get("/alerts", response_model=AlertsResponse)
async def get_alerts(
    time_range: TimeRangeParams = Depends(),
    severity: Optional[str] = Query(None, description="Filter by severity (low, medium, high)"),
    type: Optional[str] = Query(None, description="Filter by type (performance, drift, system)"),
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledgement status")
):
    """
    Get alerts.
    
    Args:
        time_range: Time range parameters
        severity: Filter by severity
        type: Filter by type
        acknowledged: Filter by acknowledgement status
        
    Returns:
        AlertsResponse: Alerts
    """
    # In a real implementation, we would load alerts from storage
    # For now, return mock data
    alerts = get_mock_alerts()
    
    # Apply filters
    if severity is not None:
        alerts = [alert for alert in alerts if alert.severity == severity]
    
    if type is not None:
        alerts = [alert for alert in alerts if alert.type == type]
    
    if acknowledged is not None:
        alerts = [alert for alert in alerts if alert.acknowledged == acknowledged]
    
    return {
        "alerts": alerts,
        "timestamp": datetime.now()
    }

@app.post("/alerts/{alert_id}/acknowledge", response_model=Dict[str, Any])
async def acknowledge_alert(alert_id: int):
    """
    Acknowledge an alert.
    
    Args:
        alert_id: ID of the alert to acknowledge
        
    Returns:
        Dict[str, Any]: Acknowledgement status
    """
    # In a real implementation, we would update the alert in storage
    # For now, just return success
    return {
        "status": "success",
        "message": f"Alert {alert_id} acknowledged",
        "timestamp": datetime.now()
    }

@app.post("/run_drift_detection", response_model=Dict[str, Any])
async def run_drift_detection(
    background_tasks: BackgroundTasks,
    data_path: Optional[str] = Query(None, description="Path to current data"),
    reference_path: Optional[str] = Query(None, description="Path to reference data")
):
    """
    Run drift detection as a background task.
    
    Args:
        background_tasks: FastAPI background tasks
        data_path: Path to current data
        reference_path: Path to reference data
        
    Returns:
        Dict[str, Any]: Task status
    """
    # In a real implementation, we would run drift detection in the background
    # For now, just return success
    return {
        "status": "success",
        "message": "Drift detection started",
        "timestamp": datetime.now()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("monitoring.api:app", host="0.0.0.0", port=8001, reload=True) 