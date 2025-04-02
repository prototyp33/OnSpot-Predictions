"""
Pydantic models for monitoring data structures.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field

class MetricBase(BaseModel):
    """Base model for metrics."""
    name: str = Field(..., description="Name of the metric")
    value: float = Field(..., description="Current value of the metric")
    timestamp: datetime = Field(..., description="Timestamp of the metric")
    model_id: Optional[str] = Field(None, description="ID of the model this metric belongs to")
    meta_data: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class MetricCreate(MetricBase):
    """Model for creating a new metric."""
    pass

class Metric(MetricBase):
    """Model for a metric with additional computed fields."""
    id: int = Field(..., description="Unique identifier")
    previous_value: Optional[float] = Field(None, description="Previous value of the metric")
    change_percentage: Optional[float] = Field(None, description="Percentage change from previous value")
    improved: Optional[bool] = Field(None, description="Whether the metric improved")

    class Config:
        orm_mode = True

class DriftAnalysisBase(BaseModel):
    """Base model for drift analysis."""
    feature_name: str = Field(..., description="Name of the feature analyzed")
    p_value: float = Field(..., description="P-value from statistical test")
    drift_detected: bool = Field(..., description="Whether drift was detected")
    timestamp: datetime = Field(..., description="Timestamp of the analysis")
    model_id: Optional[str] = Field(None, description="ID of the model this analysis belongs to")
    meta_data: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class DriftAnalysisCreate(DriftAnalysisBase):
    """Model for creating a new drift analysis."""
    pass

class DriftAnalysis(DriftAnalysisBase):
    """Model for a drift analysis with ID."""
    id: int = Field(..., description="Unique identifier")

    class Config:
        orm_mode = True

class AlertBase(BaseModel):
    """Base model for alerts."""
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    severity: str = Field(..., description="Alert severity level")
    type: str = Field(..., description="Type of alert")
    meta_data: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class AlertCreate(AlertBase):
    """Model for creating a new alert."""
    pass

class Alert(AlertBase):
    """Model for an alert with additional fields."""
    id: int = Field(..., description="Unique identifier")
    timestamp: datetime = Field(..., description="Timestamp when the alert was created")
    acknowledged: bool = Field(False, description="Whether the alert has been acknowledged")
    acknowledged_at: Optional[datetime] = Field(None, description="Timestamp when the alert was acknowledged")

    class Config:
        orm_mode = True

class MetricsResponse(BaseModel):
    """Response model for metrics endpoints."""
    metrics: List[Metric] = Field(..., description="List of metrics")

class DriftAnalysisResponse(BaseModel):
    """Response model for drift analysis endpoints."""
    drift_checks: List[DriftAnalysis] = Field(..., description="List of drift analysis results")

class AlertsResponse(BaseModel):
    """Response model for alerts endpoints."""
    alerts: List[Alert] = Field(..., description="List of alerts") 