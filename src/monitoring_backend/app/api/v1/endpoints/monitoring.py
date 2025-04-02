"""
API endpoints for model monitoring.
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from ....services.supabase_service import SupabaseService
from ....schemas.monitoring import (
    MetricsResponse,
    DriftAnalysisResponse,
    AlertsResponse,
    AlertCreate,
    Alert
)

router = APIRouter()

async def get_supabase_service() -> SupabaseService:
    """Dependency for getting Supabase service instance."""
    return SupabaseService()

@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    days: int = Query(7, description="Number of days of data to retrieve"),
    model_id: Optional[str] = Query(None, description="Optional model ID to filter by"),
    service: SupabaseService = Depends(get_supabase_service)
) -> MetricsResponse:
    """
    Get model performance metrics.
    
    Args:
        days: Number of days of data to retrieve
        model_id: Optional model ID to filter by
        service: Supabase service instance
        
    Returns:
        List of metrics
    """
    metrics = await service.get_metrics(days=days, model_id=model_id)
    return MetricsResponse(metrics=metrics)

@router.get("/drift", response_model=DriftAnalysisResponse)
async def get_drift_analysis(
    days: int = Query(7, description="Number of days of data to retrieve"),
    model_id: Optional[str] = Query(None, description="Optional model ID to filter by"),
    feature: Optional[str] = Query(None, description="Optional feature name to filter by"),
    service: SupabaseService = Depends(get_supabase_service)
) -> DriftAnalysisResponse:
    """
    Get drift analysis results.
    
    Args:
        days: Number of days of data to retrieve
        model_id: Optional model ID to filter by
        feature: Optional feature name to filter by
        service: Supabase service instance
        
    Returns:
        List of drift analysis results
    """
    drift_checks = await service.get_drift_analysis(
        days=days,
        model_id=model_id,
        feature=feature
    )
    return DriftAnalysisResponse(drift_checks=drift_checks)

@router.get("/alerts", response_model=AlertsResponse)
async def get_alerts(
    severity: Optional[str] = Query(None, description="Optional severity level to filter by"),
    alert_type: Optional[str] = Query(None, description="Optional alert type to filter by"),
    acknowledged: Optional[bool] = Query(None, description="Optional acknowledgment status to filter by"),
    service: SupabaseService = Depends(get_supabase_service)
) -> AlertsResponse:
    """
    Get system alerts.
    
    Args:
        severity: Optional severity level to filter by
        alert_type: Optional alert type to filter by
        acknowledged: Optional acknowledgment status to filter by
        service: Supabase service instance
        
    Returns:
        List of alerts
    """
    alerts = await service.get_alerts(
        severity=severity,
        alert_type=alert_type,
        acknowledged=acknowledged
    )
    return AlertsResponse(alerts=alerts)

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: int,
    service: SupabaseService = Depends(get_supabase_service)
) -> dict:
    """
    Acknowledge an alert.
    
    Args:
        alert_id: ID of the alert to acknowledge
        service: Supabase service instance
        
    Returns:
        Success status
    """
    success = await service.acknowledge_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found or already acknowledged")
    return {"status": "success", "message": "Alert acknowledged successfully"}

@router.post("/alerts", response_model=Alert)
async def create_alert(
    alert: AlertCreate,
    service: SupabaseService = Depends(get_supabase_service)
) -> Alert:
    """
    Create a new alert.
    
    Args:
        alert: Alert data
        service: Supabase service instance
        
    Returns:
        Created alert
    """
    created_alert = await service.create_alert(
        title=alert.title,
        message=alert.message,
        severity=alert.severity,
        alert_type=alert.type,
        meta_data=alert.meta_data
    )
    
    if not created_alert:
        raise HTTPException(status_code=500, detail="Failed to create alert")
    
    return Alert(**created_alert) 