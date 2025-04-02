"""
Supabase service for data access.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from supabase import create_client, Client
from ..core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class SupabaseService:
    """Service for interacting with Supabase."""
    
    def __init__(self):
        """Initialize Supabase client."""
        self.client: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY
        )
    
    async def get_metrics(
        self,
        days: int = settings.DEFAULT_METRICS_WINDOW,
        model_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get model metrics for the specified time period.
        
        Args:
            days: Number of days of data to retrieve
            model_id: Optional model ID to filter by
            
        Returns:
            List of metric records
        """
        try:
            query = self.client.from_("model_metrics").select("*")
            
            if model_id:
                query = query.eq("model_id", model_id)
            
            # Add time window filter
            start_date = datetime.now() - timedelta(days=days)
            query = query.gte("timestamp", start_date.isoformat())
            
            # Execute query
            response = await query.order("timestamp", desc=True).execute()
            return response.data if hasattr(response, "data") else []
            
        except Exception as e:
            logger.error(f"Error fetching metrics: {e}")
            return []
    
    async def get_drift_analysis(
        self,
        days: int = settings.DEFAULT_METRICS_WINDOW,
        model_id: Optional[str] = None,
        feature: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get drift analysis results.
        
        Args:
            days: Number of days of data to retrieve
            model_id: Optional model ID to filter by
            feature: Optional feature name to filter by
            
        Returns:
            List of drift analysis records
        """
        try:
            query = self.client.from_("drift_analysis").select("*")
            
            if model_id:
                query = query.eq("model_id", model_id)
            
            if feature:
                query = query.eq("feature_name", feature)
            
            # Add time window filter
            start_date = datetime.now() - timedelta(days=days)
            query = query.gte("timestamp", start_date.isoformat())
            
            # Execute query
            response = await query.order("timestamp", desc=True).execute()
            return response.data if hasattr(response, "data") else []
            
        except Exception as e:
            logger.error(f"Error fetching drift analysis: {e}")
            return []
    
    async def get_alerts(
        self,
        severity: Optional[str] = None,
        alert_type: Optional[str] = None,
        acknowledged: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Get system alerts.
        
        Args:
            severity: Optional severity level to filter by
            alert_type: Optional alert type to filter by
            acknowledged: Optional acknowledgment status to filter by
            
        Returns:
            List of alert records
        """
        try:
            query = self.client.from_("alerts").select("*")
            
            if severity:
                query = query.eq("severity", severity)
            
            if alert_type:
                query = query.eq("type", alert_type)
            
            if acknowledged is not None:
                query = query.eq("acknowledged", acknowledged)
            
            # Add retention period filter
            retention_date = datetime.now() - timedelta(days=settings.ALERT_RETENTION_DAYS)
            query = query.gte("timestamp", retention_date.isoformat())
            
            # Execute query
            response = await query.order("timestamp", desc=True).execute()
            return response.data if hasattr(response, "data") else []
            
        except Exception as e:
            logger.error(f"Error fetching alerts: {e}")
            return []
    
    async def acknowledge_alert(self, alert_id: int) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = await self.client.from_("alerts").update({
                "acknowledged": True,
                "acknowledged_at": datetime.now().isoformat()
            }).eq("id", alert_id).execute()
            
            return bool(response.data)
            
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False
    
    async def create_alert(
        self,
        title: str,
        message: str,
        severity: str,
        alert_type: str,
        meta_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new alert.
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity level
            alert_type: Type of alert
            meta_data: Optional additional data
            
        Returns:
            Created alert record or None if failed
        """
        try:
            alert_data = {
                "title": title,
                "message": message,
                "severity": severity,
                "type": alert_type,
                "timestamp": datetime.now().isoformat(),
                "acknowledged": False,
                "meta_data": meta_data or {}
            }
            
            response = await self.client.from_("alerts").insert(alert_data).execute()
            return response.data[0] if hasattr(response, "data") else None
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return None 