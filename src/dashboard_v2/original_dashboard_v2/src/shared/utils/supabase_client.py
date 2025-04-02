"""
Supabase client utility for the ML Monitoring Dashboard.
"""

from typing import Dict, Any, Optional, List
import logging
from supabase import Client
from ...core.config.supabase_config import get_supabase_client, TABLES

logger = logging.getLogger(__name__)

class SupabaseManager:
    """Manager class for Supabase operations."""
    
    def __init__(self):
        """Initialize the Supabase manager."""
        self._client = get_supabase_client()
    
    def get_metrics(self, model_id: str, time_range: str = "7d") -> List[Dict[str, Any]]:
        """
        Get metrics for a specific model.
        
        Args:
            model_id: ID of the model
            time_range: Time range to fetch metrics for (e.g., "7d" for 7 days)
            
        Returns:
            List of metrics
        """
        try:
            response = self._client.from_("model_metrics").select("*").eq("model_id", model_id).execute()
            return response.data if hasattr(response, "data") else []
        except Exception as e:
            logger.error(f"Failed to fetch metrics for model {model_id}: {e}")
            return []
    
    def get_drift_analysis(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Get drift analysis results for a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            List of drift analysis results
        """
        try:
            response = self._client.from_("drift_analysis").select("*").eq("model_id", model_id).execute()
            return response.data if hasattr(response, "data") else []
        except Exception as e:
            logger.error(f"Failed to fetch drift analysis for model {model_id}: {e}")
            return []
    
    def get_predictions(self, model_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get recent predictions for a model.
        
        Args:
            model_id: ID of the model
            limit: Maximum number of predictions to fetch
            
        Returns:
            List of predictions
        """
        try:
            response = self._client.from_("predictions").select("*").eq("model_id", model_id).limit(limit).execute()
            return response.data if hasattr(response, "data") else []
        except Exception as e:
            logger.error(f"Failed to fetch predictions for model {model_id}: {e}")
            return []
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model information or None if not found
        """
        try:
            response = self._client.from_("models").select("*").eq("model_id", model_id).limit(1).execute()
            if hasattr(response, "data") and response.data:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Failed to fetch model info for {model_id}: {e}")
            return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of all models.
        
        Returns:
            List of models
        """
        try:
            response = self._client.from_("models").select("*").execute()
            return response.data if hasattr(response, "data") else []
        except Exception as e:
            logger.error(f"Failed to fetch models list: {e}")
            return []
    
    def update_metrics(self, metrics: List[Dict[str, Any]]) -> bool:
        """
        Update or insert new metrics.
        
        Args:
            metrics: List of metric records to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self._client.from_("model_metrics").upsert(metrics).execute()
            return bool(response.data)
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
            return False

# Global instance
supabase = SupabaseManager() 