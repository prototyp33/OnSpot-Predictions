from dotenv import load_dotenv
load_dotenv()
"""
Database integration helper script for the monitoring system.
This script provides functions to store monitoring results in the database.
"""

import os
import sys
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import psycopg2
from psycopg2.extras import Json, RealDictCursor
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


class DatabaseClient:
    """Client for interacting with the database."""

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize the database client.
        
        Args:
            connection_string: Database connection string. If None, will try to use environment variable.
        """
        self.connection_string = connection_string or os.getenv("DATABASE_URL")
        if not self.connection_string:
            logger.warning("No database connection string provided. Using dummy client.")
            self.is_dummy = True
        else:
            self.is_dummy = False
        
        self.conn = None
        
    def connect(self):
        """Establish a database connection."""
        if self.is_dummy:
            logger.info("Using dummy database client - no actual connection established")
            return
            
        try:
            self.conn = psycopg2.connect(self.connection_string)
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self.is_dummy = True
            
    def disconnect(self):
        """Close the database connection."""
        if self.conn and not self.is_dummy:
            self.conn.close()
            logger.info("Database connection closed")
            
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        
    def store_drift_analysis(
        self, 
        model_id: str, 
        drift_metrics: Dict[str, Dict[str, float]],
        baseline_timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Store drift analysis results in the database.
        
        Args:
            model_id: The ID of the model being monitored
            drift_metrics: Dictionary of drift metrics per feature
            baseline_timestamp: When the baseline data was collected
            
        Returns:
            bool: Success status
        """
        if self.is_dummy:
            logger.info(f"[DUMMY] Storing drift analysis for model {model_id}")
            logger.info(f"[DUMMY] Drift metrics: {json.dumps(drift_metrics, default=str)}")
            return True
            
        try:
            with self.conn.cursor() as cur:
                for feature_name, metrics in drift_metrics.items():
                    # Get required values with defaults
                    drift_score = max(
                        metrics.get('ks_statistic', 0),
                        metrics.get('distribution_difference', 0) / 100
                    )
                    
                    # Create query with parameters
                    query = """
                    INSERT INTO drift_analysis (
                        id, model_id, feature_name, drift_score, p_value, 
                        mean_difference, std_difference, distribution_difference,
                        new_categories, missing_categories, 
                        timestamp, baseline_timestamp, metadata
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """
                    
                    # Clean up values that may not exist
                    p_value = metrics.get('p_value')
                    mean_diff = metrics.get('mean_difference')
                    std_diff = metrics.get('std_difference')
                    dist_diff = metrics.get('distribution_difference')
                    
                    # Handle JSON fields
                    new_cats = Json(metrics.get('new_categories', [])) if 'new_categories' in metrics else None
                    missing_cats = Json(metrics.get('missing_categories', [])) if 'missing_categories' in metrics else None
                    metadata = Json(metrics)
                    
                    # Generate UUID
                    id = uuid.uuid4()
                    
                    # Current timestamp
                    now = datetime.now()
                    
                    # Execute query
                    cur.execute(
                        query, 
                        (
                            id, model_id, feature_name, drift_score, p_value,
                            mean_diff, std_diff, dist_diff, 
                            new_cats, missing_cats,
                            now, baseline_timestamp, metadata
                        )
                    )
                
                self.conn.commit()
                logger.info(f"Stored drift analysis for model {model_id} with {len(drift_metrics)} features")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store drift analysis: {e}")
            if self.conn:
                self.conn.rollback()
            return False
            
    def store_retraining_event(
        self,
        model_id: str,
        reason: str,
        success: bool = False,
        metrics_before: Optional[Dict[str, Any]] = None,
        metrics_after: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store model retraining event in the database.
        
        Args:
            model_id: The ID of the model being retrained
            reason: Reason for retraining
            success: Whether retraining was successful
            metrics_before: Performance metrics before retraining
            metrics_after: Performance metrics after retraining
            
        Returns:
            bool: Success status
        """
        if self.is_dummy:
            logger.info(f"[DUMMY] Storing retraining event for model {model_id}")
            logger.info(f"[DUMMY] Reason: {reason}, Success: {success}")
            return True
            
        try:
            with self.conn.cursor() as cur:
                query = """
                INSERT INTO retraining_events (
                    id, model_id, timestamp, reason, success, metrics_before, metrics_after
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s
                )
                """
                
                # Generate UUID
                id = uuid.uuid4()
                
                # Current timestamp
                now = datetime.now()
                
                # Handle JSON fields
                metrics_before_json = Json(metrics_before) if metrics_before else None
                metrics_after_json = Json(metrics_after) if metrics_after else None
                
                # Execute query
                cur.execute(
                    query, 
                    (
                        id, model_id, now, reason, success,
                        metrics_before_json, metrics_after_json
                    )
                )
                
                self.conn.commit()
                logger.info(f"Stored retraining event for model {model_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store retraining event: {e}")
            if self.conn:
                self.conn.rollback()
            return False
            
    def store_business_metric(
        self,
        metric_name: str,
        metric_value: float,
        category: str,
        location_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store business metric in the database.
        
        Args:
            metric_name: Name of the metric
            metric_value: Value of the metric
            category: Metric category (revenue, costs, efficiency, etc.)
            location_id: Optional location ID
            metadata: Additional metadata
            
        Returns:
            bool: Success status
        """
        if self.is_dummy:
            logger.info(f"[DUMMY] Storing business metric: {metric_name}={metric_value}")
            return True
            
        try:
            with self.conn.cursor() as cur:
                query = """
                INSERT INTO business_metrics (
                    id, metric_name, metric_value, timestamp, category, location_id, metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s
                )
                """
                
                # Generate UUID
                id = uuid.uuid4()
                
                # Current timestamp
                now = datetime.now()
                
                # Handle JSON field
                metadata_json = Json(metadata) if metadata else None
                
                # Execute query
                cur.execute(
                    query, 
                    (
                        id, metric_name, metric_value, now, category,
                        location_id, metadata_json
                    )
                )
                
                self.conn.commit()
                logger.info(f"Stored business metric: {metric_name}={metric_value}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store business metric: {e}")
            if self.conn:
                self.conn.rollback()
            return False
            
    def store_location_metrics(
        self,
        location_id: str,
        date: datetime,
        occupancy_accuracy: Optional[float] = None,
        utilization_rate: Optional[float] = None,
        revenue: Optional[float] = None,
        opportunity_cost: Optional[float] = None
    ) -> bool:
        """
        Store location-specific metrics in the database.
        
        Args:
            location_id: The location identifier
            date: The date for these metrics
            occupancy_accuracy: Accuracy of occupancy predictions
            utilization_rate: Actual parking utilization rate
            revenue: Revenue generated from this location
            opportunity_cost: Lost revenue from incorrect predictions
            
        Returns:
            bool: Success status
        """
        if self.is_dummy:
            logger.info(f"[DUMMY] Storing location metrics for {location_id} on {date}")
            return True
            
        try:
            with self.conn.cursor() as cur:
                query = """
                INSERT INTO location_metrics (
                    id, location_id, date, occupancy_accuracy, utilization_rate, revenue, opportunity_cost
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s
                )
                """
                
                # Generate UUID
                id = uuid.uuid4()
                
                # Convert datetime to date if needed
                if isinstance(date, datetime):
                    date = date.date()
                
                # Execute query
                cur.execute(
                    query, 
                    (
                        id, location_id, date, occupancy_accuracy,
                        utilization_rate, revenue, opportunity_cost
                    )
                )
                
                self.conn.commit()
                logger.info(f"Stored location metrics for {location_id} on {date}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store location metrics: {e}")
            if self.conn:
                self.conn.rollback()
            return False
            
    def store_system_health(
        self,
        component: str,
        status: str,
        metrics: Optional[Dict[str, Any]] = None,
        alert_level: Optional[str] = None,
        message: Optional[str] = None
    ) -> bool:
        """
        Store system health information in the database.
        
        Args:
            component: System component name
            status: Component status (operational, degraded, down)
            metrics: Performance metrics
            alert_level: Alert level (info, warning, critical)
            message: Additional message
            
        Returns:
            bool: Success status
        """
        if self.is_dummy:
            logger.info(f"[DUMMY] Storing system health for {component}: {status}")
            return True
            
        try:
            with self.conn.cursor() as cur:
                query = """
                INSERT INTO system_health (
                    id, timestamp, component, status, metrics, alert_level, message
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s
                )
                """
                
                # Generate UUID
                id = uuid.uuid4()
                
                # Current timestamp
                now = datetime.now()
                
                # Handle JSON field
                metrics_json = Json(metrics) if metrics else None
                
                # Execute query
                cur.execute(
                    query, 
                    (
                        id, now, component, status,
                        metrics_json, alert_level, message
                    )
                )
                
                self.conn.commit()
                logger.info(f"Stored system health for {component}: {status}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store system health: {e}")
            if self.conn:
                self.conn.rollback()
            return False
            
    def get_drift_metrics(
        self,
        model_id: Optional[str] = None,
        feature_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve drift metrics from the database.
        
        Args:
            model_id: Optional filter by model ID
            feature_name: Optional filter by feature name
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of records to return
            
        Returns:
            List of drift metrics
        """
        if self.is_dummy:
            logger.info(f"[DUMMY] Getting drift metrics for model {model_id}")
            return []
            
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build query
                query = "SELECT * FROM drift_analysis WHERE 1=1"
                params = []
                
                if model_id:
                    query += " AND model_id = %s"
                    params.append(model_id)
                    
                if feature_name:
                    query += " AND feature_name = %s"
                    params.append(feature_name)
                    
                if start_time:
                    query += " AND timestamp >= %s"
                    params.append(start_time)
                    
                if end_time:
                    query += " AND timestamp <= %s"
                    params.append(end_time)
                
                query += " ORDER BY timestamp DESC LIMIT %s"
                params.append(limit)
                
                # Execute query
                cur.execute(query, params)
                results = cur.fetchall()
                
                logger.info(f"Retrieved {len(results)} drift metrics records")
                return results
                
        except Exception as e:
            logger.error(f"Failed to retrieve drift metrics: {e}")
            return []
            
    def get_retraining_events(
        self,
        model_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve retraining events from the database.
        
        Args:
            model_id: Optional filter by model ID
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of records to return
            
        Returns:
            List of retraining events
        """
        if self.is_dummy:
            logger.info(f"[DUMMY] Getting retraining events for model {model_id}")
            return []
            
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build query
                query = "SELECT * FROM retraining_events WHERE 1=1"
                params = []
                
                if model_id:
                    query += " AND model_id = %s"
                    params.append(model_id)
                    
                if start_time:
                    query += " AND timestamp >= %s"
                    params.append(start_time)
                    
                if end_time:
                    query += " AND timestamp <= %s"
                    params.append(end_time)
                
                query += " ORDER BY timestamp DESC LIMIT %s"
                params.append(limit)
                
                # Execute query
                cur.execute(query, params)
                results = cur.fetchall()
                
                logger.info(f"Retrieved {len(results)} retraining events")
                return results
                
        except Exception as e:
            logger.error(f"Failed to retrieve retraining events: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Example drift metrics
    example_drift_metrics = {
        "temperature": {
            "ks_statistic": 0.32,
            "p_value": 0.01,
            "mean_difference": 0.15,
            "std_difference": 0.08,
        },
        "precipitation": {
            "ks_statistic": 0.22,
            "p_value": 0.07,
            "mean_difference": 0.05,
            "std_difference": 0.02,
        },
        "location_id": {
            "distribution_difference": 25.0,
            "new_categories": ["loc_123", "loc_456"],
            "missing_categories": ["loc_789"]
        }
    }
    
    # Create database client and store example data
    with DatabaseClient() as db:
        # Store drift analysis
        db.store_drift_analysis(
            model_id="model_v1",
            drift_metrics=example_drift_metrics,
            baseline_timestamp=datetime.now()
        )
        
        # Store retraining event
        db.store_retraining_event(
            model_id="model_v1",
            reason="Data drift detected",
            success=True,
            metrics_before={"rmse": 0.85, "mae": 0.35},
            metrics_after={"rmse": 0.72, "mae": 0.30}
        )
        
        # Store business metric
        db.store_business_metric(
            metric_name="revenue_impact",
            metric_value=12500.0,
            category="revenue",
            location_id="downtown_1",
            metadata={"currency": "USD", "period": "monthly"}
        )
        
        # Store location metrics
        db.store_location_metrics(
            location_id="downtown_1",
            date=datetime.now(),
            occupancy_accuracy=0.89,
            utilization_rate=0.76,
            revenue=2500.0,
            opportunity_cost=350.0
        )
        
        # Store system health
        db.store_system_health(
            component="prediction_api",
            status="operational",
            metrics={"latency_ms": 120, "error_rate": 0.001, "request_count": 15000},
            alert_level="info",
            message="System operating normally"
        ) 