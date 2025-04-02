"""
Automated Response System Module
Handles automated responses to monitoring issues and provides recommendations
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ActionType(Enum):
    RETRAIN = "retrain"
    CLEAN_DATA = "clean_data"
    ALERT_ESCALATION = "alert_escalation"
    ADJUST_THRESHOLDS = "adjust_thresholds"
    FEATURE_INVESTIGATION = "feature_investigation"

@dataclass
class RecommendedAction:
    action_type: ActionType
    priority: int  # 1-5, with 1 being highest priority
    description: str
    suggested_params: Dict[str, any]
    timestamp: datetime
    issue_source: str
    confidence: float  # 0-1 confidence in the recommendation

class AutoResponseManager:
    """Manages automated responses to monitoring issues"""
    
    def __init__(
        self,
        metrics_repository,
        alert_manager,
        health_monitor,
        data_quality_monitor,
        config: Optional[Dict] = None
    ):
        self.metrics_repository = metrics_repository
        self.alert_manager = alert_manager
        self.health_monitor = health_monitor
        self.data_quality_monitor = data_quality_monitor
        self.config = config or self._get_default_config()
        self.action_history: List[RecommendedAction] = []
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for automated responses"""
        return {
            "drift_retrain_threshold": 0.7,  # KS statistic threshold for retraining
            "quality_threshold": 0.8,  # Minimum acceptable quality score
            "error_rate_threshold": 0.05,  # Maximum acceptable error rate
            "latency_threshold_ms": 100,  # Maximum acceptable latency
            "min_confidence_threshold": 0.8,  # Minimum confidence for automated actions
            "max_daily_retrains": 2,  # Maximum number of retraining triggers per day
            "escalation_timeout_minutes": 30  # Time before escalating unresolved issues
        }
    
    def analyze_and_respond(
        self,
        model_id: str,
        current_metrics: Dict[str, float],
        quality_metrics: Dict[str, float],
        health_metrics: Dict[str, float]
    ) -> List[RecommendedAction]:
        """Analyze monitoring data and generate recommended actions"""
        recommended_actions = []
        
        # Check for data quality issues
        quality_actions = self._check_data_quality(quality_metrics)
        recommended_actions.extend(quality_actions)
        
        # Check for performance degradation
        performance_actions = self._check_performance(current_metrics)
        recommended_actions.extend(performance_actions)
        
        # Check for health issues
        health_actions = self._check_health(health_metrics)
        recommended_actions.extend(health_actions)
        
        # Sort actions by priority
        recommended_actions.sort(key=lambda x: x.priority)
        
        # Store actions in history
        self.action_history.extend(recommended_actions)
        
        return recommended_actions
    
    def _check_data_quality(self, quality_metrics: Dict[str, float]) -> List[RecommendedAction]:
        """Check data quality metrics and recommend actions"""
        actions = []
        
        # Check overall quality score
        quality_score = quality_metrics.get("overall_score", 1.0)
        if quality_score < self.config["quality_threshold"]:
            actions.append(
                RecommendedAction(
                    action_type=ActionType.CLEAN_DATA,
                    priority=2,
                    description="Data quality below threshold - cleanup recommended",
                    suggested_params={
                        "target_score": self.config["quality_threshold"],
                        "current_score": quality_score
                    },
                    timestamp=datetime.now(),
                    issue_source="data_quality",
                    confidence=0.9
                )
            )
        
        # Check for high missing rates
        missing_rate = quality_metrics.get("missing_rate", 0.0)
        if missing_rate > 0.1:  # More than 10% missing values
            actions.append(
                RecommendedAction(
                    action_type=ActionType.FEATURE_INVESTIGATION,
                    priority=3,
                    description="High missing value rate detected",
                    suggested_params={
                        "missing_rate": missing_rate,
                        "affected_features": quality_metrics.get("missing_features", [])
                    },
                    timestamp=datetime.now(),
                    issue_source="data_quality",
                    confidence=0.85
                )
            )
        
        return actions
    
    def _check_performance(self, current_metrics: Dict[str, float]) -> List[RecommendedAction]:
        """Check performance metrics and recommend actions"""
        actions = []
        
        # Check for significant drift
        drift_magnitude = current_metrics.get("drift_magnitude", 0.0)
        if drift_magnitude > self.config["drift_retrain_threshold"]:
            actions.append(
                RecommendedAction(
                    action_type=ActionType.RETRAIN,
                    priority=1,
                    description="Significant drift detected - model retraining recommended",
                    suggested_params={
                        "drift_magnitude": drift_magnitude,
                        "affected_features": current_metrics.get("drifted_features", [])
                    },
                    timestamp=datetime.now(),
                    issue_source="performance",
                    confidence=0.95
                )
            )
        
        # Check for threshold adjustments
        if current_metrics.get("false_positives", 0) > current_metrics.get("false_negatives", 0) * 2:
            actions.append(
                RecommendedAction(
                    action_type=ActionType.ADJUST_THRESHOLDS,
                    priority=2,
                    description="High false positive rate - threshold adjustment recommended",
                    suggested_params={
                        "current_threshold": current_metrics.get("threshold", 0.5),
                        "suggested_threshold": current_metrics.get("optimal_threshold", 0.6)
                    },
                    timestamp=datetime.now(),
                    issue_source="performance",
                    confidence=0.85
                )
            )
        
        return actions
    
    def _check_health(self, health_metrics: Dict[str, float]) -> List[RecommendedAction]:
        """Check health metrics and recommend actions"""
        actions = []
        
        # Check error rate
        error_rate = health_metrics.get("error_rate", 0.0)
        if error_rate > self.config["error_rate_threshold"]:
            actions.append(
                RecommendedAction(
                    action_type=ActionType.ALERT_ESCALATION,
                    priority=1,
                    description="High error rate detected - immediate attention required",
                    suggested_params={
                        "error_rate": error_rate,
                        "threshold": self.config["error_rate_threshold"]
                    },
                    timestamp=datetime.now(),
                    issue_source="health",
                    confidence=0.98
                )
            )
        
        # Check latency
        latency = health_metrics.get("latency_ms", 0.0)
        if latency > self.config["latency_threshold_ms"]:
            actions.append(
                RecommendedAction(
                    action_type=ActionType.ALERT_ESCALATION,
                    priority=2,
                    description="High latency detected - performance optimization recommended",
                    suggested_params={
                        "current_latency": latency,
                        "threshold": self.config["latency_threshold_ms"]
                    },
                    timestamp=datetime.now(),
                    issue_source="health",
                    confidence=0.9
                )
            )
        
        return actions
    
    def get_action_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        action_types: Optional[List[ActionType]] = None
    ) -> List[RecommendedAction]:
        """Get history of recommended actions with optional filters"""
        filtered_actions = self.action_history
        
        if start_time:
            filtered_actions = [a for a in filtered_actions if a.timestamp >= start_time]
        if end_time:
            filtered_actions = [a for a in filtered_actions if a.timestamp <= end_time]
        if action_types:
            filtered_actions = [a for a in filtered_actions if a.action_type in action_types]
        
        return filtered_actions
    
    def execute_action(self, action: RecommendedAction) -> bool:
        """Execute a recommended action if confidence meets threshold"""
        if action.confidence < self.config["min_confidence_threshold"]:
            logger.warning(f"Action confidence {action.confidence} below threshold - skipping execution")
            return False
        
        try:
            if action.action_type == ActionType.RETRAIN:
                # Implement model retraining logic
                logger.info("Triggering model retraining")
                # Add retraining implementation
                
            elif action.action_type == ActionType.CLEAN_DATA:
                # Implement data cleaning logic
                logger.info("Triggering data cleaning")
                # Add data cleaning implementation
                
            elif action.action_type == ActionType.ALERT_ESCALATION:
                # Implement alert escalation logic
                logger.info("Escalating alert")
                self.alert_manager.escalate_alert(action.suggested_params)
                
            elif action.action_type == ActionType.ADJUST_THRESHOLDS:
                # Implement threshold adjustment logic
                logger.info("Adjusting thresholds")
                # Add threshold adjustment implementation
                
            return True
            
        except Exception as e:
            logger.error(f"Error executing action: {str(e)}")
            return False 