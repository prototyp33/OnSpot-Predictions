#!/usr/bin/env python
"""
Dynamic Scheduler for ML Model Retraining

This script implements dynamic scheduling for model retraining based on:
1. Data drift metrics
2. Performance degradation trends
3. Resource utilization

The dynamic scheduler adjusts retraining intervals based on real-time monitoring data.
"""

import os
import sys
import argparse
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   filename='logs/dynamic_scheduler.log')

# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logger = logging.getLogger('DynamicScheduler')

class DynamicScheduler:
    """
    Dynamic scheduler that adjusts retraining intervals based on monitoring metrics.
    """
    
    def __init__(self, config_path: str = "config/monitoring_config.json"):
        """Initialize the dynamic scheduler with configuration."""
        self.config = self._load_config(config_path)
        self.schedule_history = self._load_schedule_history()
        self.metrics_history = self._load_metrics_history()
        self.drift_history = self._load_drift_history()

        # Default retraining interval in days
        self.base_interval = self.config.get("dynamic_scheduling", {}).get("base_interval_days", 30)
        
        # Adjustment limits
        self.min_interval = self.config.get("dynamic_scheduling", {}).get("adjustment_limits", {}).get("min_interval_days", 7)
        self.max_interval = self.config.get("dynamic_scheduling", {}).get("adjustment_limits", {}).get("max_interval_days", 60)
        
        # Create necessary directories
        Path("data/monitoring").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
                # Ensure dynamic scheduling section exists
                if "dynamic_scheduling" not in config:
                    config["dynamic_scheduling"] = {
                        "enabled": True,
                        "base_interval_days": 30,
                        "adjustment_limits": {
                            "min_interval_days": 7,
                            "max_interval_days": 60
                        },
                        "drift_thresholds": {
                            "low": 0.1,
                            "medium": 0.3,
                            "high": 0.5
                        },
                        "performance_thresholds": {
                            "rmse_degradation": 0.15,
                            "accuracy_drop": 0.05
                        }
                    }
                    
                    # Save the updated config
                    with open(config_path, 'w') as f_out:
                        json.dump(config, f_out, indent=4)
                
                return config
                
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            
            # Create a default config
            default_config = {
                "dynamic_scheduling": {
                    "enabled": True,
                    "base_interval_days": 30,
                    "adjustment_limits": {
                        "min_interval_days": 7,
                        "max_interval_days": 60
                    },
                    "drift_thresholds": {
                        "low": 0.1,
                        "medium": 0.3,
                        "high": 0.5
                    },
                    "performance_thresholds": {
                        "rmse_degradation": 0.15,
                        "accuracy_drop": 0.05
                    }
                }
            }
            
            # Create the directory if it doesn't exist
            Path(os.path.dirname(config_path)).mkdir(parents=True, exist_ok=True)
            
            # Save the default config
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            
            logger.info(f"Created default configuration at {config_path}")
            return default_config
    
    def _load_schedule_history(self) -> List[Dict]:
        """Load retraining schedule history from file."""
        history_path = "data/monitoring/schedule_history.jsonl"
        history = []
        
        try:
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    for line in f:
                        history.append(json.loads(line))
        except Exception as e:
            logger.error(f"Error loading schedule history: {e}")
        
        return history
    
    def _load_metrics_history(self) -> List[Dict]:
        """Load performance metrics history from file."""
        history_path = "data/monitoring/performance_history.json"
        
        try:
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    return json.load(f)
            else:
                return []
        except Exception as e:
            logger.error(f"Error loading metrics history: {e}")
            return []
    
    def _load_drift_history(self) -> List[Dict]:
        """Load drift metrics history from file."""
        history_path = "data/monitoring/drift_history.jsonl"
        history = []
        
        try:
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    for line in f:
                        history.append(json.loads(line))
        except Exception as e:
            logger.error(f"Error loading drift history: {e}")
        
        return history
    
    def _save_schedule_history(self):
        """Save retraining schedule history to file."""
        history_path = "data/monitoring/schedule_history.jsonl"
        
        try:
            with open(history_path, 'w') as f:
                for entry in self.schedule_history:
                    f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"Error saving schedule history: {e}")
    
    def _calculate_drift_factor(self) -> float:
        """Calculate adjustment factor based on data drift."""
        if not self.drift_history:
            return 1.0
        
        # Get the most recent drift metrics
        recent_drift = self.drift_history[-1]
        drift_scores = []
        
        # Extract drift scores from the metrics
        for feature, metrics in recent_drift.get("drift_metrics", {}).items():
            # Collect all the ks_statistic values
            if "ks_statistic" in metrics:
                drift_scores.append(metrics["ks_statistic"])
        
        if not drift_scores:
            return 1.0
        
        # Calculate average drift score
        avg_drift = sum(drift_scores) / len(drift_scores)
        
        # Get thresholds from config
        thresholds = self.config.get("dynamic_scheduling", {}).get("drift_thresholds", {})
        low = thresholds.get("low", 0.1)
        medium = thresholds.get("medium", 0.3)
        high = thresholds.get("high", 0.5)
        
        # Calculate adjustment factor
        if avg_drift > high:
            # Significant drift detected - halve the interval
            return 0.5
        elif avg_drift > medium:
            # Moderate drift - reduce by 25%
            return 0.75
        elif avg_drift < low:
            # Low drift - increase interval by 25%
            return 1.25
        else:
            # Normal drift - keep the same interval
            return 1.0
    
    def _calculate_performance_factor(self) -> float:
        """Calculate adjustment factor based on performance metrics."""
        if len(self.metrics_history) < 2:
            return 1.0
        
        # Get the most recent metrics
        recent_metrics = self.metrics_history[-1]["metrics"]
        previous_metrics = self.metrics_history[-2]["metrics"]
        
        # Get thresholds from config
        thresholds = self.config.get("dynamic_scheduling", {}).get("performance_thresholds", {})
        
        # Initialization
        degradation_rate = 0
        count = 0
        
        # Check for degradation in each metric
        for metric, value in recent_metrics.items():
            if metric in previous_metrics:
                if metric in ["accuracy", "f1", "precision", "recall", "r2"]:
                    # For these metrics, higher is better
                    change = previous_metrics[metric] - value
                    threshold = thresholds.get(f"{metric}_drop", 0.05)
                    
                    if change > 0:  # There is degradation
                        # Normalize change to threshold
                        degradation_rate += min(change / threshold, 2.0)
                        count += 1
                
                elif metric in ["rmse", "mae"]:
                    # For these metrics, lower is better
                    if previous_metrics[metric] > 0:  # Avoid division by zero
                        change = (value - previous_metrics[metric]) / previous_metrics[metric]
                        threshold = thresholds.get(f"{metric}_increase", 0.15)
                        
                        if change > 0:  # There is degradation
                            # Normalize change to threshold
                            degradation_rate += min(change / threshold, 2.0)
                            count += 1
        
        # Calculate average degradation rate
        if count > 0:
            avg_degradation = degradation_rate / count
            
            # Calculate adjustment factor
            if avg_degradation > 1.5:
                # Severe degradation - reduce interval by 50%
                return 0.5
            elif avg_degradation > 1.0:
                # Significant degradation - reduce interval by 25%
                return 0.75
            elif avg_degradation > 0.5:
                # Mild degradation - reduce interval by 10%
                return 0.9
            else:
                # Minimal degradation - keep the same interval
                return 1.0
        else:
            # No comparable metrics found
            return 1.0
    
    def _calculate_resource_factor(self) -> float:
        """
        Calculate adjustment factor based on resource utilization.
        
        This is a placeholder for a more sophisticated resource-based adjustment.
        In a production environment, this would take into account factors like:
        - CPU/GPU utilization
        - Memory usage
        - Training cost
        - Inference demand
        """
        # For demonstration, return a fixed factor
        # In a real implementation, this would analyze system metrics
        return 1.0
    
    def calculate_next_retraining_date(self, model_id: str) -> Tuple[datetime, Dict]:
        """Calculate the next retraining date for a specific model."""
        # Find the last retraining date for this model
        last_retraining = None
        for entry in reversed(self.schedule_history):
            if entry.get("model_id") == model_id:
                last_retraining = entry
                break
        
        if last_retraining is None:
            # No previous retraining - use current date and base interval
            last_date = datetime.now()
            next_interval = self.base_interval
            factors = {
                "drift_factor": 1.0,
                "performance_factor": 1.0,
                "resource_factor": 1.0
            }
        else:
            # Use the last retraining date and calculate new interval
            last_date = datetime.fromisoformat(last_retraining.get("scheduled_date"))
            
            # Calculate adjustment factors
            drift_factor = self._calculate_drift_factor()
            performance_factor = self._calculate_performance_factor()
            resource_factor = self._calculate_resource_factor()
            
            # Combine factors to get the overall adjustment
            combined_factor = drift_factor * performance_factor * resource_factor
            
            # Apply to base interval
            next_interval = last_retraining.get("interval_days", self.base_interval) * combined_factor
            
            # Apply limits
            next_interval = max(self.min_interval, min(self.max_interval, next_interval))
            
            factors = {
                "drift_factor": drift_factor,
                "performance_factor": performance_factor,
                "resource_factor": resource_factor,
                "combined_factor": combined_factor
            }
        
        # Calculate next date
        next_date = last_date + timedelta(days=next_interval)
        
        # Return the next date and interval
        return next_date, {
            "interval_days": next_interval,
            "factors": factors
        }
    
    def update_schedule(self, model_id: str) -> Dict:
        """Update the retraining schedule for a specific model."""
        next_date, details = self.calculate_next_retraining_date(model_id)
        
        # Create schedule entry
        schedule_entry = {
            "model_id": model_id,
            "scheduled_date": next_date.isoformat(),
            "interval_days": details["interval_days"],
            "factors": details["factors"],
            "updated_at": datetime.now().isoformat()
        }
        
        # Add to history
        self.schedule_history.append(schedule_entry)
        
        # Save updated history
        self._save_schedule_history()
        
        logger.info(f"Updated schedule for model {model_id}")
        logger.info(f"Next retraining date: {next_date.strftime('%Y-%m-%d')}")
        logger.info(f"Interval: {details['interval_days']:.1f} days")
        logger.info(f"Adjustment factors: {json.dumps(details['factors'])}")
        
        return schedule_entry
    
    def get_models_due_for_retraining(self) -> List[str]:
        """Get list of models that are due for retraining based on dynamic schedule."""
        due_models = []
        models_processed = set()
        now = datetime.now()
        
        # Check schedule history for due dates
        for entry in reversed(self.schedule_history):
            model_id = entry.get("model_id")
            
            # Skip if already processed this model
            if model_id in models_processed:
                continue
            
            models_processed.add(model_id)
            
            # Check if due date has passed
            scheduled_date = datetime.fromisoformat(entry.get("scheduled_date"))
            if scheduled_date <= now:
                due_models.append(model_id)
                logger.info(f"Model {model_id} is due for retraining (scheduled on {scheduled_date.strftime('%Y-%m-%d')})")
        
        return due_models
    
    def log_retraining_event(self, model_id: str, reason: str):
        """Log when a model is retrained."""
        # Record retraining event
        event = {
            "model_id": model_id,
            "event_type": "retrained",
            "timestamp": datetime.now().isoformat(),
            "reason": reason
        }
        
        # Save event to log file
        log_path = "data/monitoring/retraining_events.jsonl"
        
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            logger.error(f"Error logging retraining event: {e}")
        
        # Update schedule after retraining
        self.update_schedule(model_id)
        
        logger.info(f"Logged retraining event for model {model_id}")
        logger.info(f"Reason: {reason}")

def main():
    """Main function to run the dynamic scheduler."""
    parser = argparse.ArgumentParser(description="Dynamic Scheduler for ML Model Retraining")
    parser.add_argument("--config", default="config/monitoring_config.json", help="Path to configuration file")
    parser.add_argument("--model-id", default="global", help="Model ID to update schedule for")
    parser.add_argument("--list-due", action="store_true", help="List models due for retraining")
    parser.add_argument("--update-schedule", action="store_true", help="Update retraining schedule")
    parser.add_argument("--log-retraining", action="store_true", help="Log a retraining event")
    parser.add_argument("--reason", default="manual", help="Reason for retraining (when using --log-retraining)")
    
    args = parser.parse_args()
    
    # Initialize scheduler
    scheduler = DynamicScheduler(config_path=args.config)
    
    # Process commands
    if args.list_due:
        due_models = scheduler.get_models_due_for_retraining()
        if due_models:
            print(f"Models due for retraining: {', '.join(due_models)}")
        else:
            print("No models are currently due for retraining")
    
    if args.update_schedule:
        schedule = scheduler.update_schedule(args.model_id)
        print(f"Updated schedule for model {args.model_id}")
        print(f"Next retraining: {schedule['scheduled_date']}")
        print(f"Interval: {schedule['interval_days']:.1f} days")
    
    if args.log_retraining:
        scheduler.log_retraining_event(args.model_id, args.reason)
        print(f"Logged retraining event for model {args.model_id}")
        print(f"Reason: {args.reason}")
        
        # Automatically update schedule after retraining
        schedule = scheduler.update_schedule(args.model_id)
        print(f"Updated schedule. Next retraining: {schedule['scheduled_date']}")

if __name__ == "__main__":
    main() 