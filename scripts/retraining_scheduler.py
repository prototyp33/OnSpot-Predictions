#!/usr/bin/env python
"""
RetrainingScheduler for OnSpot Predictive Model.

This module implements time-based scheduling for model retraining,
supporting both interval-based and calendar-based maintenance windows.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RetrainingScheduler:
    """
    Handles scheduling of model retraining based on time intervals
    and calendar-based maintenance windows.
    """
    
    def __init__(self, config_path: str = 'config/retraining_config.json'):
        """
        Initialize the retraining scheduler.
        
        Args:
            config_path: Path to the retraining configuration file
        """
        self.config = self._load_config(config_path)
        self.base_interval_days = self.config['time_based'].get('default_interval_days', 30)
        self.scheduled_windows = {}
        
        # Initialize scheduled windows from config
        if 'maintenance_windows' in self.config['time_based']:
            for model_id, window in self.config['time_based']['maintenance_windows'].items():
                self.add_maintenance_window(
                    model_id,
                    day_of_month=window.get('day_of_month'),
                    day_of_week=window.get('day_of_week'),
                    hour=window.get('hour', 3),
                    minute=window.get('minute', 0)
                )
        
        logger.info(f"Initialized RetrainingScheduler with {len(self.scheduled_windows)} maintenance windows")
        logger.info(f"Default retraining interval: {self.base_interval_days} days")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        if not os.path.exists(config_path):
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                'time_based': {
                    'enabled': True,
                    'default_interval_days': 30
                }
            }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {
                'time_based': {
                    'enabled': True,
                    'default_interval_days': 30
                }
            }
    
    def add_maintenance_window(self, model_id: str, day_of_month: Optional[int] = None, 
                              day_of_week: Optional[int] = None, hour: int = 3, minute: int = 0) -> None:
        """
        Add a calendar-based maintenance window for a specific model.
        
        Args:
            model_id: ID of the model to add a maintenance window for
            day_of_month: Day of the month (1-31) for monthly scheduling
            day_of_week: Day of the week (0-6, where 0 is Monday) for weekly scheduling
            hour: Hour of the day (0-23) for the maintenance window
            minute: Minute of the hour (0-59) for the maintenance window
        """
        if day_of_month is not None and (day_of_month < 1 or day_of_month > 31):
            raise ValueError(f"Invalid day_of_month: {day_of_month}. Must be between 1 and 31.")
        
        if day_of_week is not None and (day_of_week < 0 or day_of_week > 6):
            raise ValueError(f"Invalid day_of_week: {day_of_week}. Must be between 0 and 6.")
        
        if hour < 0 or hour > 23:
            raise ValueError(f"Invalid hour: {hour}. Must be between 0 and 23.")
        
        if minute < 0 or minute > 59:
            raise ValueError(f"Invalid minute: {minute}. Must be between 0 and 59.")
        
        self.scheduled_windows[model_id] = {
            'day_of_month': day_of_month,
            'day_of_week': day_of_week,
            'hour': hour,
            'minute': minute
        }
        
        logger.info(f"Added maintenance window for model {model_id}: " +
                  f"day_of_month={day_of_month}, day_of_week={day_of_week}, " +
                  f"time={hour:02d}:{minute:02d}")
    
    def get_last_training_time(self, model_id: str) -> datetime:
        """
        Get the time when a model was last trained.
        
        Args:
            model_id: ID of the model to get the last training time for
            
        Returns:
            datetime: The last training time, or epoch if not found
        """
        # TODO: Implement fetching from database or model metadata
        # For now, return a mock value for development
        # In production, this should query your model metadata storage
        
        # Mock implementation
        model_file = f"models/{model_id}.pkl"
        if os.path.exists(model_file):
            # Use file modification time as a proxy for training time
            mod_time = os.path.getmtime(model_file)
            return datetime.fromtimestamp(mod_time)
        
        # If model not found, assume it was trained 30 days ago
        return datetime.now() - timedelta(days=30)
    
    def is_in_maintenance_window(self, model_id: str, current_time: Optional[datetime] = None) -> bool:
        """
        Check if the current time falls within a maintenance window for a model.
        
        Args:
            model_id: ID of the model to check
            current_time: The time to check, defaults to now
            
        Returns:
            bool: True if in maintenance window, False otherwise
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Check if model has a scheduled maintenance window
        if model_id not in self.scheduled_windows:
            # Fall back to interval-based scheduling
            last_trained = self.get_last_training_time(model_id)
            days_since_training = (current_time - last_trained).days
            in_window = days_since_training >= self.base_interval_days
            
            if in_window:
                logger.info(f"Model {model_id} is due for retraining (interval-based): " +
                          f"{days_since_training} days since last training")
            
            return in_window
        
        # Check against calendar-based window
        window = self.scheduled_windows[model_id]
        
        # Check day of month (e.g., 1st of every month)
        if window['day_of_month'] is not None and current_time.day != window['day_of_month']:
            return False
        
        # Check day of week (e.g., every Monday)
        if window['day_of_week'] is not None and current_time.weekday() != window['day_of_week']:
            return False
        
        # Check time of day (hour and minute)
        if current_time.hour != window['hour'] or current_time.minute != window['minute']:
            return False
        
        logger.info(f"Model {model_id} is in maintenance window (calendar-based)")
        return True
    
    def get_next_maintenance_window(self, model_id: str, from_time: Optional[datetime] = None) -> datetime:
        """
        Get the next scheduled maintenance window for a model.
        
        Args:
            model_id: ID of the model to get the next window for
            from_time: The time to start looking from, defaults to now
            
        Returns:
            datetime: The start time of the next maintenance window
        """
        if from_time is None:
            from_time = datetime.now()
        
        # If no specific window is scheduled, use interval-based scheduling
        if model_id not in self.scheduled_windows:
            last_trained = self.get_last_training_time(model_id)
            next_training = last_trained + timedelta(days=self.base_interval_days)
            
            # If next_training is in the past, return current time
            if next_training < from_time:
                return from_time
            
            return next_training
        
        # Calculate next calendar-based window
        window = self.scheduled_windows[model_id]
        next_time = from_time.replace(hour=window['hour'], minute=window['minute'], second=0, microsecond=0)
        
        # Handle day of month scheduling
        if window['day_of_month'] is not None:
            if from_time.day > window['day_of_month'] or (from_time.day == window['day_of_month'] and 
                                                         (from_time.hour > window['hour'] or 
                                                          (from_time.hour == window['hour'] and 
                                                           from_time.minute >= window['minute']))):
                # Move to next month
                if from_time.month == 12:
                    next_time = next_time.replace(year=from_time.year + 1, month=1, day=window['day_of_month'])
                else:
                    next_time = next_time.replace(month=from_time.month + 1, day=window['day_of_month'])
            else:
                # Still in current month
                next_time = next_time.replace(day=window['day_of_month'])
        
        # Handle day of week scheduling
        elif window['day_of_week'] is not None:
            days_ahead = window['day_of_week'] - from_time.weekday()
            if days_ahead < 0:  # Target day already happened this week
                days_ahead += 7
            elif days_ahead == 0:  # Target is today, check time
                if from_time.hour > window['hour'] or (from_time.hour == window['hour'] and from_time.minute >= window['minute']):
                    days_ahead = 7  # Move to next week
            
            next_time = next_time.replace(day=from_time.day) + timedelta(days=days_ahead)
        
        return next_time
    
    def get_models_due_for_retraining(self, current_time: Optional[datetime] = None) -> List[str]:
        """
        Get a list of models that are due for retraining.
        
        Args:
            current_time: The time to check against, defaults to now
            
        Returns:
            List[str]: List of model IDs that are due for retraining
        """
        if current_time is None:
            current_time = datetime.now()
        
        due_models = []
        
        # Check models with scheduled windows
        for model_id in self.scheduled_windows.keys():
            if self.is_in_maintenance_window(model_id, current_time):
                due_models.append(model_id)
        
        # TODO: Check models without scheduled windows
        # This would require querying all available models in the system
        
        return due_models

    def get_retraining_schedule(self, days_ahead: int = 30) -> Dict[str, List[datetime]]:
        """
        Generate a retraining schedule for the specified number of days.
        
        Args:
            days_ahead: Number of days to schedule ahead
            
        Returns:
            Dict[str, List[datetime]]: Dictionary mapping model IDs to lists of scheduled retraining times
        """
        schedule = {}
        now = datetime.now()
        end_date = now + timedelta(days=days_ahead)
        
        # Schedule for models with defined maintenance windows
        for model_id in self.scheduled_windows.keys():
            schedule[model_id] = []
            next_window = self.get_next_maintenance_window(model_id, now)
            
            while next_window <= end_date:
                schedule[model_id].append(next_window)
                next_window = self.get_next_maintenance_window(model_id, next_window + timedelta(minutes=1))
        
        # TODO: Add interval-based scheduling for other models
        
        return schedule


if __name__ == "__main__":
    """Example usage of the RetrainingScheduler."""
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Model Retraining Scheduler")
    parser.add_argument("--config", default="config/retraining_config.json", help="Path to config file")
    parser.add_argument("--list-models", action="store_true", help="List models due for retraining")
    parser.add_argument("--schedule", type=int, default=30, help="Generate schedule for N days ahead")
    parser.add_argument("--check-model", type=str, help="Check if a specific model is due for retraining")
    parser.add_argument("--next-window", type=str, help="Get next retraining window for a model")
    
    args = parser.parse_args()
    
    # Initialize the scheduler
    scheduler = RetrainingScheduler(config_path=args.config)
    
    # Process commands
    if args.list_models:
        due_models = scheduler.get_models_due_for_retraining()
        if due_models:
            print(f"Models due for retraining: {', '.join(due_models)}")
        else:
            print("No models are currently due for retraining.")
    
    elif args.schedule:
        schedule = scheduler.get_retraining_schedule(days_ahead=args.schedule)
        print(f"\nRetraining schedule for the next {args.schedule} days:")
        print("=" * 50)
        
        for model_id, times in schedule.items():
            if times:
                print(f"\nModel: {model_id}")
                print("-" * 20)
                for t in times:
                    print(f"  - {t.strftime('%Y-%m-%d %H:%M')}")
            else:
                print(f"\nModel: {model_id} (no scheduled retrainings in this period)")
    
    elif args.check_model:
        is_due = scheduler.is_in_maintenance_window(args.check_model)
        if is_due:
            print(f"Model {args.check_model} is currently due for retraining.")
        else:
            next_window = scheduler.get_next_maintenance_window(args.check_model)
            print(f"Model {args.check_model} is not currently due for retraining.")
            print(f"Next retraining window: {next_window.strftime('%Y-%m-%d %H:%M')}")
    
    elif args.next_window:
        next_window = scheduler.get_next_maintenance_window(args.next_window)
        print(f"Next retraining window for model {args.next_window}: {next_window.strftime('%Y-%m-%d %H:%M')}")
    
    else:
        # Default: show a brief schedule for all models
        schedule = scheduler.get_retraining_schedule(days_ahead=7)
        print("\nRetraining schedule for the next 7 days:")
        print("=" * 50)
        
        for model_id, times in schedule.items():
            if times:
                print(f"\nModel: {model_id}")
                print("-" * 20)
                for t in times:
                    print(f"  - {t.strftime('%Y-%m-%d %H:%M')}")
            else:
                print(f"\nModel: {model_id} (no scheduled retrainings in the next 7 days)") 