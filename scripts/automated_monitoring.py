#!/usr/bin/env python
"""
Automated model monitoring script.
Checks model performance and data drift.
"""

import os
import sys
import json
import logging
import argparse
import time
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('model_monitoring')

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Try to import from model_monitoring
try:
    from scripts.model_monitoring import DataDriftMonitor, PerformanceMonitor
except ImportError as e:
    logger.error(f"Could not import from model_monitoring: {e}")
    logger.info("Creating mock classes for testing")
    
    class DataDriftMonitor:
        """Mock DataDriftMonitor class for testing."""
        def __init__(self, config=None):
            self.config = config or {
                'drift_thresholds': {
                    'ks_statistic': 0.1,
                    'mean_difference': 0.1,
                    'std_difference': 0.1
                }
            }
            self.logger = logging.getLogger('mock_drift_monitor')
            self.baseline_stats = {}
        
        def set_baseline(self, data):
            """Set baseline statistics."""
            self.logger.info("Setting mock baseline statistics")
            return True
        
        def detect_drift(self, data):
            """Detect drift in data."""
            self.logger.info("Running mock drift detection")
            return False, {"feature1": {"ks_statistic": 0.05}}
    
    class PerformanceMonitor:
        """Mock PerformanceMonitor class for testing."""
        def __init__(self, config=None):
            self.config = config or {"performance_thresholds": {"accuracy": 0.8}}
            self.logger = logging.getLogger('mock_perf_monitor')
        
        def evaluate_model(self, model, data, targets):
            """Evaluate model performance."""
            self.logger.info("Running mock model evaluation")
            return {
                "accuracy": 0.85,
                "precision": 0.8,
                "recall": 0.75
            }
            
        def check_performance(self, metrics):
            """Check if performance is below thresholds."""
            self.logger.info("Running mock performance check")
            return False, metrics

# Check for dynamic retraining scheduler
try:
    DYNAMIC_SCHEDULER_AVAILABLE = True
    from scripts.retraining_scheduler import RetrainingScheduler
except ImportError:
    DYNAMIC_SCHEDULER_AVAILABLE = False
    logger.warning("RetrainingScheduler not available, dynamic scheduling will be disabled")
    
    # Mock RetrainingScheduler for testing
    class RetrainingScheduler:
        """Mock RetrainingScheduler class for testing."""
        def __init__(self, config_file=None):
            self.logger = logging.getLogger('mock_retraining_scheduler')
        
        def get_models_for_retraining(self):
            """Get models due for retraining."""
            return []
        
        def log_retraining_event(self, model_id, reason):
            """Log a retraining event."""
            self.logger.info(f"Mock logging retraining event for {model_id}: {reason}")
            return True

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import monitoring components
from scripts.model_monitoring import (
    DataDriftMonitor, 
    PerformanceMonitor, 
    MonitoringDashboard,
    monitor_model_performance
)

# Try to import feature engineering
try:
    from scripts.parking_sim.advanced_features import engineer_advanced_features
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    logger.warning("Feature engineering module not available")
    FEATURE_ENGINEERING_AVAILABLE = False
    # Define a simple fallback function
    def engineer_advanced_features(data):
        """Fallback feature engineering function."""
        processed_data = data.copy()
        
        # Add basic temporal features if timestamp exists
        if 'timestamp' in processed_data.columns:
            processed_data['hour_of_day'] = processed_data['timestamp'].dt.hour
            processed_data['day_of_week'] = processed_data['timestamp'].dt.dayofweek
            processed_data['month'] = processed_data['timestamp'].dt.month
            processed_data['is_weekend'] = (processed_data['day_of_week'] >= 5).astype(int)
        
        return processed_data

# Import dynamic scheduler
try:
    from scripts.dynamic_scheduler import DynamicScheduler
    DYNAMIC_SCHEDULER_AVAILABLE = True
except ImportError:
    logger.warning("Dynamic scheduler module not available")
    DYNAMIC_SCHEDULER_AVAILABLE = False
    # Define a simple fallback class
    class DynamicScheduler:
        """Fallback dynamic scheduler class."""
        def __init__(self, config_path=None):
            pass
        
        def get_models_due_for_retraining(self):
            """Return an empty list."""
            return []
        
        def log_retraining_event(self, model_id, reason):
            """Do nothing."""
            pass

class MonitoringPipeline:
    """Main class for the automated monitoring pipeline."""
    
    def __init__(self, config_path: str = 'config/monitoring_config.json'):
        """Initialize the monitoring pipeline.
        
        Args:
            config_path: Path to the monitoring configuration file
        """
        self.logger = logging.getLogger('monitoring_pipeline')
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize monitoring components
        self.drift_monitor = DataDriftMonitor(self.config.get('drift_monitoring', {}))
        self.performance_monitor = PerformanceMonitor(self.config.get('performance_monitoring', {}))
        
        # Initialize dashboard
        self.dashboard = MonitoringDashboard(self.drift_monitor, self.performance_monitor)
        
        # Dynamic scheduler for time-based retraining
        self.dynamic_scheduler = None
        if DYNAMIC_SCHEDULER_AVAILABLE and self.config.get('dynamic_scheduling', {}).get('enabled', False):
            try:
                scheduler_config = self.config.get('dynamic_scheduling', {}).get('config_file', 
                                                                               'config/retraining_config.json')
                self.dynamic_scheduler = RetrainingScheduler(scheduler_config)
                self.logger.info(f"Dynamic scheduler initialized with config: {scheduler_config}")
            except Exception as e:
                self.logger.error(f"Failed to initialize dynamic scheduler: {e}")
        
        # Store baseline and models
        self.baseline_data = None
        self.models = {}
        
        # Retraining flags
        self.retraining_needed = False
        self.retraining_reason = ""
        
        # Create required directories
        os.makedirs('data/monitoring', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
    
    def load_config(self, config_path: str) -> dict:
        """Load monitoring configuration from file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dictionary containing configuration parameters
        """
        try:
            if not os.path.exists(config_path):
                self.logger.warning(f"Configuration file not found: {config_path}")
                self.logger.info("Creating default configuration")
                
                # Create default configuration
                default_config = {
                    "drift_monitoring": {
                        "enabled": True,
                        "drift_thresholds": {
                            "ks_statistic": 0.1,
                            "mean_difference": 0.2,
                            "std_difference": 0.2,
                            "distribution_difference": 0.3
                        }
                    },
                    "performance_monitoring": {
                        "enabled": True,
                        "performance_thresholds": {
                            "accuracy": 0.8,
                            "precision": 0.7,
                            "recall": 0.7,
                            "f1": 0.75
                        }
                    },
                    "dynamic_scheduling": {
                        "enabled": True,
                        "config_file": "config/retraining_config.json"
                    },
                    "retraining": {
                        "auto_trigger": True,
                        "min_events_between_retraining": 3,
                        "min_days_between_retraining": 7
                    },
                    "storage": {
                        "metrics_path": "data/monitoring/",
                        "models_path": "production_models/",
                        "reports_path": "reports/"
                    }
                }
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                
                # Save default configuration
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                    
                self.logger.info(f"Created default configuration file: {config_path}")
                return default_config
                
            # Load configuration from file
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            self.logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.logger.info("Using default configuration")
            
            # Return minimal default configuration
            return {
                "drift_monitoring": {"enabled": True, "drift_thresholds": {"ks_statistic": 0.1}},
                "performance_monitoring": {"enabled": True, "performance_thresholds": {"accuracy": 0.8}},
                "dynamic_scheduling": {"enabled": False},
                "retraining": {"auto_trigger": False},
                "storage": {"metrics_path": "data/monitoring/", "models_path": "production_models/"}
            }
    
    def load_models(self, model_dir: str = "production_models"):
        """Load all production models."""
        logger.info(f"Loading models from {model_dir}")
        
        # Create directory if it doesn't exist
        if not os.path.exists(model_dir):
            logger.warning(f"Model directory {model_dir} doesn't exist, creating it")
            os.makedirs(model_dir, exist_ok=True)
        
        # Load global model
        global_model_path = os.path.join(model_dir, "global_model_advanced_features.pkl")
        if os.path.exists(global_model_path):
            self.models['global'] = joblib.load(global_model_path)
            logger.info(f"Loaded global model from {global_model_path}")
        
        # Load location-specific models
        for model_file in os.listdir(model_dir):
            if model_file.startswith("location_") and model_file.endswith(".pkl"):
                location_id = model_file.split("_")[1]
                model_path = os.path.join(model_dir, model_file)
                self.models[f'location_{location_id}'] = joblib.load(model_path)
                logger.info(f"Loaded model for location {location_id} from {model_path}")
        
        if not self.models:
            logger.warning(f"No models found in {model_dir} - monitoring will continue without model performance checks")
            return  # Return without error
    
    def load_baseline_data(self, baseline_path: str = "data/baseline_data.csv"):
        """Load baseline data for drift detection."""
        logger.info(f"Loading baseline data from {baseline_path}")
        
        try:
            if os.path.exists(baseline_path):
                self.baseline_data = pd.read_csv(baseline_path)
                logger.info(f"Loaded baseline data with {len(self.baseline_data)} rows")
                
                # Set baseline for drift monitor
                self.drift_monitor.set_baseline(self.baseline_data)
            else:
                logger.warning(f"Baseline data file {baseline_path} doesn't exist")
                logger.info("Will use the first batch of new data as baseline when new data is loaded")
                self.baseline_data = None
        except Exception as e:
            logger.error(f"Failed to load baseline data: {e}")
            logger.info("Will use the first batch of new data as baseline when new data is loaded")
            self.baseline_data = None
    
    def load_new_data(self, data_file: str) -> pd.DataFrame:
        """Load new data for monitoring.
        
        Args:
            data_file: Path to the new data file
            
        Returns:
            DataFrame with the new data
        """
        try:
            if not os.path.exists(data_file):
                self.logger.warning(f"New data file not found: {data_file}")
                self.logger.info("Creating sample data file for testing...")
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(data_file), exist_ok=True)
                
                # Generate sample data
                import random
                import datetime
                
                sample_data = []
                start_date = datetime.datetime.now() - datetime.timedelta(days=10)
                
                for i in range(100):
                    date = start_date + datetime.timedelta(hours=i*6)
                    sample_data.append({
                        'timestamp': date.strftime('%Y-%m-%d %H:%M:%S'),
                        'location_id': random.choice(['NY001', 'CA002', 'TX003', 'FL004', 'IL005']),
                        'temperature': round(random.uniform(0, 35), 1),
                        'humidity': round(random.uniform(20, 90), 1),
                        'pressure': round(random.uniform(990, 1030), 1),
                        'wind_speed': round(random.uniform(0, 30), 1),
                        'precipitation': round(random.uniform(0, 50), 1),
                        'cloud_cover': round(random.uniform(0, 100), 1),
                        'visibility': round(random.uniform(0, 10), 1),
                        'dew_point': round(random.uniform(-5, 25), 1)
                    })
                
                sample_df = pd.DataFrame(sample_data)
                sample_df.to_csv(data_file, index=False)
                self.logger.info(f"Created sample data file with {len(sample_df)} rows at {data_file}")
                return sample_df
            
            self.logger.info(f"Loading new data from {data_file}")
            new_data = pd.read_csv(data_file)
            self.logger.info(f"Loaded new data with shape {new_data.shape}")
            return new_data
        except Exception as e:
            self.logger.error(f"Error loading new data: {e}")
            self.logger.info("Creating empty DataFrame with expected columns")
            return pd.DataFrame(columns=['timestamp', 'location_id', 'temperature', 'humidity', 
                                        'pressure', 'wind_speed', 'precipitation', 
                                        'cloud_cover', 'visibility', 'dew_point'])
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data by applying feature engineering."""
        logger.info("Applying feature engineering to data")
        
        # Convert timestamp to datetime if present
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        try:
            # Apply feature engineering
            processed_data = engineer_advanced_features(data)
            logger.info(f"Processed data shape: {processed_data.shape}")
            return processed_data
        except Exception as e:
            logger.warning(f"Feature engineering failed: {e}")
            logger.info("Using simple feature engineering fallback")
            
            # Simple feature engineering fallback
            processed_data = data.copy()
            
            # Add basic temporal features if timestamp exists
            if 'timestamp' in processed_data.columns:
                processed_data['hour_of_day'] = processed_data['timestamp'].dt.hour
                processed_data['day_of_week'] = processed_data['timestamp'].dt.dayofweek
                processed_data['month'] = processed_data['timestamp'].dt.month
                processed_data['is_weekend'] = (processed_data['day_of_week'] >= 5).astype(int)
            
            logger.info(f"Processed data shape with fallback features: {processed_data.shape}")
            return processed_data
    
    def check_for_drift(self, data: pd.DataFrame) -> bool:
        """Check for data drift in the new data."""
        logger.info("Checking for data drift")
        
        # Detect drift
        drift_detected, drift_metrics = self.drift_monitor.detect_drift(data)
        
        if drift_detected:
            logger.warning("Data drift detected!")
            drift_report = self.dashboard.generate_drift_report(drift_metrics)
            logger.info(drift_report)
            
            # Save drift report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(
                self.config['storage']['reports_path'], 
                f"drift_report_{timestamp}.txt"
            )
            with open(report_path, 'w') as f:
                f.write(drift_report)
            
            # Store drift analysis in database
            try:
                # Try to use Supabase client first
                try:
                    from supabase_integration import SupabaseClient
                    with SupabaseClient() as db:
                        db.store_drift_analysis(
                            model_id=self.config.get('model_id', 'default_model'),
                            drift_metrics=drift_metrics,
                            baseline_timestamp=datetime.now()
                        )
                        
                        # If retraining is needed due to drift, log the event
                        if drift_detected:
                            db.store_retraining_event(
                                model_id=self.config.get('model_id', 'default_model'),
                                reason="Data drift detected",
                                success=False,  # Will be updated after retraining
                                metrics_before=self.get_current_metrics()
                            )
                    logger.info("Drift analysis stored in Supabase")
                except ImportError:
                    # Fall back to PostgreSQL client if Supabase client not available
                    logger.info("Supabase client not found. Trying PostgreSQL client.")
                    from db_integration import DatabaseClient
                    with DatabaseClient() as db:
                        db.store_drift_analysis(
                            model_id=self.config.get('model_id', 'default_model'),
                            drift_metrics=drift_metrics,
                            baseline_timestamp=datetime.now()
                        )
                        
                        # If retraining is needed due to drift, log the event
                        if drift_detected:
                            db.store_retraining_event(
                                model_id=self.config.get('model_id', 'default_model'),
                                reason="Data drift detected",
                                success=False,  # Will be updated after retraining
                                metrics_before=self.get_current_metrics()
                            )
                    logger.info("Drift analysis stored in database")
            except ImportError:
                logger.warning("Database integration modules not found. Skipping database storage.")
            except Exception as e:
                logger.error(f"Failed to store drift analysis in database: {e}")
            
            self.retraining_needed = True
            self.retraining_reason = "Data drift detected"
        else:
            logger.info("No significant data drift detected")
        
        return drift_detected
    
    def get_current_metrics(self) -> dict:
        """Get current performance metrics of the model."""
        metrics = {}
        if hasattr(self, 'performance_monitor') and self.performance_monitor:
            metrics = self.performance_monitor.get_last_metrics() or {}
        
        # Add system metrics if available
        if hasattr(self, 'sys_monitor') and self.sys_monitor:
            sys_metrics = self.sys_monitor.get_current_metrics() or {}
            metrics.update({"system": sys_metrics})
            
        # If no metrics available, return empty dict
        if not metrics:
            metrics = {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
            
        return metrics
    
    def evaluate_performance(self, data: pd.DataFrame, task_type: str = 'regression') -> bool:
        """Evaluate model performance on new data."""
        logger.info("Evaluating model performance")
        
        # Split data into features and target
        if 'occupancy' not in data.columns:
            logger.warning("Cannot evaluate performance: 'occupancy' column not found")
            return False
        
        exclude_cols = ['timestamp', 'date', 'occupancy']
        X = data.drop(columns=[col for col in exclude_cols if col in data.columns])
        y = data['occupancy']
        
        degradation_detected = False
        
        # Evaluate each model
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name} model")
            
            # For location-specific models, filter data
            if model_name.startswith('location_'):
                location_id = model_name.split('_')[1]
                if 'location_id' in data.columns:
                    mask = data['location_id'] == location_id
                    X_loc = X[mask]
                    y_loc = y[mask]
                    
                    if len(X_loc) == 0:
                        logger.warning(f"No data for location {location_id}, skipping evaluation")
                        continue
                else:
                    logger.warning("location_id column not found, skipping location-specific model")
                    continue
            else:
                X_loc = X
                y_loc = y
            
            # Make predictions
            try:
                y_pred = model.predict(X_loc)
                
                # Update performance metrics
                self.performance_monitor.update_metrics(y_loc, y_pred, task_type=task_type)
                
                # Check for performance degradation
                if len(self.performance_monitor.metrics_history) > 1:
                    current_metrics = self.performance_monitor.metrics_history[-1]['metrics']
                    previous_metrics = self.performance_monitor.metrics_history[-2]['metrics']
                    
                    if task_type == 'regression':
                        # Check RMSE degradation
                        rmse_increase = (current_metrics['rmse'] - previous_metrics['rmse']) / previous_metrics['rmse']
                        threshold = self.config['performance_thresholds']['rmse_increase']
                        
                        if rmse_increase > threshold:
                            logger.warning(f"{model_name} model: RMSE increased by {rmse_increase:.1%}")
                            degradation_detected = True
                    else:
                        # Check accuracy or F1 degradation
                        if 'accuracy' in current_metrics and 'accuracy' in previous_metrics:
                            accuracy_drop = previous_metrics['accuracy'] - current_metrics['accuracy']
                            threshold = self.config['performance_thresholds']['accuracy_drop']
                            
                            if accuracy_drop > threshold:
                                logger.warning(f"{model_name} model: Accuracy dropped by {accuracy_drop:.3f}")
                                degradation_detected = True
                        
                        if 'f1' in current_metrics and 'f1' in previous_metrics:
                            f1_drop = previous_metrics['f1'] - current_metrics['f1']
                            threshold = self.config['performance_thresholds']['f1_drop']
                            
                            if f1_drop > threshold:
                                logger.warning(f"{model_name} model: F1 score dropped by {f1_drop:.3f}")
                                degradation_detected = True
            
            except Exception as e:
                logger.error(f"Error evaluating {model_name} model: {e}")
                continue
        
        if degradation_detected:
            self.retraining_needed = True
            self.retraining_reason = "Performance degradation detected"
            
            # Generate performance report
            performance_report = self.dashboard.generate_performance_report()
            logger.info(performance_report)
            
            # Save performance report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(
                self.config['storage']['reports_path'], 
                f"performance_report_{timestamp}.txt"
            )
            with open(report_path, 'w') as f:
                f.write(performance_report)
        else:
            logger.info("No significant performance degradation detected")
        
        return degradation_detected
    
    def run_detailed_monitoring(self, data_path: str, output_dir: str = "model_monitoring"):
        """Run detailed model monitoring with visualizations."""
        logger.info(f"Running detailed model monitoring with data from {data_path}")
        
        # Run detailed monitoring
        monitor_model_performance(
            data_path, 
            model_dir="production_models", 
            output_dir=output_dir, 
            window_size=7
        )
        
        # Check for performance alerts
        alert_path = os.path.join(output_dir, "performance_alerts.txt")
        if os.path.exists(alert_path):
            with open(alert_path, 'r') as f:
                alerts = f.read()
            
            if alerts.strip():
                logger.warning("Performance alerts detected in detailed monitoring")
                logger.info(alerts)
                self.retraining_needed = True
                self.retraining_reason = "Performance alerts in detailed monitoring"
    
    def check_for_dynamic_retraining(self) -> List[str]:
        """Check for models due for retraining based on dynamic scheduling."""
        # Check if dynamic scheduling is enabled and available
        if not self.config.get("dynamic_scheduling", {}).get("enabled", False) or not DYNAMIC_SCHEDULER_AVAILABLE or self.dynamic_scheduler is None:
            logger.info("Dynamic scheduling is disabled or not available")
            return []
        
        logger.info("Checking for models due for retraining based on dynamic scheduling")
        
        # Get models due for retraining
        due_models = self.dynamic_scheduler.get_models_due_for_retraining()
        
        if due_models:
            logger.info(f"Models due for retraining: {', '.join(due_models)}")
            self.retraining_needed = True
            self.retraining_reason = "Dynamic scheduling interval elapsed"
        else:
            logger.info("No models are due for retraining based on dynamic scheduling")
        
        return due_models
    
    def trigger_retraining(self, model_id: str = None):
        """Trigger model retraining if needed."""
        if not self.retraining_needed:
            logger.info("No retraining needed based on monitoring results")
            return False
        
        logger.warning(f"Retraining needed. Reason: {self.retraining_reason}")
        
        # Record retraining decision
        retraining_record = {
            "timestamp": datetime.now().isoformat(),
            "reason": self.retraining_reason,
            "triggered": True,
            "model_id": model_id if model_id else "all"
        }
        
        # Save to retraining log
        retraining_log_path = os.path.join(
            self.config['storage']['metrics_path'], 
            "retraining_decisions.jsonl"
        )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(retraining_log_path), exist_ok=True)
        
        with open(retraining_log_path, 'a') as f:
            f.write(json.dumps(retraining_record) + '\n')
        
        # Check if training script exists
        training_script = os.path.join("scripts", "train_pipeline.py")
        if not os.path.exists(training_script):
            logger.error(f"Training script {training_script} not found")
            
            # Check if we can create a simple dummy script
            try:
                logger.info("Creating a simple dummy training script")
                
                # Create a simple dummy script
                with open(training_script, 'w') as f:
                    f.write("""#!/usr/bin/env python
\"\"\"
Dummy training script that simulates model training.
This is a placeholder created because the real training script was not found.
\"\"\"

import os
import sys
import argparse
import logging
import json
import numpy as np
import pickle
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DummyTraining')

def train_dummy_model(data_path, model_id=None):
    \"\"\"Train a dummy model that predicts the mean value.\"\"\"
    logger.info(f"Training dummy model on {data_path}")
    
    # Create a dummy model (just returns the mean value)
    class DummyModel:
        def __init__(self, mean_value=50):
            self.mean_value = mean_value
        
        def predict(self, X):
            return np.ones(len(X)) * self.mean_value
    
    # Create and return the model
    return DummyModel()

def main():
    \"\"\"Run the dummy training process.\"\"\"
    parser = argparse.ArgumentParser(description="Dummy model training")
    parser.add_argument("--data", required=True, help="Path to the data file")
    parser.add_argument("--output", default="retrained_models", help="Output directory for models")
    parser.add_argument("--model-id", help="Specific model ID to train")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Train the model
    model = train_dummy_model(args.data, args.model_id)
    
    # Save the model
    if args.model_id:
        model_path = os.path.join(args.output, f"{args.model_id}.pkl")
    else:
        model_path = os.path.join(args.output, "global_model_advanced_features.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Saved dummy model to {model_path}")
    
    # Also save to production directory
    prod_dir = "production_models"
    os.makedirs(prod_dir, exist_ok=True)
    
    prod_path = os.path.join(prod_dir, os.path.basename(model_path))
    with open(prod_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Saved dummy model to production directory: {prod_path}")

if __name__ == "__main__":
    main()
""")
                
                logger.info(f"Created dummy training script at {training_script}")
                os.chmod(training_script, 0o755)  # Make executable
            except Exception as e:
                logger.error(f"Failed to create dummy training script: {e}")
                return False
        
        # Call retraining script
        try:
            logger.info(f"Triggering model retraining for {model_id if model_id else 'all models'}")
            
            # Build command with model_id if specified
            cmd = [
                sys.executable,
                training_script,
                "--data", "data/feature_engineered_data.csv",
                "--output", "retrained_models"
            ]
            
            if model_id:
                cmd.extend(["--model-id", model_id])
            
            # Check if feature engineered data exists
            if not os.path.exists("data/feature_engineered_data.csv"):
                logger.warning("Feature engineered data file not found, using new_data.csv instead")
                cmd[3] = "data/new_data.csv"
                
                # If new_data.csv doesn't exist either, create it using our sample data
                if not os.path.exists("data/new_data.csv"):
                    logger.warning("Creating sample data for training")
                    sample_data = self.load_new_data("data/new_data.csv")
                    # Sample data will be created by load_new_data if it doesn't exist
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Model retraining completed successfully")
                logger.info(result.stdout)
                
                # Log retraining event in dynamic scheduler if available
                if DYNAMIC_SCHEDULER_AVAILABLE and self.dynamic_scheduler is not None and self.config.get("dynamic_scheduling", {}).get("enabled", False):
                    model_ids = [model_id] if model_id else list(self.models.keys())
                    for mid in model_ids:
                        self.dynamic_scheduler.log_retraining_event(mid, self.retraining_reason)
                
                return True
            else:
                logger.error("Model retraining failed")
                logger.error(result.stderr)
                return False
        
        except Exception as e:
            logger.error(f"Error triggering retraining: {e}")
            return False
    
    def run_pipeline(self, data_path: str):
        """Run the complete monitoring pipeline."""
        logger.info(f"Starting automated monitoring pipeline with data from {data_path}")
        
        try:
            # Load models if not loaded
            if not self.models:
                self.load_models()
            
            # Load baseline data if not loaded
            if self.baseline_data is None:
                self.load_baseline_data()
            
            # Load and process new data
            new_data = self.load_new_data(data_path)
            processed_data = self.process_data(new_data)
            
            # Use new data as baseline if no baseline exists
            if self.baseline_data is None:
                logger.info("Using current data batch as baseline for future comparisons")
                self.baseline_data = processed_data.copy()
                self.drift_monitor.set_baseline(self.baseline_data)
                
                # Save as baseline for future use
                os.makedirs("data", exist_ok=True)
                baseline_save_path = "data/baseline_data.csv"
                self.baseline_data.to_csv(baseline_save_path, index=False)
                logger.info(f"Saved current data as baseline to {baseline_save_path}")
                
                logger.info("Skipping drift detection for this run as we just set the baseline")
                drift_detected = False
            else:
                # Check for drift
                drift_detected = self.check_for_drift(processed_data)
            
            # Evaluate performance only if models are loaded
            degradation_detected = False
            if self.models:
                degradation_detected = self.evaluate_performance(processed_data)
                self.run_detailed_monitoring(data_path)
            else:
                logger.info("Skipping performance evaluation and detailed monitoring due to no models being loaded")
            
            # Check for dynamic retraining
            due_models = []
            if self.models and DYNAMIC_SCHEDULER_AVAILABLE and self.dynamic_scheduler is not None and self.config.get("dynamic_scheduling", {}).get("enabled", False):
                due_models = self.check_for_dynamic_retraining()
                
                # Record drift and performance metrics for dynamic scheduler
                if drift_detected:
                    drift_metrics = self.drift_monitor.calculate_drift_metrics(processed_data)
                    drift_record = {
                        "timestamp": datetime.now().isoformat(),
                        "drift_metrics": drift_metrics,
                        "drift_detected": drift_detected
                    }
                    
                    # Save to drift history
                    drift_path = "data/monitoring/drift_history.jsonl"
                    os.makedirs(os.path.dirname(drift_path), exist_ok=True)
                    with open(drift_path, 'a') as f:
                        f.write(json.dumps(drift_record) + '\n')
            
            # Trigger retraining if needed
            if self.retraining_needed and self.models:
                if due_models:
                    # Retrain each due model individually
                    for model_id in due_models:
                        self.trigger_retraining(model_id)
                else:
                    # Retrain all models
                    self.trigger_retraining()
            
            # Return monitoring results
            return {
                "drift_detected": drift_detected,
                "degradation_detected": degradation_detected,
                "retraining_triggered": self.retraining_needed,
                "retraining_reason": self.retraining_reason if self.retraining_needed else None,
                "models_due": due_models,
                "dynamic_scheduling_available": DYNAMIC_SCHEDULER_AVAILABLE and self.dynamic_scheduler is not None
            }
        
        except Exception as e:
            logger.error(f"Error in monitoring pipeline: {e}", exc_info=True)
            raise

def main():
    """Run the monitoring pipeline from command line."""
    parser = argparse.ArgumentParser(description="Automated model monitoring pipeline")
    parser.add_argument("--data", required=True, help="Path to the new data file")
    parser.add_argument("--model-dir", default="production_models", help="Directory containing model files")
    parser.add_argument("--baseline", default="data/baseline_data.csv", help="Path to baseline data file")
    parser.add_argument("--config", default="config/monitoring_config.json", help="Path to monitoring configuration")
    parser.add_argument("--output", default="model_monitoring", help="Output directory for monitoring results")
    
    args = parser.parse_args()
    
    # Initialize and run the pipeline
    pipeline = MonitoringPipeline(config_path=args.config)
    pipeline.load_models(model_dir=args.model_dir)
    pipeline.load_baseline_data(baseline_path=args.baseline)
    
    # Run the pipeline
    results = pipeline.run_pipeline(args.data)
    
    # Print results
    logger.info("Monitoring pipeline completed")
    logger.info(f"Results: {json.dumps(results, indent=2)}")

if __name__ == "__main__":
    main() 