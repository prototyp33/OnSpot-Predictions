#!/usr/bin/env python
"""
Supabase Integration Example

This script demonstrates the proper usage of the refactored Supabase monitoring system,
including standardized decorator usage and centralized configuration.
"""

import os
import sys
import json
import logging
from datetime import datetime
import time
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Import required components
try:
    from scripts.supabase_integration import SupabaseClient
    from scripts.supabase_monitor import get_monitor, monitor_operation, monitor_connection
    from scripts.supabase_dashboard import SupabaseDashboard
    from scripts.supabase_metrics_extension import integrate_supabase_monitoring, load_config
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise


def display_config():
    """Display the current configuration values from the config file."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'config',
        'retraining_config.json'
    )
    
    config = load_config(config_path)
    supabase_thresholds = config.get('supabase_thresholds', {})
    
    print("\n=== Supabase Monitoring Configuration ===")
    print(f"Configuration loaded from: {config_path}")
    print("\nMonitoring Thresholds:")
    for key, value in supabase_thresholds.items():
        print(f"  {key}: {value}")
    print("==========================================\n")


def run_decorated_example():
    """Example demonstrating proper use of the monitoring decorators."""
    print("\n=== Demonstrating Standard Decorator Usage ===")
    
    # Create a Supabase client
    with SupabaseClient() as client:
        # Insert example data (using properly decorated methods)
        print("\nPerforming database operations...")
        
        try:
            # Store drift analysis data
            client.store_drift_analysis(
                model_id="example_model",
                drift_metrics={
                    "feature1": {"ks_statistic": 0.15, "p_value": 0.05},
                    "feature2": {"ks_statistic": 0.08, "p_value": 0.12}
                }
            )
            print("✓ store_drift_analysis - Operation completed")
            
            # Store retraining event
            client.store_retraining_event(
                model_id="example_model",
                reason="Scheduled retraining",
                success=True
            )
            print("✓ store_retraining_event - Operation completed")
            
            # Store system health data
            client.store_system_health(
                component="api_server",
                status="operational",
                metrics={"response_time_ms": 230, "error_rate": 0.001}
            )
            print("✓ store_system_health - Operation completed")
            
            # Query drift metrics
            metrics = client.get_drift_metrics(model_id="example_model")
            print("✓ get_drift_metrics - Operation completed")
            
            # Update a drift record
            client.update_drift_analysis(
                record_id="example_id",
                updates={"drift_score": 0.35}
            )
            print("✓ update_drift_analysis - Operation completed")
            
        except Exception as e:
            # In a real implementation, these operations might fail when connecting to
            # an actual database, but in dummy mode they should succeed
            logger.error(f"Operation failed: {e}")
    
    # Retrieve and display the collected metrics
    monitor = get_monitor()
    metrics = monitor.get_health_metrics()
    
    print("\nOperation counts:")
    for op, count in metrics['operations']['operation_counts'].items():
        print(f"  {op}: {count}")


def test_integration_with_config():
    """Example demonstrating integration with the monitoring system using configuration file."""
    print("\n=== Testing Configuration Integration ===")
    
    # Initialize the metrics extension with configuration
    extension = integrate_supabase_monitoring()
    
    if extension:
        print("\nSuccessfully integrated Supabase monitoring with the following thresholds:")
        for key, value in extension.config.items():
            print(f"  {key}: {value}")
        
        # Generate a performance report
        report = extension.generate_supabase_performance_report()
        print("\nPerformance Report Summary:")
        print("  " + report.split("\n")[0])  # First line of the report
        print("  Status: " + report.split("\n")[3].split(": ")[1])  # Status line
    else:
        print("\nFailed to integrate Supabase monitoring")


def main():
    """Main function demonstrating proper use of the refactored components."""
    print("=== Supabase Monitoring System Example ===")
    
    # Display the current configuration
    display_config()
    
    # Run examples demonstrating proper decorator usage
    run_decorated_example()
    
    # Test integration with config file
    test_integration_with_config()
    
    print("\nExample completed. Check the metrics collected above.")


if __name__ == "__main__":
    main() 