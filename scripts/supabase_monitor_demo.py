#!/usr/bin/env python
"""
Supabase Monitoring Demonstration

This script demonstrates the usage of the Supabase monitoring tools
by simulating database operations and visualizing the metrics.
"""

import os
import sys
import time
import random
import json
import logging
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import local modules
try:
    from scripts.supabase_monitor import get_monitor, monitor_operation, monitor_connection
    from scripts.supabase_dashboard import SupabaseDashboard
    from scripts.supabase_metrics_extension import integrate_supabase_monitoring
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise


class SupabaseSimulator:
    """
    Simulates Supabase database operations for demonstration purposes.
    """
    
    def __init__(self, error_rate=0.05, latency_base=50, latency_variance=30):
        """
        Initialize the Supabase simulator.
        
        Args:
            error_rate: Probability of operation failure (0-1)
            latency_base: Base latency in ms
            latency_variance: Maximum variance in latency in ms
        """
        self.error_rate = error_rate
        self.latency_base = latency_base
        self.latency_variance = latency_variance
        self.connected = False
        self.monitor = get_monitor()
        logger.info("Supabase simulator initialized")
    
    @monitor_connection
    def connect(self):
        """Simulate connecting to Supabase."""
        # Simulate connection time
        latency = self._simulate_latency()
        time.sleep(latency / 1000)  # Convert ms to seconds
        
        # Simulate potential connection failure
        if random.random() < self.error_rate:
            self.connected = False
            raise Exception("Simulated connection failure")
        
        self.connected = True
        logger.info(f"Connected to Supabase (simulated latency: {latency:.2f}ms)")
        return True
    
    def disconnect(self):
        """Simulate disconnecting from Supabase."""
        self.connected = False
        logger.info("Disconnected from Supabase")
    
    @monitor_operation("insert", "drift_analysis")
    def insert_drift_analysis(self, records):
        """
        Simulate inserting records into the drift_analysis table.
        
        Args:
            records: Number of records to insert
        
        Returns:
            Dict with operation result
        """
        self._check_connection()
        
        # Simulate operation time (more records = more time)
        latency = self._simulate_latency(multiplier=records/10)
        time.sleep(latency / 1000)
        
        # Simulate potential operation failure
        if random.random() < self.error_rate:
            raise Exception("Simulated insert failure in drift_analysis table")
        
        logger.info(f"Inserted {records} records into drift_analysis (simulated latency: {latency:.2f}ms)")
        return {"success": True, "records": records}
    
    @monitor_operation("query", "drift_analysis")
    def query_drift_analysis(self, filter_criteria=None):
        """
        Simulate querying records from the drift_analysis table.
        
        Args:
            filter_criteria: Optional criteria to filter results
        
        Returns:
            Dict with operation result
        """
        self._check_connection()
        
        # Simulate operation time
        latency = self._simulate_latency()
        time.sleep(latency / 1000)
        
        # Simulate potential operation failure
        if random.random() < self.error_rate:
            raise Exception("Simulated query failure in drift_analysis table")
        
        # Simulate result set
        result_count = random.randint(1, 50)
        logger.info(f"Queried drift_analysis with {result_count} results (simulated latency: {latency:.2f}ms)")
        return {"success": True, "results": result_count}
    
    @monitor_operation("insert", "retraining_events")
    def insert_retraining_event(self):
        """
        Simulate inserting a record into the retraining_events table.
        
        Returns:
            Dict with operation result
        """
        self._check_connection()
        
        # Simulate operation time
        latency = self._simulate_latency()
        time.sleep(latency / 1000)
        
        # Simulate potential operation failure
        if random.random() < self.error_rate:
            raise Exception("Simulated insert failure in retraining_events table")
        
        logger.info(f"Inserted retraining event (simulated latency: {latency:.2f}ms)")
        return {"success": True, "records": 1}
    
    @monitor_operation("query", "business_metrics")
    def query_business_metrics(self):
        """
        Simulate querying records from the business_metrics table.
        
        Returns:
            Dict with operation result
        """
        self._check_connection()
        
        # Simulate operation time
        latency = self._simulate_latency()
        time.sleep(latency / 1000)
        
        # Simulate potential operation failure
        if random.random() < self.error_rate:
            raise Exception("Simulated query failure in business_metrics table")
        
        # Simulate result set
        result_count = random.randint(1, 20)
        logger.info(f"Queried business_metrics with {result_count} results (simulated latency: {latency:.2f}ms)")
        return {"success": True, "results": result_count}
    
    @monitor_operation("insert", "system_health")
    def insert_system_health(self):
        """
        Simulate inserting a record into the system_health table.
        
        Returns:
            Dict with operation result
        """
        self._check_connection()
        
        # Simulate operation time
        latency = self._simulate_latency()
        time.sleep(latency / 1000)
        
        # Simulate potential operation failure
        if random.random() < self.error_rate:
            raise Exception("Simulated insert failure in system_health table")
        
        logger.info(f"Inserted system health record (simulated latency: {latency:.2f}ms)")
        return {"success": True, "records": 1}
    
    def _check_connection(self):
        """Check if connected and raise an exception if not."""
        if not self.connected:
            raise Exception("Not connected to Supabase")
    
    def _simulate_latency(self, multiplier=1.0):
        """
        Simulate operation latency.
        
        Args:
            multiplier: Multiplier for the latency
        
        Returns:
            Simulated latency in ms
        """
        base = self.latency_base * multiplier
        variance = self.latency_variance * multiplier
        
        # Add some randomness with occasional spikes
        if random.random() < 0.05:  # 5% chance of latency spike
            return base + random.uniform(variance, variance * 5)
        else:
            return base + random.uniform(0, variance)


def simulate_workload(simulator, duration_seconds=60, operation_interval=1.0):
    """
    Simulate a workload of operations over a period of time.
    
    Args:
        simulator: SupabaseSimulator instance
        duration_seconds: Duration of the simulation in seconds
        operation_interval: Average interval between operations in seconds
    """
    start_time = time.time()
    end_time = start_time + duration_seconds
    
    operations = [
        lambda: simulator.insert_drift_analysis(random.randint(1, 20)),
        lambda: simulator.query_drift_analysis(),
        lambda: simulator.insert_retraining_event(),
        lambda: simulator.query_business_metrics(),
        lambda: simulator.insert_system_health()
    ]
    
    logger.info(f"Starting simulation for {duration_seconds} seconds")
    
    # Connect to Supabase
    try:
        simulator.connect()
    except Exception as e:
        logger.error(f"Initial connection failed: {e}")
        # Try again
        try:
            time.sleep(1)
            simulator.connect()
        except Exception as e:
            logger.error(f"Second connection attempt failed: {e}")
    
    # Run operations until duration is reached
    while time.time() < end_time:
        # Occasionally disconnect and reconnect to simulate connection issues
        if random.random() < 0.02:  # 2% chance per operation
            logger.info("Simulating disconnection...")
            simulator.disconnect()
            try:
                simulator.connect()
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
        
        # Choose and execute a random operation
        operation = random.choice(operations)
        try:
            operation()
        except Exception as e:
            logger.error(f"Operation failed: {e}")
        
        # Wait before next operation
        wait_time = random.uniform(operation_interval * 0.5, operation_interval * 1.5)
        time.sleep(wait_time)
    
    # Disconnect at the end
    simulator.disconnect()
    
    logger.info(f"Simulation completed after {time.time() - start_time:.2f} seconds")


def run_demo(args):
    """
    Run the Supabase monitoring demonstration.
    
    Args:
        args: Command line arguments
    """
    # Create simulator with specified error rate
    simulator = SupabaseSimulator(error_rate=args.error_rate)
    
    # Initialize dashboard if visualization is enabled
    dashboard = None
    if args.visualize:
        dashboard = SupabaseDashboard()
    
    # Run the simulation
    logger.info(f"Running simulation with {args.duration}s duration, {args.error_rate:.1%} error rate")
    simulate_workload(simulator, duration_seconds=args.duration, operation_interval=args.interval)
    
    # Get the metrics after simulation
    monitor = get_monitor()
    metrics = monitor.get_health_metrics()
    
    # Print metrics summary
    logger.info("Simulation complete! Final metrics:")
    print(json.dumps(metrics, indent=2))
    
    # Generate visualizations if enabled
    if args.visualize:
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Generate plots
            latency_file = os.path.join(args.output_dir, "supabase_latency.png")
            success_file = os.path.join(args.output_dir, "supabase_success_rate.png")
            ops_file = os.path.join(args.output_dir, "supabase_operations.png")
            errors_file = os.path.join(args.output_dir, "supabase_errors.png")
            
            dashboard.plot_latency_trends(latency_file)
            dashboard.plot_success_rate_trends(success_file)
            dashboard.plot_operation_distribution(ops_file)
            dashboard.plot_error_distribution(errors_file)
            
            # Export metrics to CSV
            csv_file = os.path.join(args.output_dir, "supabase_metrics.csv")
            dashboard.export_metrics_to_csv(csv_file)
            
            # Generate report
            report_file = os.path.join(args.output_dir, "supabase_report.txt")
            report = dashboard.generate_performance_report(report_file)
            
            logger.info(f"Output files saved to {args.output_dir}")
        else:
            # Show interactive plots
            dashboard.plot_latency_trends()
            dashboard.plot_success_rate_trends()
            dashboard.plot_operation_distribution()
            dashboard.plot_error_distribution()
            
            # Print report
            report = dashboard.generate_performance_report()
            print("\n" + report)
    
    # Integrate with performance monitoring if enabled
    if args.integrate:
        extension = integrate_supabase_monitoring()
        if extension:
            report = extension.generate_supabase_performance_report()
            print("\nIntegrated Performance Report:")
            print(report)
        else:
            logger.error("Failed to integrate with performance monitoring")


def main():
    """Main function for the demonstration script."""
    parser = argparse.ArgumentParser(description="Supabase Monitoring Demonstration")
    parser.add_argument("--duration", type=int, default=60, help="Duration of simulation in seconds")
    parser.add_argument("--interval", type=float, default=1.0, help="Average interval between operations in seconds")
    parser.add_argument("--error-rate", type=float, default=0.05, help="Probability of operation failure (0-1)")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--output-dir", help="Directory to save output files")
    parser.add_argument("--integrate", action="store_true", help="Integrate with performance monitoring")
    
    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main() 