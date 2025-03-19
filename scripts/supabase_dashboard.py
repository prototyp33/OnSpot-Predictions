#!/usr/bin/env python
"""
Supabase Metrics Dashboard

This module provides a lightweight dashboard for visualizing 
Supabase database performance metrics.
"""

import os
import sys
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import time
import threading
import uuid
import argparse

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
    from scripts.supabase_monitor import get_monitor
    from scripts.supabase_metrics_extension import integrate_supabase_monitoring
    from scripts.supabase_integration import SupabaseClient
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise

class SupabaseDashboard:
    """
    Dashboard for visualizing Supabase performance metrics.
    
    This class provides methods to generate visualizations of Supabase
    connection health, query performance, and resource utilization.
    """
    
    def __init__(self):
        """Initialize the Supabase dashboard."""
        # Get the Supabase monitor singleton
        self.supabase_monitor = get_monitor()
        
        # Initialize the Supabase performance monitor
        self.performance_monitor = integrate_supabase_monitoring()
        
        # Initialize metrics history for time-series visualization
        self.metrics_history = []
        self.max_history_points = 100
        self.collection_interval = 60  # seconds
        
        # Start background metrics collection
        self._start_metrics_collection()
    
    def _start_metrics_collection(self):
        """Start a background thread for continuous metrics collection."""
        def metrics_collector():
            while True:
                # Get current metrics
                metrics = self.supabase_monitor.get_health_metrics()
                
                # Add timestamp if not present
                if 'timestamp' not in metrics:
                    metrics['timestamp'] = datetime.now().isoformat()
                
                # Add to history
                self.metrics_history.append(metrics)
                
                # Limit history size
                if len(self.metrics_history) > self.max_history_points:
                    self.metrics_history = self.metrics_history[-self.max_history_points:]
                
                # Sleep for the interval
                time.sleep(self.collection_interval)
        
        # Start the thread
        thread = threading.Thread(target=metrics_collector, daemon=True)
        thread.start()
    
    def generate_health_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of Supabase health metrics.
        
        Returns:
            Dict containing summary metrics
        """
        # Get latest health status
        health_status = self.performance_monitor.get_latest_supabase_health()
        
        # Create summary dict
        summary = {
            'timestamp': datetime.now().isoformat(),
            'status': health_status['status'],
            'alert_level': health_status.get('alert_level', 'info'),
            'issues': health_status.get('issues', []),
            'connection_stats': {
                'success_rate': health_status['metrics']['connection']['success_rate'],
                'avg_latency_ms': health_status['metrics']['connection']['avg_latency_ms']
            },
            'operation_stats': {
                'success_rate': health_status['metrics']['operations']['success_rate'],
                'total_count': health_status['metrics']['operations']['total_count'],
                'avg_latency_ms': health_status['metrics']['performance']['latency_avg_ms'],
                'p95_latency_ms': health_status['metrics']['performance']['latency_p95_ms']
            }
        }
        
        return summary
    
    def plot_latency_trends(self, output_file: Optional[str] = None) -> None:
        """
        Plot trends in Supabase operation latency.
        
        Args:
            output_file: Optional file path to save the plot
        """
        if not self.metrics_history:
            logger.warning("No metrics history available for plotting")
            return
        
        # Extract data points
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in self.metrics_history]
        avg_latencies = [m['performance']['latency_avg_ms'] for m in self.metrics_history]
        p95_latencies = [m['performance']['latency_p95_ms'] for m in self.metrics_history]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, avg_latencies, 'b-', label='Average Latency')
        plt.plot(timestamps, p95_latencies, 'r-', label='P95 Latency')
        
        # Add threshold lines
        thresholds = self.performance_monitor.config['supabase_thresholds']
        plt.axhline(y=thresholds['latency_threshold_ms'], color='b', linestyle='--', 
                   label=f"Avg Latency Threshold ({thresholds['latency_threshold_ms']}ms)")
        plt.axhline(y=thresholds['p95_latency_threshold_ms'], color='r', linestyle='--',
                   label=f"P95 Latency Threshold ({thresholds['p95_latency_threshold_ms']}ms)")
        
        # Add labels and title
        plt.xlabel('Time')
        plt.ylabel('Latency (ms)')
        plt.title('Supabase Operation Latency Trends')
        plt.legend()
        plt.grid(True)
        
        # Save or show
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Latency trend plot saved to {output_file}")
        else:
            plt.show()
    
    def plot_success_rate_trends(self, output_file: Optional[str] = None) -> None:
        """
        Plot trends in Supabase operation success rates.
        
        Args:
            output_file: Optional file path to save the plot
        """
        if not self.metrics_history:
            logger.warning("No metrics history available for plotting")
            return
        
        # Extract data points
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in self.metrics_history]
        operation_success_rates = [m['operations']['success_rate'] for m in self.metrics_history]
        connection_success_rates = [m['connection']['success_rate'] for m in self.metrics_history]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, operation_success_rates, 'g-', label='Operation Success Rate')
        plt.plot(timestamps, connection_success_rates, 'b-', label='Connection Success Rate')
        
        # Add threshold lines
        thresholds = self.performance_monitor.config['supabase_thresholds']
        plt.axhline(y=(1 - thresholds['operation_failure_rate']), color='g', linestyle='--',
                   label=f"Operation Success Threshold ({(1-thresholds['operation_failure_rate']):.2%})")
        plt.axhline(y=(1 - thresholds['connection_failure_rate']), color='b', linestyle='--',
                   label=f"Connection Success Threshold ({(1-thresholds['connection_failure_rate']):.2%})")
        
        # Add labels and title
        plt.xlabel('Time')
        plt.ylabel('Success Rate')
        plt.title('Supabase Success Rate Trends')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1.05)  # Set y-axis limit from 0 to 1.05
        
        # Save or show
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Success rate trend plot saved to {output_file}")
        else:
            plt.show()
    
    def plot_operation_distribution(self, output_file: Optional[str] = None) -> None:
        """
        Plot distribution of Supabase operations.
        
        Args:
            output_file: Optional file path to save the plot
        """
        # Get latest metrics
        metrics = self.supabase_monitor.get_health_metrics()
        
        # Extract operation counts
        operation_counts = metrics['operations']['operation_counts']
        
        if not operation_counts:
            logger.warning("No operation data available for plotting")
            return
        
        # Prepare data
        operations = list(operation_counts.keys())
        counts = list(operation_counts.values())
        
        # Sort by count
        operations, counts = zip(*sorted(zip(operations, counts), key=lambda x: x[1], reverse=True))
        
        # Create figure
        plt.figure(figsize=(12, 6))
        plt.bar(operations, counts)
        
        # Add labels and title
        plt.xlabel('Operation')
        plt.ylabel('Count')
        plt.title('Supabase Operation Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save or show
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Operation distribution plot saved to {output_file}")
        else:
            plt.show()
    
    def plot_error_distribution(self, output_file: Optional[str] = None) -> None:
        """
        Plot distribution of Supabase operation errors.
        
        Args:
            output_file: Optional file path to save the plot
        """
        # Get latest metrics
        metrics = self.supabase_monitor.get_health_metrics()
        
        # Extract error counts
        error_counts = metrics['operations']['error_counts']
        
        if not error_counts:
            logger.info("No errors to plot")
            return
        
        # Prepare data
        operations = list(error_counts.keys())
        counts = list(error_counts.values())
        
        # Sort by count
        operations, counts = zip(*sorted(zip(operations, counts), key=lambda x: x[1], reverse=True))
        
        # Create figure
        plt.figure(figsize=(12, 6))
        plt.bar(operations, counts, color='r')
        
        # Add labels and title
        plt.xlabel('Operation')
        plt.ylabel('Error Count')
        plt.title('Supabase Error Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save or show
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Error distribution plot saved to {output_file}")
        else:
            plt.show()
    
    def export_metrics_to_csv(self, output_file: str) -> None:
        """
        Export Supabase metrics history to a CSV file.
        
        Args:
            output_file: Path to the output CSV file
        """
        if not self.metrics_history:
            logger.warning("No metrics history available for export")
            return
        
        # Create a list of flattened metrics
        flattened_metrics = []
        
        for metrics in self.metrics_history:
            flat_metrics = {
                'timestamp': metrics['timestamp'],
                'operations_total_count': metrics['operations']['total_count'],
                'operations_success_rate': metrics['operations']['success_rate'],
                'latency_avg_ms': metrics['performance']['latency_avg_ms'],
                'latency_p95_ms': metrics['performance']['latency_p95_ms'],
                'latency_p99_ms': metrics['performance']['latency_p99_ms'],
                'error_rate': metrics['performance']['error_rate'],
                'connection_attempts': metrics['connection']['attempts'],
                'connection_failures': metrics['connection']['failures'],
                'connection_success_rate': metrics['connection']['success_rate'],
                'connection_avg_latency_ms': metrics['connection']['avg_latency_ms']
            }
            
            flattened_metrics.append(flat_metrics)
        
        # Create DataFrame
        df = pd.DataFrame(flattened_metrics)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Metrics exported to {output_file}")
    
    def generate_performance_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            output_file: Optional file path to save the report
            
        Returns:
            String containing the performance report
        """
        # Get report from performance monitor
        report = self.performance_monitor.generate_supabase_performance_report()
        
        # Add additional information if needed
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Performance report saved to {output_file}")
        
        return report


def main():
    """Main function for the Supabase dashboard."""
    parser = argparse.ArgumentParser(description='Supabase Metrics Dashboard')
    parser.add_argument('--output-dir', help='Directory to save output files')
    parser.add_argument('--report', action='store_true', help='Generate performance report')
    parser.add_argument('--latency', action='store_true', help='Generate latency plot')
    parser.add_argument('--success', action='store_true', help='Generate success rate plot')
    parser.add_argument('--operations', action='store_true', help='Generate operations plot')
    parser.add_argument('--errors', action='store_true', help='Generate errors plot')
    parser.add_argument('--export', action='store_true', help='Export metrics to CSV')
    parser.add_argument('--all', action='store_true', help='Generate all outputs')
    
    args = parser.parse_args()
    
    # Create dashboard
    dashboard = SupabaseDashboard()
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate outputs
    if args.all or args.report:
        report_file = os.path.join(args.output_dir, 'supabase_performance_report.txt') if args.output_dir else None
        report = dashboard.generate_performance_report(report_file)
        if not args.output_dir:
            print(report)
    
    if args.all or args.latency:
        latency_file = os.path.join(args.output_dir, 'supabase_latency.png') if args.output_dir else None
        dashboard.plot_latency_trends(latency_file)
    
    if args.all or args.success:
        success_file = os.path.join(args.output_dir, 'supabase_success_rate.png') if args.output_dir else None
        dashboard.plot_success_rate_trends(success_file)
    
    if args.all or args.operations:
        operations_file = os.path.join(args.output_dir, 'supabase_operations.png') if args.output_dir else None
        dashboard.plot_operation_distribution(operations_file)
    
    if args.all or args.errors:
        errors_file = os.path.join(args.output_dir, 'supabase_errors.png') if args.output_dir else None
        dashboard.plot_error_distribution(errors_file)
    
    if args.all or args.export:
        export_file = os.path.join(args.output_dir, 'supabase_metrics.csv') if args.output_dir else 'supabase_metrics.csv'
        dashboard.export_metrics_to_csv(export_file)
    
    # If no specific outputs requested, print summary
    if not any([args.report, args.latency, args.success, args.operations, args.errors, args.export, args.all]):
        summary = dashboard.generate_health_summary()
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main() 