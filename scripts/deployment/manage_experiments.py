#!/usr/bin/env python3
"""CLI tool for managing A/B test experiments."""

import argparse
import json
import logging
import sys
from typing import Optional
from datetime import datetime
from .experiment_manager import ExperimentManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_experiment(manager: ExperimentManager, args: argparse.Namespace) -> None:
    """Create a new experiment."""
    try:
        test = manager.create_experiment(args.name)
        logger.info(f"Created experiment: {args.name}")
        logger.info(f"Variants: {test.variants}")
        logger.info(f"Traffic split: {test.traffic_split}")
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        sys.exit(1)

def list_experiments(manager: ExperimentManager, args: argparse.Namespace) -> None:
    """List active experiments."""
    experiments = manager.list_experiments()
    if not experiments:
        logger.info("No active experiments")
        return
        
    logger.info("Active experiments:")
    for name in experiments:
        test = manager.get_experiment(name)
        stats = test.get_statistics()
        
        logger.info(f"\nExperiment: {name}")
        logger.info(f"Started: {test.start_time}")
        logger.info(f"Duration: {datetime.now() - test.start_time}")
        logger.info("Sample sizes:")
        for variant, variant_stats in stats.items():
            logger.info(f"  {variant}: {variant_stats['sample_size']}")

def show_experiment(manager: ExperimentManager, args: argparse.Namespace) -> None:
    """Show details of a specific experiment."""
    test = manager.get_experiment(args.name)
    if not test:
        logger.error(f"Experiment not found: {args.name}")
        sys.exit(1)
        
    stats = test.get_statistics()
    significance = test.calculate_significance()
    
    logger.info(f"\nExperiment: {args.name}")
    logger.info(f"Started: {test.start_time}")
    logger.info(f"Duration: {datetime.now() - test.start_time}")
    logger.info(f"\nVariants: {test.variants}")
    logger.info(f"Traffic split: {test.traffic_split}")
    
    logger.info("\nStatistics:")
    for variant, variant_stats in stats.items():
        logger.info(f"\n{variant}:")
        for metric, value in variant_stats.items():
            logger.info(f"  {metric}: {value}")
            
    if significance:
        logger.info("\nSignificance Tests:")
        for metric, results in significance.items():
            logger.info(f"\n{metric}:")
            for key, value in results.items():
                logger.info(f"  {key}: {value}")

def end_experiment(manager: ExperimentManager, args: argparse.Namespace) -> None:
    """End an experiment and show results."""
    results = manager.end_experiment(args.name)
    if not results:
        logger.error(f"Failed to end experiment: {args.name}")
        sys.exit(1)
        
    logger.info(f"\nEnded experiment: {args.name}")
    logger.info(f"Duration: {results['duration']} seconds")
    logger.info(f"Winner: {results['winner']}")
    
    if results['improvements']:
        logger.info("\nImprovements:")
        for metric, improvement in results['improvements'].items():
            logger.info(f"  {metric}: {improvement:.2f}%")
            
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to: {args.output}")

def check_experiments(manager: ExperimentManager, args: argparse.Namespace) -> None:
    """Check and complete experiments meeting criteria."""
    completed = manager.check_and_complete_experiments()
    
    if not completed:
        logger.info("No experiments completed")
        return
        
    logger.info(f"\nCompleted {len(completed)} experiments:")
    for results in completed:
        logger.info(f"\nExperiment: {results['experiment']}")
        logger.info(f"Winner: {results['winner']}")
        
        if results['improvements']:
            logger.info("Improvements:")
            for metric, improvement in results['improvements'].items():
                logger.info(f"  {metric}: {improvement:.2f}%")
                
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(completed, f, indent=2, default=str)
        logger.info(f"\nResults saved to: {args.output}")

def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Manage A/B test experiments")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create experiment
    create_parser = subparsers.add_parser("create", help="Create a new experiment")
    create_parser.add_argument("name", help="Name of the experiment")
    
    # List experiments
    subparsers.add_parser("list", help="List active experiments")
    
    # Show experiment
    show_parser = subparsers.add_parser("show", help="Show experiment details")
    show_parser.add_argument("name", help="Name of the experiment")
    
    # End experiment
    end_parser = subparsers.add_parser("end", help="End an experiment")
    end_parser.add_argument("name", help="Name of the experiment")
    end_parser.add_argument("--output", "-o", help="Save results to file")
    
    # Check experiments
    check_parser = subparsers.add_parser("check", help="Check and complete experiments")
    check_parser.add_argument("--output", "-o", help="Save results to file")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    try:
        manager = ExperimentManager()
        
        commands = {
            "create": create_experiment,
            "list": list_experiments,
            "show": show_experiment,
            "end": end_experiment,
            "check": check_experiments
        }
        
        commands[args.command](manager, args)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 