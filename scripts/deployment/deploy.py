#!/usr/bin/env python3
"""
Deployment script for OnSpot Predictive Model.

This script automates the deployment process for the OnSpot prediction service:
1. Registers the best model version from experiments
2. Promotes it to production if it meets criteria
3. Configures deployment patterns (canary, shadow, A/B testing)
4. Manages container builds and deployments
5. Sets up monitoring and alerts

Usage:
  python deploy.py --strategy canary --traffic 20
  python deploy.py --promote --model-id abc123
  python deploy.py --rollback
"""

import os
import sys
import argparse
import logging
import json
import yaml
import time
import shutil
import subprocess
from datetime import datetime
import requests
from typing import Dict, List, Any, Optional, Union

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from model versioning system
from scripts.models import ModelVersioning, ModelRegistry

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_REGISTRY_PATH = os.environ.get('MODEL_REGISTRY_PATH', './model_registry')
PRODUCTION_MODELS_PATH = os.environ.get('PRODUCTION_MODELS_PATH', './production_models')
GATEWAY_API = os.environ.get('GATEWAY_API', 'http://localhost:8080')
DOCKER_COMPOSE_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../docker-compose.yml'))

# Initialize model versioning
versioning = ModelVersioning(
    registry_path=MODEL_REGISTRY_PATH,
    production_models_path=PRODUCTION_MODELS_PATH
)

def build_containers(force_rebuild=False):
    """Build Docker containers for the deployment."""
    logger.info("Building Docker containers...")
    
    cmd = ['docker-compose', '-f', DOCKER_COMPOSE_FILE, 'build']
    if force_rebuild:
        cmd.append('--no-cache')
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Container build completed successfully: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error building containers: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False

def start_deployment(detached=True):
    """Start the deployment containers."""
    logger.info("Starting deployment containers...")
    
    cmd = ['docker-compose', '-f', DOCKER_COMPOSE_FILE, 'up']
    if detached:
        cmd.append('-d')
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Deployment started successfully: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error starting deployment: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False

def stop_deployment():
    """Stop the deployment containers."""
    logger.info("Stopping deployment containers...")
    
    cmd = ['docker-compose', '-f', DOCKER_COMPOSE_FILE, 'down']
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Deployment stopped successfully: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error stopping deployment: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False

def restart_service(service_name):
    """Restart a specific service."""
    logger.info(f"Restarting service: {service_name}...")
    
    cmd = ['docker-compose', '-f', DOCKER_COMPOSE_FILE, 'restart', service_name]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Service {service_name} restarted successfully: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error restarting service {service_name}: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False

def check_health(wait_seconds=30, check_interval=5):
    """Check the health of all deployment services."""
    logger.info(f"Checking deployment health (waiting up to {wait_seconds} seconds)...")
    
    end_time = time.time() + wait_seconds
    services_healthy = False
    
    while time.time() < end_time:
        try:
            response = requests.get(f"{GATEWAY_API}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                services = health_data.get('services', {})
                
                # Check if all services are healthy
                all_healthy = True
                for service, status in services.items():
                    if status != 'healthy' and status != 'not_checked':
                        all_healthy = False
                        logger.warning(f"Service {service} is not healthy: {status}")
                
                if all_healthy:
                    logger.info("All services are healthy!")
                    services_healthy = True
                    break
            else:
                logger.warning(f"Health check failed with status code: {response.status_code}")
        
        except requests.RequestException as e:
            logger.warning(f"Error checking health: {str(e)}")
        
        logger.info(f"Waiting {check_interval} seconds before next health check...")
        time.sleep(check_interval)
    
    if not services_healthy:
        logger.error("Health check failed after waiting period")
    
    return services_healthy

def configure_deployment_strategy(strategy, traffic_percentage=None):
    """Configure the deployment strategy through the gateway API."""
    logger.info(f"Configuring deployment strategy: {strategy}")
    
    config = {'deployment_strategy': strategy}
    
    if traffic_percentage is not None:
        if strategy == 'canary':
            config['canary_traffic_percentage'] = float(traffic_percentage)
        elif strategy == 'shadow':
            config['shadow_traffic_percentage'] = float(traffic_percentage)
        elif strategy == 'ab_test':
            config['ab_test_b_traffic_percentage'] = float(traffic_percentage)
    
    try:
        response = requests.post(f"{GATEWAY_API}/config", json=config, timeout=10)
        if response.status_code == 200:
            logger.info(f"Deployment strategy configured successfully: {response.json()}")
            return True
        else:
            logger.error(f"Failed to configure deployment strategy: {response.status_code}")
            return False
    
    except requests.RequestException as e:
        logger.error(f"Error configuring deployment strategy: {str(e)}")
        return False

def find_best_model_candidates(model_name=None, min_performance=None):
    """Find best model candidates for deployment."""
    logger.info("Finding best model candidates for deployment...")
    
    # Get models from registry
    models = versioning.registry.list_models()
    
    # Filter by model name if specified
    if model_name:
        models = [m for m in models if m.get('name') == model_name]
    
    # Load model details and sort by performance
    candidates = []
    for model_info in models:
        model_id = model_info.get('model_id')
        if not model_id:
            continue
        
        try:
            _, metadata = versioning.registry.load_model(model_id, with_metadata=True)
            
            # Skip models without performance metrics
            if 'performance_metrics' not in metadata:
                continue
            
            # Get primary metric (assuming first metric is primary)
            metrics = metadata.get('performance_metrics', {})
            if not metrics:
                continue
            
            primary_metric = next(iter(metrics.items()), None)
            if not primary_metric:
                continue
            
            metric_name, metric_value = primary_metric
            
            # Skip if below minimum performance
            if min_performance is not None and metric_value < min_performance:
                continue
            
            # Add to candidates
            candidates.append({
                'model_id': model_id,
                'name': metadata.get('name', 'unknown'),
                'version': metadata.get('version', 'unknown'),
                'metric_name': metric_name,
                'metric_value': metric_value,
                'created_at': metadata.get('created_at', 'unknown')
            })
        
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}")
    
    # Sort by metric value (assuming higher is better)
    candidates.sort(key=lambda x: x['metric_value'], reverse=True)
    
    return candidates

def promote_model_to_production(model_id, deployment_type='production'):
    """Promote a model to production."""
    logger.info(f"Promoting model {model_id} to production...")
    
    try:
        # Promote the model
        model, metadata = versioning.promote_to_production(model_id)
        
        logger.info(f"Model {metadata.get('name')} version {metadata.get('version')} " +
                   f"promoted to production successfully")
        
        # Configure deployment if needed
        if deployment_type != 'production':
            configure_deployment_strategy(deployment_type)
            
            # Restart services to pick up new model
            restart_service('prediction-api')
            if deployment_type in ['shadow', 'canary', 'ab_test']:
                restart_service('shadow-api')
                restart_service('canary-api')
        
        return True
    
    except Exception as e:
        logger.error(f"Error promoting model to production: {str(e)}")
        return False

def rollback_to_previous_version(model_name=None):
    """Rollback to the previous production version of a model."""
    logger.info(f"Rolling back to previous production version of model: {model_name or 'all'}")
    
    try:
        # Get production models
        production_models = versioning.list_production_models()
        
        # Filter by model name if specified
        if model_name:
            production_models = [m for m in production_models if m.get('name') == model_name]
        
        if not production_models:
            logger.error(f"No production models found for {model_name or 'any model'}")
            return False
        
        # For each model, get its lineage and find the previous version
        for prod_model in production_models:
            model_name = prod_model.get('name')
            current_version = prod_model.get('version')
            
            if not model_name or not current_version:
                continue
            
            # Get model lineage
            lineage = versioning.get_model_lineage(model_name)
            
            # Sort by version (assuming semantic versioning)
            lineage.sort(key=lambda x: x.get('version', '0.0.0'), reverse=True)
            
            # Find current version index
            current_idx = None
            for i, model in enumerate(lineage):
                if model.get('version') == current_version:
                    current_idx = i
                    break
            
            if current_idx is None or current_idx >= len(lineage) - 1:
                logger.warning(f"No previous version found for {model_name}")
                continue
            
            # Get previous version
            previous = lineage[current_idx + 1]
            previous_id = previous.get('model_id')
            previous_version = previous.get('version')
            
            logger.info(f"Rolling back {model_name} from v{current_version} to v{previous_version}")
            
            # Promote previous version to production
            promote_model_to_production(previous_id)
        
        # Configure deployment to production
        configure_deployment_strategy('production')
        
        # Restart services
        restart_service('prediction-api')
        
        return True
    
    except Exception as e:
        logger.error(f"Error rolling back to previous version: {str(e)}")
        return False

def setup_deployment_environment(force_rebuild=False):
    """Set up the deployment environment."""
    logger.info("Setting up deployment environment...")
    
    # Build containers
    if not build_containers(force_rebuild):
        logger.error("Failed to build containers")
        return False
    
    # Start deployment
    if not start_deployment():
        logger.error("Failed to start deployment")
        return False
    
    # Check health
    if not check_health():
        logger.warning("Deployment health check failed, but continuing...")
    
    logger.info("Deployment environment setup completed successfully")
    return True

def main():
    """Main entry point for the deployment script."""
    parser = argparse.ArgumentParser(description='Deploy OnSpot Predictive Model')
    
    # Deployment actions
    parser.add_argument('--setup', action='store_true', help='Set up deployment environment')
    parser.add_argument('--find-candidates', action='store_true', help='Find best model candidates')
    parser.add_argument('--promote', action='store_true', help='Promote a model to production')
    parser.add_argument('--rollback', action='store_true', help='Rollback to previous version')
    parser.add_argument('--strategy', choices=['production', 'shadow', 'canary', 'ab_test'], 
                        help='Set deployment strategy')
    
    # Model specification
    parser.add_argument('--model-id', help='Specific model ID to deploy')
    parser.add_argument('--model-name', help='Model name to filter by')
    
    # Deployment configuration
    parser.add_argument('--traffic', type=float, help='Traffic percentage for canary/shadow/ab_test')
    parser.add_argument('--min-performance', type=float, help='Minimum performance metric for candidates')
    parser.add_argument('--force-rebuild', action='store_true', help='Force container rebuild')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup deployment environment
    if args.setup:
        setup_deployment_environment(args.force_rebuild)
    
    # Find best model candidates
    if args.find_candidates:
        candidates = find_best_model_candidates(args.model_name, args.min_performance)
        if candidates:
            print("\nBest model candidates:")
            for i, model in enumerate(candidates):
                print(f"{i+1}. {model['name']} v{model['version']} - " +
                      f"{model['metric_name']}: {model['metric_value']:.4f} " +
                      f"(Created: {model['created_at']})")
            print(f"\nFound {len(candidates)} candidates")
        else:
            print("No model candidates found matching criteria")
    
    # Promote to production
    if args.promote:
        if not args.model_id:
            # If no model ID provided, use best candidate
            candidates = find_best_model_candidates(args.model_name, args.min_performance)
            if not candidates:
                logger.error("No suitable candidates found for promotion")
                return
            args.model_id = candidates[0]['model_id']
            logger.info(f"Selected best candidate for promotion: {candidates[0]['name']} " +
                       f"v{candidates[0]['version']} with {candidates[0]['metric_name']} " +
                       f"of {candidates[0]['metric_value']:.4f}")
        
        promote_model_to_production(args.model_id, args.strategy or 'production')
    
    # Rollback to previous version
    if args.rollback:
        rollback_to_previous_version(args.model_name)
    
    # Set deployment strategy
    if args.strategy and not args.promote:
        traffic = args.traffic if args.traffic is not None else None
        configure_deployment_strategy(args.strategy, traffic)

if __name__ == '__main__':
    main() 