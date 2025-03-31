#!/usr/bin/env python3
"""
Model Migration Script for OnSpot Predictive Model.

This script migrates existing production models to the new model registry system.
It scans the production_models directory, loads each model and its metadata,
and registers them with the new model registry.
"""

import os
import sys
import json
import logging
import argparse
import joblib
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_migration')

# Add the project root to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)

# Import the model registry
from scripts.models.integration import ModelIntegration

def list_production_models(production_models_path: str) -> List[str]:
    """
    List all production model directories.
    
    Args:
        production_models_path: Path to production models directory
        
    Returns:
        List of model directory paths
    """
    if not os.path.exists(production_models_path):
        logger.warning(f"Production models path {production_models_path} does not exist")
        return []
    
    model_dirs = []
    
    for item in os.listdir(production_models_path):
        item_path = os.path.join(production_models_path, item)
        
        # Skip symlinks
        if os.path.islink(item_path):
            continue
        
        # Only include directories
        if os.path.isdir(item_path):
            model_dirs.append(item_path)
    
    logger.info(f"Found {len(model_dirs)} production model directories")
    return model_dirs

def parse_model_name_from_directory(directory: str) -> str:
    """
    Parse model name from directory name.
    
    Args:
        directory: Directory path
        
    Returns:
        Model name
    """
    # Assume format is model_name_YYYYMMDD_HHMMSS
    base_name = os.path.basename(directory)
    
    # Try to split by timestamp pattern
    parts = base_name.split('_')
    
    # If there are at least 3 parts and the second-to-last part looks like a date
    if len(parts) >= 3 and len(parts[-2]) == 8 and parts[-2].isdigit():
        # Everything before the date is the model name
        return '_'.join(parts[:-2])
    
    # If no timestamp pattern found, use the whole name
    return base_name

def migrate_model(
    model_path: str, 
    integration: ModelIntegration,
    model_type_map: Dict[str, str] = None
) -> str:
    """
    Migrate a single model to the new registry.
    
    Args:
        model_path: Path to the model directory
        integration: ModelIntegration instance
        model_type_map: Map of model names to model types
        
    Returns:
        Model ID of the migrated model
    """
    try:
        model_name = parse_model_name_from_directory(model_path)
        
        # Determine model type
        model_type = "unknown"
        if model_type_map and model_name in model_type_map:
            model_type = model_type_map[model_name]
        
        # Migrate the model
        model_id = integration.migrate_existing_model(
            model_path=model_path,
            model_name=model_name,
            model_type=model_type,
            change_type="minor"
        )
        
        logger.info(f"Successfully migrated model {model_name} from {model_path} with ID {model_id}")
        return model_id
    
    except Exception as e:
        logger.error(f"Error migrating model {model_path}: {e}")
        return None

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Migrate existing models to the new registry')
    
    parser.add_argument(
        '--production-models',
        type=str,
        default='production_models',
        help='Path to production models directory'
    )
    
    parser.add_argument(
        '--registry-path',
        type=str,
        default='model_registry',
        help='Path to model registry directory'
    )
    
    parser.add_argument(
        '--model-type-map',
        type=str,
        help='Path to JSON file mapping model names to model types'
    )
    
    parser.add_argument(
        '--promote',
        action='store_true',
        help='Promote the latest version of each model to production'
    )
    
    return parser.parse_args()

def main():
    """Main function to migrate models."""
    args = parse_arguments()
    
    # Create the integration instance
    integration = ModelIntegration(
        registry_path=args.registry_path,
        production_models_path=args.production_models
    )
    
    # Load model type map if provided
    model_type_map = None
    if args.model_type_map:
        try:
            with open(args.model_type_map, 'r') as f:
                model_type_map = json.load(f)
            logger.info(f"Loaded model type map with {len(model_type_map)} entries")
        except Exception as e:
            logger.warning(f"Could not load model type map: {e}")
    
    # List production models
    model_dirs = list_production_models(args.production_models)
    
    if not model_dirs:
        logger.info("No production models found to migrate")
        return
    
    # Track successful migrations by model name
    successful_migrations = {}
    
    # Migrate each model
    for model_dir in model_dirs:
        model_id = migrate_model(model_dir, integration, model_type_map)
        
        if model_id:
            model_name = parse_model_name_from_directory(model_dir)
            
            # Track this model (we'll keep the latest one for each name)
            if model_name not in successful_migrations:
                successful_migrations[model_name] = []
            
            successful_migrations[model_name].append({
                "model_id": model_id,
                "directory": model_dir,
                "timestamp": os.path.getmtime(model_dir)
            })
    
    # Find the latest model of each type
    latest_models = {}
    for model_name, migrations in successful_migrations.items():
        # Sort by timestamp (newest first)
        sorted_migrations = sorted(migrations, key=lambda x: x["timestamp"], reverse=True)
        if sorted_migrations:
            latest_models[model_name] = sorted_migrations[0]["model_id"]
    
    logger.info(f"Successfully migrated {len(latest_models)} unique models")
    
    # Promote the latest version of each model if requested
    if args.promote:
        logger.info("Promoting latest version of each model to production...")
        
        for model_name, model_id in latest_models.items():
            try:
                prod_dir = integration.versioning.promote_to_production(model_id)
                logger.info(f"Promoted model {model_name} ({model_id}) to production at {prod_dir}")
            except Exception as e:
                logger.error(f"Error promoting model {model_name} ({model_id}): {e}")
    
    logger.info("Migration complete")

if __name__ == "__main__":
    main() 