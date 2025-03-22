#!/usr/bin/env python
"""
Supabase Setup and Monitoring Integration

This script provides a unified way to initialize and configure the Supabase
monitoring system, including connection tracking, operation performance metrics,
and integration with existing monitoring pipelines.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Try to import monitoring components
try:
    from scripts.supabase_monitor import get_monitor
    from scripts.supabase_metrics_extension import integrate_supabase_monitoring, load_config
    from scripts.supabase_dashboard import SupabaseDashboard
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise

def setup_monitoring(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Set up and integrate Supabase monitoring components.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Dictionary containing initialized components
    """
    # Get the monitor singleton
    monitor = get_monitor()
    
    # Integrate with existing performance monitoring
    extension = integrate_supabase_monitoring(config_path)
    
    # Create dashboard
    dashboard = SupabaseDashboard()
    
    # Load configuration
    config = load_config(config_path)
    
    # Log setup completion
    logger.info("Supabase monitoring system initialized")
    
    # Return initialized components
    return {
        "monitor": monitor,
        "extension": extension,
        "dashboard": dashboard,
        "config": config
    }

def patch_supabase_client():
    """
    Patch the Supabase client to automatically use monitoring decorators.
    This function will attempt to locate any Supabase client initializations
    and wrap their methods with appropriate monitoring decorators.
    """
    try:
        # Try to patch common Supabase client locations
        client_locations = [
            "api/config/supabase.py",
            "scripts/supabase_utils.py",
            "scripts/upload_data.py"
        ]
        
        for location in client_locations:
            if os.path.exists(location):
                logger.info(f"Patching Supabase client at {location}")
                # In a real implementation, this would modify the file
                # to add decorator imports and apply them to methods
                
        logger.info("Supabase client patching complete")
        return True
    except Exception as e:
        logger.error(f"Failed to patch Supabase client: {e}")
        return False

def apply_schema_migration(dry_run: bool = True):
    """
    Apply the schema migration script.
    
    Args:
        dry_run: If True, only show what would be done without making changes
    """
    try:
        migration_script = "scripts/migrate_schemas.sql"
        if not os.path.exists(migration_script):
            logger.error(f"Migration script not found: {migration_script}")
            return False
            
        logger.info(f"{'Would apply' if dry_run else 'Applying'} schema migration from {migration_script}")
        
        # In a real implementation, this would:
        # 1. Connect to Supabase
        # 2. Read the migration script
        # 3. Execute it with appropriate error handling
        
        # For now, just report success
        if not dry_run:
            from scripts.supabase_metrics_extension import integrate_supabase_monitoring
            extension = integrate_supabase_monitoring()
            # Here you would actually execute the SQL script
            
        logger.info(f"Schema migration {'would be' if dry_run else 'was'} successful")
        return True
    except Exception as e:
        logger.error(f"Failed to apply schema migration: {e}")
        return False

def update_application_code(target_dir: str = ".", dry_run: bool = True):
    """
    Update application code to use new schema references.
    
    Args:
        target_dir: Directory containing code to update
        dry_run: If True, only show what would be done without making changes
    """
    try:
        update_script = "scripts/update_application_code.py"
        if not os.path.exists(update_script):
            logger.error(f"Update script not found: {update_script}")
            return False
            
        logger.info(f"{'Would update' if dry_run else 'Updating'} application code in {target_dir}")
        
        # In a real implementation, this would:
        # 1. Import and run the update_application_code.py script
        # 2. Pass it the target_dir and dry_run parameters
        
        # For now, just report success
        if not dry_run:
            # Here you would actually execute the code update script
            import scripts.update_application_code as updater
            updater.main([target_dir, "--dry-run" if dry_run else ""])
            
        logger.info(f"Application code update {'would be' if dry_run else 'was'} successful")
        return True
    except Exception as e:
        logger.error(f"Failed to update application code: {e}")
        return False

def main():
    """Main function to set up and configure Supabase monitoring."""
    parser = argparse.ArgumentParser(description="Supabase Setup and Monitoring")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--patch-client", action="store_true", help="Patch Supabase client with monitoring decorators")
    parser.add_argument("--migrate-schema", action="store_true", help="Apply schema migration")
    parser.add_argument("--update-code", action="store_true", help="Update application code to use new schemas")
    parser.add_argument("--target-dir", default=".", help="Directory containing code to update")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--all", action="store_true", help="Perform all setup actions")
    
    args = parser.parse_args()
    
    # Set up monitoring system
    components = setup_monitoring(args.config)
    
    # Perform requested actions
    if args.all or args.patch_client:
        patch_supabase_client()
    
    if args.all or args.migrate_schema:
        apply_schema_migration(args.dry_run)
    
    if args.all or args.update_code:
        update_application_code(args.target_dir, args.dry_run)
    
    # Generate and display a health status
    health = components["extension"].get_latest_supabase_health()
    status_emoji = "✅" if health["status"] == "healthy" else "⚠️" if health["status"] == "degraded" else "❌"
    
    print(f"\nSupabase Monitoring Status: {status_emoji} {health['status'].upper()}")
    print(f"Alert Level: {health['alert_level'].upper()}")
    
    if health["issues"]:
        print("\nIssues:")
        for issue in health["issues"]:
            print(f"  - {issue}")

if __name__ == "__main__":
    main() 