#!/usr/bin/env python
"""
Apply database migrations script.
This script will apply SQL migrations to the connected database.
"""

import os
import sys
import logging
import psycopg2
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def apply_migration(connection_string: str, migration_file: str) -> bool:
    """
    Apply a single SQL migration file to the database.
    
    Args:
        connection_string: Database connection string
        migration_file: Path to the SQL migration file
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Applying migration from file: {migration_file}")
    
    try:
        # Read the SQL file
        with open(migration_file, 'r') as f:
            sql = f.read()
            
        # Connect to the database
        with psycopg2.connect(connection_string) as conn:
            with conn.cursor() as cur:
                # Execute the SQL
                cur.execute(sql)
                
            # Commit the transaction
            conn.commit()
            
        logger.info(f"Migration applied successfully: {migration_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply migration {migration_file}: {e}")
        return False
        
def apply_migrations(connection_string: str, migrations_dir: str) -> None:
    """
    Apply all SQL migrations in the specified directory.
    
    Args:
        connection_string: Database connection string
        migrations_dir: Directory containing SQL migration files
    """
    migrations_path = Path(migrations_dir)
    
    if not migrations_path.exists() or not migrations_path.is_dir():
        logger.error(f"Migrations directory does not exist: {migrations_dir}")
        return
        
    # Get all .sql files in the directory
    migration_files = sorted([f for f in migrations_path.glob('*.sql')])
    
    if not migration_files:
        logger.warning(f"No migration files found in {migrations_dir}")
        return
        
    logger.info(f"Found {len(migration_files)} migration files")
    
    # Apply each migration
    for migration_file in migration_files:
        success = apply_migration(connection_string, str(migration_file))
        if not success:
            logger.error(f"Failed to apply migration {migration_file}. Stopping.")
            break
            
    logger.info("Migration process completed")

def main():
    """Main function to parse arguments and apply migrations."""
    parser = argparse.ArgumentParser(description='Apply database migrations')
    parser.add_argument(
        '--connection-string', 
        type=str, 
        default=os.getenv('DATABASE_URL'),
        help='Database connection string (default: DATABASE_URL env var)'
    )
    parser.add_argument(
        '--migrations-dir', 
        type=str, 
        default='sql',
        help='Directory containing SQL migration files (default: sql/)'
    )
    parser.add_argument(
        '--file', 
        type=str, 
        help='Apply a specific migration file instead of all files in the directory'
    )
    
    args = parser.parse_args()
    
    if not args.connection_string:
        logger.error("No database connection string provided.")
        logger.error("Please set the DATABASE_URL environment variable or use --connection-string.")
        sys.exit(1)
        
    if args.file:
        # Apply a single migration file
        success = apply_migration(args.connection_string, args.file)
        if not success:
            sys.exit(1)
    else:
        # Apply all migrations in the directory
        apply_migrations(args.connection_string, args.migrations_dir)
        
if __name__ == "__main__":
    main() 