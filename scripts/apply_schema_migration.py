#!/usr/bin/env python
"""
Schema Migration Application Script

This script applies the schema migration SQL to reorganize tables in
the OnSpot Predictive Model database.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
import supabase

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_sql_file(file_path):
    """Load SQL file contents."""
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error reading SQL file {file_path}: {str(e)}")
        sys.exit(1)

def apply_migration(client, sql_content):
    """
    Apply the migration SQL to Supabase.
    
    Args:
        client: Supabase client
        sql_content: SQL content to execute
    """
    try:
        # Execute the SQL
        logger.info("Starting schema migration...")
        client.query(sql_content).execute()
        logger.info("Schema migration SQL executed successfully")
        return True
    except Exception as e:
        logger.error(f"Error executing schema migration: {str(e)}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Apply schema migration to Supabase")
    parser.add_argument("--sql-file", default="scripts/migrate_schemas.sql", 
                      help="Path to the SQL migration file")
    parser.add_argument("--env-file", default=".env",
                      help="Path to the .env file with Supabase credentials")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv(args.env_file)
    
    # Check for required env vars
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        logger.error("SUPABASE_URL and SUPABASE_KEY must be set in the environment or .env file")
        sys.exit(1)
    
    # Check if SQL file exists
    sql_file_path = Path(args.sql_file)
    if not sql_file_path.exists():
        logger.error(f"SQL file not found: {sql_file_path}")
        sys.exit(1)
    
    # Load SQL file
    logger.info(f"Loading SQL file: {sql_file_path}")
    sql_content = load_sql_file(sql_file_path)
    
    # Initialize Supabase client
    try:
        client = supabase.create_client(supabase_url, supabase_key)
        logger.info("Connected to Supabase")
    except Exception as e:
        logger.error(f"Error connecting to Supabase: {str(e)}")
        sys.exit(1)
    
    # Apply migration
    if apply_migration(client, sql_content):
        logger.info("Schema migration completed successfully")
    else:
        logger.error("Schema migration failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 