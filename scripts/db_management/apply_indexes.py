#!/usr/bin/env python
"""
Script to apply SQL indexes to Supabase.

This script:
1. Reads the SQL from create_indexes.sql
2. Executes each index creation statement
3. Reports on the success or failure of each operation
"""

import os
import sys
import logging
import time
from typing import List, Tuple
from dotenv import load_dotenv
from supabase import create_client
import httpx

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_supabase_client():
    """Get authenticated Supabase client."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        logger.error("Missing required environment variables: SUPABASE_URL or SUPABASE_KEY/SUPABASE_SERVICE_KEY")
        sys.exit(1)
        
    return create_client(url, key)

def parse_sql_file(file_path: str) -> List[str]:
    """
    Parse SQL file into individual statements.
    
    Args:
        file_path: Path to SQL file
    
    Returns:
        List of SQL statements
    """
    if not os.path.exists(file_path):
        logger.error(f"SQL file not found: {file_path}")
        sys.exit(1)
        
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split the SQL by semicolons
    statements = []
    for statement in content.split(';'):
        # Clean up the statement
        clean_statement = statement.strip()
        
        # Skip empty statements and comments-only statements
        if not clean_statement or clean_statement.startswith('--') or clean_statement.startswith('/*'):
            continue
            
        statements.append(clean_statement)
    
    return statements

def execute_index_statements(client, statements: List[str]) -> List[Tuple[str, bool, str]]:
    """
    Execute each index creation statement.
    
    Args:
        client: Supabase client
        statements: List of SQL statements to execute
        
    Returns:
        List of tuples with (statement, success, message)
    """
    results = []
    
    for i, statement in enumerate(statements, 1):
        # Extract index name for logging
        index_name = "unknown"
        if "idx_" in statement:
            try:
                index_name = statement.split("idx_")[1].split(" ")[0]
            except:
                pass
                
        logger.info(f"[{i}/{len(statements)}] Creating index: {index_name}")
        
        try:
            # For each SQL statement, make a 'rpc' call to execute it
            response = client.rpc(
                'execute_sql',
                {'sql': statement}
            ).execute()
            
            # Log response for debugging
            if hasattr(response, 'error') and response.error:
                logger.error(f"Error creating index {index_name}: {response.error}")
                results.append((statement, False, str(response.error)))
            else:
                logger.info(f"Successfully created index {index_name}")
                results.append((statement, True, "Success"))
                
            # Add a small delay to avoid overwhelming the database
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Exception creating index {index_name}: {str(e)}")
            results.append((statement, False, str(e)))
            
            # Add a longer delay after errors
            time.sleep(2)
    
    return results

def fallback_execute_indexes(sql_path: str):
    """
    Fallback method to suggest using the SQL script directly via SQL console.
    
    Args:
        sql_path: Path to the SQL file
    """
    logger.warning("Direct SQL execution not available through Supabase Python client.")
    logger.warning("Please run the SQL script directly via the Supabase SQL console:")
    logger.warning(f"1. Open your Supabase project dashboard")
    logger.warning(f"2. Go to the SQL Editor")
    logger.warning(f"3. Create a new query and paste the contents of {sql_path}")
    logger.warning(f"4. Execute the query")
    
    # Print the SQL file contents for convenience
    try:
        with open(sql_path, 'r') as f:
            logger.info("SQL script contents:")
            for line in f:
                print(line.rstrip())
    except Exception as e:
        logger.error(f"Error reading SQL file: {str(e)}")

def main():
    """Main function."""
    sql_file = "create_indexes.sql"
    
    logger.info(f"Reading SQL from {sql_file}")
    statements = parse_sql_file(sql_file)
    logger.info(f"Found {len(statements)} SQL statements to execute")
    
    # Get Supabase client
    client = get_supabase_client()
    
    # First, check if we have direct SQL execution capability
    try:
        # Try to execute a simple SQL statement to check if the RPC is available
        test_response = client.rpc(
            'execute_sql', 
            {'sql': 'SELECT current_timestamp'}
        ).execute()
        
        # If we get here without an error, we can execute SQL
        logger.info("SQL execution capability confirmed")
        
        # Execute all statements
        results = execute_index_statements(client, statements)
        
        # Report results
        success_count = sum(1 for _, success, _ in results if success)
        logger.info(f"Execution complete: {success_count} of {len(results)} statements successful")
        
        # Log failures
        failures = [(i+1, stmt, msg) for i, (stmt, success, msg) in enumerate(results) if not success]
        if failures:
            logger.warning(f"Failed to execute {len(failures)} statements:")
            for i, stmt, msg in failures:
                logger.warning(f"  [{i}] {msg}")
                logger.warning(f"  SQL: {stmt}")
                
    except Exception as e:
        logger.error(f"SQL execution not available: {str(e)}")
        logger.warning("Falling back to manual execution...")
        fallback_execute_indexes(sql_file)

if __name__ == "__main__":
    main() 