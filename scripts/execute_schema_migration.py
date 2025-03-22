#!/usr/bin/env python
"""
Schema Migration Execution Script

This script executes the schema migration SQL file against the Supabase database
and reports on the progress and results.
"""

import os
import sys
import logging
import time
import argparse
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("schema_migration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extras import RealDictCursor
    DB_DEPS_AVAILABLE = True
except ImportError:
    DB_DEPS_AVAILABLE = False
    logger.warning("Database dependencies not available. Install with: pip install psycopg2-binary")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("dotenv not available. Install with: pip install python-dotenv")

# Import Supabase utilities if available
try:
    from scripts.supabase_utils import get_supabase_client
    SUPABASE_UTILS_AVAILABLE = True
except ImportError:
    SUPABASE_UTILS_AVAILABLE = False
    logger.warning("Supabase utils not available. Direct database connection will be used.")

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

class MigrationExecutor:
    """
    Executes database schema migrations.
    """
    
    def __init__(self, migration_file: str, dry_run: bool = False):
        """
        Initialize the migration executor.
        
        Args:
            migration_file: Path to migration SQL file
            dry_run: If True, only show what would be done without making changes
        """
        self.migration_file = migration_file
        self.dry_run = dry_run
        self.connection = None
        self.cursor = None
        self.statements = []
        self.results = []
        
        # Ensure migration file exists
        if not os.path.exists(migration_file):
            raise FileNotFoundError(f"Migration file not found: {migration_file}")
        
        logger.info(f"Migration executor initialized with file: {migration_file}")
        if dry_run:
            logger.info("DRY RUN MODE: No changes will be made to the database")
    
    def connect_to_db(self) -> bool:
        """
        Connect to the Supabase PostgreSQL database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Get database connection details from environment
            db_host = os.environ.get("SUPABASE_DB_HOST", "db.xxxxxx.supabase.co")
            db_name = os.environ.get("SUPABASE_DB_NAME", "postgres")
            db_user = os.environ.get("SUPABASE_DB_USER", "postgres")
            db_password = os.environ.get("SUPABASE_DB_PASSWORD", "")
            db_port = int(os.environ.get("SUPABASE_DB_PORT", "5432"))
            
            # Connect to database
            self.connection = psycopg2.connect(
                host=db_host,
                dbname=db_name,
                user=db_user,
                password=db_password,
                port=db_port
            )
            
            # Create cursor
            self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            logger.info(f"Connected to database: {db_name} at {db_host}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def parse_sql_file(self) -> List[str]:
        """
        Parse SQL file into individual statements.
        
        Returns:
            List of SQL statements
        """
        try:
            with open(self.migration_file, 'r') as f:
                sql_content = f.read()
            
            # Split on semicolons, but handle complex statements appropriately
            # This is a simple approach - a robust parser would be needed for more complex SQL
            sql_content = sql_content.strip()
            statements = []
            current_statement = ""
            
            # Simple state for tracking SQL blocks
            in_function_body = False
            
            for line in sql_content.split('\n'):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('--'):
                    continue
                
                # Track function/procedure body blocks
                if "$$" in line:
                    in_function_body = not in_function_body
                
                current_statement += line + " "
                
                # End of statement
                if line.endswith(';') and not in_function_body:
                    statements.append(current_statement.strip())
                    current_statement = ""
            
            # Handle any remaining statement
            if current_statement.strip():
                statements.append(current_statement.strip())
            
            self.statements = statements
            logger.info(f"Parsed {len(statements)} SQL statements from migration file")
            return statements
        except Exception as e:
            logger.error(f"Failed to parse SQL file: {e}")
            self.statements = []
            return []
    
    def execute_migration(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Execute the migration statements.
        
        Returns:
            (success, results) tuple
        """
        if not self.statements:
            self.parse_sql_file()
        
        if not self.statements:
            logger.error("No statements to execute")
            return False, []
        
        if not self.connection and not self.dry_run:
            if not self.connect_to_db():
                logger.error("Cannot execute migration without database connection")
                return False, []
        
        results = []
        success = True
        error_count = 0
        
        try:
            # Start transaction
            if not self.dry_run:
                self.connection.autocommit = False
            
            for i, statement in enumerate(self.statements):
                stmt_id = i + 1
                logger.info(f"Executing statement {stmt_id}/{len(self.statements)}")
                
                if self.dry_run:
                    logger.info(f"DRY RUN: Would execute: {statement[:100]}...")
                    results.append({
                        "id": stmt_id,
                        "status": "simulated",
                        "statement": statement[:100] + "..." if len(statement) > 100 else statement
                    })
                    continue
                
                try:
                    start_time = time.time()
                    self.cursor.execute(statement)
                    end_time = time.time()
                    
                    results.append({
                        "id": stmt_id,
                        "status": "success",
                        "duration": end_time - start_time,
                        "rows_affected": self.cursor.rowcount
                    })
                    
                    logger.info(f"Statement {stmt_id} executed successfully")
                except Exception as e:
                    error_count += 1
                    success = False
                    
                    results.append({
                        "id": stmt_id,
                        "status": "error",
                        "error": str(e),
                        "statement": statement[:100] + "..." if len(statement) > 100 else statement
                    })
                    
                    logger.error(f"Error executing statement {stmt_id}: {e}")
                    
                    # For significant errors, log the full statement
                    logger.debug(f"Failed statement: {statement}")
                    
                    # Stop on first error, unless explicitly continuing
                    break
            
            # Commit or rollback
            if not self.dry_run:
                if success:
                    self.connection.commit()
                    logger.info("Migration committed successfully")
                else:
                    self.connection.rollback()
                    logger.warning("Migration rolled back due to errors")
            
            self.results = results
            return success, results
        except Exception as e:
            logger.error(f"Error during migration execution: {e}")
            
            if not self.dry_run:
                try:
                    self.connection.rollback()
                    logger.warning("Migration rolled back due to errors")
                except:
                    pass
            
            return False, results
        finally:
            if self.connection and not self.dry_run:
                self.connection.autocommit = True
    
    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def print_summary(self):
        """Print a summary of migration results."""
        if not self.results:
            logger.warning("No migration results to summarize")
            return
        
        logger.info("\n" + "="*50)
        logger.info("MIGRATION SUMMARY")
        logger.info("="*50)
        
        success_count = sum(1 for r in self.results if r["status"] == "success")
        error_count = sum(1 for r in self.results if r["status"] == "error")
        simulated_count = sum(1 for r in self.results if r["status"] == "simulated")
        
        logger.info(f"Total statements:    {len(self.statements)}")
        logger.info(f"Executed statements: {len(self.results)}")
        logger.info(f"Successful:          {success_count}")
        logger.info(f"Failed:              {error_count}")
        logger.info(f"Simulated (dry run): {simulated_count}")
        
        if error_count > 0:
            logger.info("\nERRORS:")
            for r in self.results:
                if r["status"] == "error":
                    logger.info(f"  Statement {r['id']}: {r['error']}")
        
        logger.info("="*50)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Execute schema migration")
    parser.add_argument("--file", default="scripts/migrate_schemas.sql", help="Path to migration SQL file")
    parser.add_argument("--dry-run", action="store_true", help="Don't make any actual changes")
    parser.add_argument("--force", action="store_true", help="Force migration even if previous errors")
    
    args = parser.parse_args()
    
    if not DB_DEPS_AVAILABLE:
        logger.error("Database dependencies not available. Install with: pip install psycopg2-binary")
        return 1
    
    try:
        # Initialize migration executor
        executor = MigrationExecutor(args.file, args.dry_run)
        
        # Parse SQL file
        statements = executor.parse_sql_file()
        if not statements:
            logger.error("No SQL statements found in migration file")
            return 1
        
        # Execute migration
        success, results = executor.execute_migration()
        
        # Print summary
        executor.print_summary()
        
        if success:
            logger.info("Migration executed successfully")
            return 0
        else:
            logger.error("Migration failed with errors")
            return 1
    except Exception as e:
        logger.error(f"Error executing migration: {e}")
        return 1
    finally:
        if 'executor' in locals():
            executor.close()


if __name__ == "__main__":
    sys.exit(main()) 