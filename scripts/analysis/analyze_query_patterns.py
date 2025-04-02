#!/usr/bin/env python
"""
Script to analyze Supabase query patterns and recommend indexes.

This script:
1. Analyzes the schema of each table
2. Checks for existing indexes
3. Analyzes query patterns (by parsing code that accesses the DB)
4. Suggests indexes for performance improvement
"""

import os
import sys
import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Optional
import json
from dotenv import load_dotenv
from supabase import create_client

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Supabase client
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

def get_table_schema(table_name: str) -> Dict:
    """
    Get the schema information for a table.
    
    Args:
        table_name: Name of the table
        
    Returns:
        Dict with column information
    """
    try:
        # This is a RPC call to get table schema
        # Note: This might need to be adjusted based on Supabase's actual API
        response = supabase.rpc(
            'get_table_schema',
            {'table_name': table_name}
        ).execute()
        return response.data
    except Exception as e:
        logger.warning(f"Could not get schema for table {table_name}: {str(e)}")
        # Try an alternative approach - query the first row and get column names
        try:
            response = supabase.table(table_name).select("*").limit(1).execute()
            if response.data and len(response.data) > 0:
                return {"columns": list(response.data[0].keys())}
            return {"columns": []}
        except Exception as e2:
            logger.error(f"Failed to get schema for {table_name}: {str(e2)}")
            return {"columns": []}

def analyze_codebase_for_queries(directory: str = ".") -> Dict[str, List[str]]:
    """
    Analyze the codebase to find Supabase queries and identify potential index candidates.
    
    Args:
        directory: Root directory to scan
    
    Returns:
        Dict mapping table names to lists of columns used in WHERE clauses
    """
    logger.info(f"Analyzing codebase in {directory} for Supabase queries...")
    
    query_patterns = {}
    python_files = []
    
    # Find all Python files
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(python_files)} Python files to analyze")
    
    # Patterns to find Supabase queries
    table_pattern = r"\.table\(['\"]([a-zA-Z0-9_]+)['\"]\)"
    where_pattern = r"\.filter\(['\"]([a-zA-Z0-9_]+)['\"]"
    eq_pattern = r"\.eq\(['\"]([a-zA-Z0-9_]+)['\"]"
    
    # Alternative patterns for different query styles
    alt_where_pattern = r"where\(['\"]([a-zA-Z0-9_]+)['\"]"
    select_pattern = r"\.select\(['\"]([^'\"]+)['\"]\)"
    join_pattern = r"\.join\(['\"]([a-zA-Z0-9_]+)['\"].*?['\"]([a-zA-Z0-9_]+)['\"].*?['\"]([a-zA-Z0-9_]+)['\"]\)"
    
    # Analyze each file
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
                # Find table accesses
                table_matches = re.findall(table_pattern, content)
                
                for table in table_matches:
                    if table not in query_patterns:
                        query_patterns[table] = []
                    
                    # Look for where clauses near this table
                    # Find the position of the table reference
                    table_pos = content.find(f'.table("{table}")') 
                    if table_pos == -1:
                        table_pos = content.find(f".table('{table}')")
                    
                    if table_pos != -1:
                        # Extract a chunk of code following the table reference
                        chunk_size = 500  # Arbitrary size that should capture most query chains
                        chunk_end = min(len(content), table_pos + chunk_size)
                        code_chunk = content[table_pos:chunk_end]
                        
                        # Find where clauses and eq clauses
                        where_columns = re.findall(where_pattern, code_chunk)
                        eq_columns = re.findall(eq_pattern, code_chunk)
                        alt_where_columns = re.findall(alt_where_pattern, code_chunk)
                        
                        # Combine all columns used in filtering
                        filter_columns = where_columns + eq_columns + alt_where_columns
                        
                        # Add unique columns to the table's list
                        for col in filter_columns:
                            if col not in query_patterns[table]:
                                query_patterns[table].append(col)
        
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {str(e)}")
    
    return query_patterns

def get_existing_indexes() -> Dict[str, List[str]]:
    """
    Get information about existing indexes in the database.
    
    Returns:
        Dict mapping table names to lists of indexed columns
    """
    logger.info("Retrieving existing index information...")
    existing_indexes = {}
    
    try:
        # This would be the ideal way, but may not be available in Supabase client API
        # response = supabase.rpc('get_indexes').execute()
        # index_data = response.data
        
        # For now, we'll use a placeholder since direct index listing might not be available
        logger.warning("Direct index listing not implemented - would require custom SQL RPC function")
        return {}
        
    except Exception as e:
        logger.error(f"Failed to retrieve existing indexes: {str(e)}")
        return {}

def recommend_indexes(
    table_schemas: Dict[str, Dict],
    query_patterns: Dict[str, List[str]],
    existing_indexes: Dict[str, List[str]]
) -> Dict[str, List[Dict]]:
    """
    Recommend indexes based on schema and query patterns.
    
    Args:
        table_schemas: Dict mapping table names to their schemas
        query_patterns: Dict mapping table names to filtered columns
        existing_indexes: Dict mapping table names to existing indexes
        
    Returns:
        Dict mapping table names to recommended indexes
    """
    logger.info("Generating index recommendations...")
    recommendations = {}
    
    for table, columns in query_patterns.items():
        if table not in recommendations:
            recommendations[table] = []
        
        # Get existing indexes for this table
        table_indexes = existing_indexes.get(table, [])
        
        # Get schema for this table
        schema = table_schemas.get(table, {"columns": []})
        
        for column in columns:
            # Skip if this column is already indexed
            if column in table_indexes:
                continue
                
            # Skip if column doesn't exist in schema
            if "columns" in schema and schema["columns"] and column not in schema["columns"]:
                continue
                
            # Recommend an index
            recommendations[table].append({
                "column": column,
                "type": "btree",  # Default index type
                "reason": "Frequently used in queries"
            })
    
    # Add special recommendations
    for table, schema in table_schemas.items():
        # Check timestamp columns for time-series data
        if table in ["drift_analysis", "retraining_events", "business_metrics", 
                    "location_metrics", "system_health", "test_daily_metrics"]:
            columns = schema.get("columns", [])
            timestamp_cols = [col for col in columns if "timestamp" in col.lower() or "date" in col.lower()]
            
            for ts_col in timestamp_cols:
                # Skip if already recommended
                if any(rec["column"] == ts_col for rec in recommendations.get(table, [])):
                    continue
                
                recommendations.setdefault(table, []).append({
                    "column": ts_col,
                    "type": "btree",
                    "reason": "Time-series data frequently queried by date range"
                })
    
    return recommendations

def generate_index_sql(recommendations: Dict[str, List[Dict]]) -> str:
    """
    Generate SQL for creating the recommended indexes.
    
    Args:
        recommendations: Dict mapping table names to recommended indexes
        
    Returns:
        SQL string for creating all recommended indexes
    """
    sql_lines = [
        "-- SQL Script to create recommended indexes",
        "-- Generated by analyze_query_patterns.py",
        f"-- Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    
    for table, indexes in recommendations.items():
        if not indexes:
            continue
            
        sql_lines.append(f"-- Indexes for table: {table}")
        
        for idx, index_info in enumerate(indexes):
            column = index_info["column"]
            index_type = index_info["type"]
            reason = index_info["reason"]
            
            # Create index name
            index_name = f"idx_{table}_{column}"
            
            # Create SQL
            sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} USING {index_type} ({column});"
            
            # Add to SQL script with comment
            sql_lines.append(f"-- {reason}")
            sql_lines.append(sql)
            sql_lines.append("")
    
    return "\n".join(sql_lines)

def get_all_tables() -> List[str]:
    """
    Get a list of all tables in the database.
    
    Returns:
        List of table names
    """
    try:
        # We don't have a direct way to list tables in Supabase JS client
        # This is a workaround - define known tables
        known_tables = [
            "business_metrics", 
            "location_metrics", 
            "system_health",
            "ab_tests",
            "test_variants", 
            "test_daily_metrics", 
            "users",
            "user_roles",
            "drift_analysis",
            "retraining_events",
            "predictions",
            "models",
            "raw_parking_data",
            "cleaned_parking_data",
            "feature_engineered_data"
        ]
        return known_tables
    except Exception as e:
        logger.error(f"Failed to retrieve table list: {str(e)}")
        return []

def main():
    """Main function."""
    # Get all tables
    tables = get_all_tables()
    logger.info(f"Found {len(tables)} tables in database")
    
    # Get schema for each table
    table_schemas = {}
    for table in tables:
        schema = get_table_schema(table)
        table_schemas[table] = schema
        column_count = len(schema.get("columns", []))
        logger.info(f"Table {table}: {column_count} columns")
    
    # Analyze query patterns
    query_patterns = analyze_codebase_for_queries()
    logger.info(f"Found query patterns for {len(query_patterns)} tables")
    
    # Get existing indexes
    existing_indexes = get_existing_indexes()
    logger.info(f"Found existing indexes for {len(existing_indexes)} tables")
    
    # Generate recommendations
    recommendations = recommend_indexes(table_schemas, query_patterns, existing_indexes)
    logger.info(f"Generated index recommendations for {len(recommendations)} tables")
    
    # Print summary
    total_indexes = sum(len(indexes) for indexes in recommendations.values())
    logger.info(f"Recommending {total_indexes} new indexes")
    
    # Generate SQL
    sql = generate_index_sql(recommendations)
    
    # Write SQL to file
    sql_file = "create_indexes.sql"
    with open(sql_file, "w") as f:
        f.write(sql)
    logger.info(f"SQL script written to {sql_file}")
    
    # Print recommendations
    print("\nINDEX RECOMMENDATIONS\n")
    print(f"Total indexes recommended: {total_indexes}")
    print("\nRecommendations by table:")
    
    for table, indexes in recommendations.items():
        if not indexes:
            continue
        print(f"\n{table.upper()} ({len(indexes)} indexes):")
        for index in indexes:
            print(f"  - Column: {index['column']}")
            print(f"    Type: {index['type']}")
            print(f"    Reason: {index['reason']}")
    
    print(f"\nSQL script has been written to {sql_file}")

if __name__ == "__main__":
    main() 