#!/usr/bin/env python
"""
Application Code Update Script

This script updates Python application code to use the new schema-based
table references after migrating tables from the public schema to dedicated
schemas in the OnSpot Predictive Model database.
"""

import os
import re
import glob
from pathlib import Path
import argparse

# Schema mapping
TABLE_TO_SCHEMA = {
    # Core schema
    'models': 'core',
    'predictions': 'core',
    'raw_parking_data': 'core',
    'cleaned_parking_data': 'core',
    'feature_engineered_data': 'core',
    
    # Monitoring schema
    'drift_analysis': 'monitoring',
    'retraining_events': 'monitoring',
    'system_health': 'monitoring',
    
    # Analytics schema
    'business_metrics': 'analytics',
    'location_metrics': 'analytics',
    
    # Experimentation schema
    'ab_tests': 'experimentation',
    'test_variants': 'experimentation',
    'test_daily_metrics': 'experimentation',
    
    # Auth schema
    'users': 'auth',
    'user_roles': 'auth'
}

def update_supabase_queries_in_file(file_path, dry_run=True):
    """
    Update Supabase query references in a Python file.
    
    Args:
        file_path: Path to the Python file
        dry_run: If True, print changes but don't modify the file
        
    Returns:
        Tuple of (number of changes, updated content)
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    num_changes = 0
    
    # Pattern for Supabase table references
    # This looks for table('table_name') pattern
    pattern = r"table\(['\"]([a-zA-Z_]+)['\"]\)"
    
    # Find all matches
    matches = re.findall(pattern, content)
    
    for table_name in matches:
        if table_name in TABLE_TO_SCHEMA:
            schema = TABLE_TO_SCHEMA[table_name]
            
            # Replace table('table_name') with from_('schema.table_name')
            old_pattern = f"table('{table_name}')"
            new_pattern = f"from_('{schema}.{table_name}')"
            
            old_pattern_double_quotes = f'table("{table_name}")'
            new_pattern_double_quotes = f'from_("{schema}.{table_name}")'
            
            if old_pattern in content:
                content = content.replace(old_pattern, new_pattern)
                num_changes += 1
            
            if old_pattern_double_quotes in content:
                content = content.replace(old_pattern_double_quotes, new_pattern_double_quotes)
                num_changes += 1
    
    # Handle older style syntax: supabase.from('table_name')
    pattern_from = r"from\(['\"]([a-zA-Z_]+)['\"]\)"
    matches_from = re.findall(pattern_from, content)
    
    for table_name in matches_from:
        if table_name in TABLE_TO_SCHEMA:
            schema = TABLE_TO_SCHEMA[table_name]
            
            # Replace from('table_name') with from('schema.table_name')
            old_pattern = f"from('{table_name}')"
            new_pattern = f"from('{schema}.{table_name}')"
            
            old_pattern_double_quotes = f'from("{table_name}")'
            new_pattern_double_quotes = f'from("{schema}.{table_name}")'
            
            if old_pattern in content:
                content = content.replace(old_pattern, new_pattern)
                num_changes += 1
            
            if old_pattern_double_quotes in content:
                content = content.replace(old_pattern_double_quotes, new_pattern_double_quotes)
                num_changes += 1
    
    # Only write if there were changes and not in dry run mode
    if num_changes > 0 and not dry_run:
        with open(file_path, 'w') as f:
            f.write(content)
    
    return num_changes, content

def update_sql_queries_in_file(file_path, dry_run=True):
    """
    Update SQL query references in a Python file.
    
    Args:
        file_path: Path to the Python file
        dry_run: If True, print changes but don't modify the file
        
    Returns:
        Tuple of (number of changes, updated content)
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    num_changes = 0
    
    # This handles SQL queries like "SELECT * FROM table_name"
    # and "INSERT INTO table_name"
    for table_name, schema in TABLE_TO_SCHEMA.items():
        # Match FROM table_name pattern
        from_pattern = re.compile(rf'FROM\s+{table_name}\b', re.IGNORECASE)
        content = from_pattern.sub(f'FROM {schema}.{table_name}', content)
        
        # Match JOIN table_name pattern
        join_pattern = re.compile(rf'JOIN\s+{table_name}\b', re.IGNORECASE)
        content = join_pattern.sub(f'JOIN {schema}.{table_name}', content)
        
        # Match INSERT INTO table_name pattern
        insert_pattern = re.compile(rf'INSERT\s+INTO\s+{table_name}\b', re.IGNORECASE)
        content = insert_pattern.sub(f'INSERT INTO {schema}.{table_name}', content)
        
        # Match UPDATE table_name pattern
        update_pattern = re.compile(rf'UPDATE\s+{table_name}\b', re.IGNORECASE)
        content = update_pattern.sub(f'UPDATE {schema}.{table_name}', content)
        
        # Match DELETE FROM table_name pattern
        delete_pattern = re.compile(rf'DELETE\s+FROM\s+{table_name}\b', re.IGNORECASE)
        content = delete_pattern.sub(f'DELETE FROM {schema}.{table_name}', content)
    
    # Count changes by comparing content length
    if content != original_content:
        num_changes = 1  # Just indicate there were changes
    
    # Only write if there were changes and not in dry run mode
    if num_changes > 0 and not dry_run:
        with open(file_path, 'w') as f:
            f.write(content)
    
    return num_changes, content

def scan_and_update_directory(directory, extensions=['.py'], dry_run=True, verbose=False):
    """
    Scan a directory and update all Python files with Supabase queries.
    
    Args:
        directory: Directory to scan
        extensions: File extensions to process
        dry_run: If True, print changes but don't modify files
        verbose: If True, print more detailed information
        
    Returns:
        Dictionary with statistics about updates
    """
    stats = {
        'files_scanned': 0,
        'files_updated': 0,
        'total_changes': 0
    }
    
    print(f"Scanning directory: {directory}")
    
    # Find all Python files in the directory and subdirectories
    for ext in extensions:
        files = glob.glob(f"{directory}/**/*{ext}", recursive=True)
        
        for file_path in files:
            try:
                stats['files_scanned'] += 1
                
                if verbose:
                    print(f"Scanning {file_path}...")
                
                # Update Supabase queries
                supabase_changes, _ = update_supabase_queries_in_file(file_path, dry_run)
                
                # Update SQL queries
                sql_changes, _ = update_sql_queries_in_file(file_path, dry_run)
                
                changes = supabase_changes + (1 if sql_changes > 0 else 0)
                
                if changes > 0:
                    stats['files_updated'] += 1
                    stats['total_changes'] += changes
                    
                    action = "Would update" if dry_run else "Updated"
                    print(f"{action} {file_path} ({changes} changes)")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    return stats

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Update application code after schema migration")
    parser.add_argument("--directory", default=".", help="Directory to scan")
    parser.add_argument("--extensions", default=".py", help="Comma-separated file extensions to process")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without modifying files")
    parser.add_argument("--verbose", action="store_true", help="Print more detailed information")
    
    args = parser.parse_args()
    
    # Parse extensions
    extensions = [ext.strip() if ext.strip().startswith('.') else f".{ext.strip()}" 
                  for ext in args.extensions.split(',')]
    
    print("=== OnSpot Predictive Model - Application Code Update ===")
    print(f"Directory: {args.directory}")
    print(f"Extensions: {', '.join(extensions)}")
    print(f"Dry run: {args.dry_run}")
    print(f"Verbose: {args.verbose}")
    print("=" * 60)
    
    # Scan and update directory
    stats = scan_and_update_directory(
        args.directory, 
        extensions=extensions,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Files scanned: {stats['files_scanned']}")
    print(f"Files updated: {stats['files_updated']}")
    print(f"Total changes: {stats['total_changes']}")
    
    if args.dry_run and stats['total_changes'] > 0:
        print("\nRun without --dry-run to apply these changes.")

if __name__ == "__main__":
    main() 