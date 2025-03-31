#!/usr/bin/env python
"""
Script to check Supabase tables and find those with 0 rows.
"""

import os
from dotenv import load_dotenv
from supabase import create_client
import sys

# Load environment variables
load_dotenv()

# Get Supabase credentials from environment variables
url = os.getenv('SUPABASE_URL', 'https://xdocqtlzgertsrmbocyt.supabase.co')
key = os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_KEY') or 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhkb2NxdGx6Z2VydHNybWJvY3l0Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczOTY0NTg4OSwiZXhwIjoyMDU1MjIxODg5fQ.RbhThoq-eBWNJWzILz9vPNnsNeEVIvHjl2aKkoUdeDM'

if not url or not key:
    print('Error: Missing Supabase credentials in environment variables')
    sys.exit(1)

try:
    # Connect to Supabase
    print(f'Connecting to Supabase...')
    supabase = create_client(url, key)
    
    # Get list of tables from Supabase
    print('Fetching table names...')
    
    # List of tables based on the schema we've seen from screenshots
    tables = [
        'business_sla',
        'business_indicators',
        'business_metrics_time_series',
        'risk_assessment',
        'financial_data',
        'financial_time_series',
        'kpi_data',
        'drift_analysis',
        'metrics',
        'predictions'
    ]
    
    print(f'Checking {len(tables)} tables for row counts...')
    
    # Check row count for each table
    empty_tables = []
    non_empty_tables = []
    error_tables = []
    
    for table in tables:
        try:
            result = supabase.table(table).select('count', count='exact').execute()
            count = result.count if hasattr(result, 'count') else len(result.data)
            
            if count == 0:
                empty_tables.append(table)
                print(f'Table {table}: 0 rows')
            else:
                non_empty_tables.append((table, count))
                print(f'Table {table}: {count} rows')
                
        except Exception as table_error:
            error_tables.append((table, str(table_error)))
            print(f'Error checking table {table}: {str(table_error)}')
    
    # Print summary
    print('\n=== SUMMARY ===')
    print(f'Total tables checked: {len(tables)}')
    print(f'Tables with data: {len(non_empty_tables)}')
    print(f'Empty tables: {len(empty_tables)}')
    print(f'Tables with errors: {len(error_tables)}')
    
    if non_empty_tables:
        print('\nTables with data:')
        for table, count in non_empty_tables:
            print(f'- {table}: {count} rows')
    
    if empty_tables:
        print('\nEmpty tables:')
        for table in empty_tables:
            print(f'- {table}')
            
    if error_tables:
        print('\nTables with errors:')
        for table, error in error_tables:
            print(f'- {table}: {error}')

except Exception as e:
    print(f'Error: {str(e)}') 