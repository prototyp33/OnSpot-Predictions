#!/usr/bin/env python
"""
Script to inspect the schema of empty tables in Supabase.
"""

import os
from dotenv import load_dotenv
from supabase import create_client
import json

# Load environment variables
load_dotenv()

# Get Supabase credentials
url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_KEY')

if not url or not key:
    print('Error: Missing Supabase credentials in environment variables')
    exit(1)

# Connect to Supabase
print(f'Connecting to Supabase...')
supabase = create_client(url, key)

# List of empty tables we want to inspect
empty_tables = [
    'business_metrics',
    'location_metrics',
    'system_health',
    'ab_tests',
    'test_daily_metrics',
    'test_variants',
]

for table in empty_tables:
    print(f"\n=== Table: {table} ===")
    
    # Get schema information using PostgreSQL information_schema
    schema_query = f"""
    SELECT 
        column_name, 
        data_type, 
        is_nullable,
        column_default
    FROM 
        information_schema.columns 
    WHERE 
        table_schema = 'public' 
        AND table_name = '{table}'
    ORDER BY 
        ordinal_position;
    """
    
    try:
        result = supabase.rpc('postgres', params={'query': schema_query}).execute()
        
        if hasattr(result, 'data') and result.data:
            # Print column details in a readable format
            print("Column Name".ljust(25) + "Data Type".ljust(20) + "Nullable".ljust(10) + "Default")
            print("-" * 80)
            for column in result.data:
                col_name = column['column_name'].ljust(25)
                data_type = column['data_type'].ljust(20)
                nullable = column['is_nullable'].ljust(10)
                default = str(column['column_default'] or 'NULL')
                print(f"{col_name}{data_type}{nullable}{default}")
        else:
            # Alternative approach for retrieving schema info
            print(f"Could not retrieve schema for table {table} using RPC method.")
            print("Checking a single row structure instead...")
            
            # Try to query a non-existent row to see structure
            result = supabase.table(table).select('*').limit(1).execute()
            if hasattr(result, 'data'):
                # Get column names from returned data structure
                print("Available columns (names only):")
                if not result.data:
                    # If table is truly empty, this is expected
                    print("Table is empty. Let's check metadata...")
                    
                    # This is a workaround to get column names without data
                    # Not always reliable but worth trying
                    fields = getattr(result, 'fields', None)
                    if fields:
                        for field in fields:
                            print(f"- {field}")
                    else:
                        print("Could not determine schema structure.")
    except Exception as e:
        print(f"Error inspecting table {table}: {str(e)}")

print("\nSchema inspection complete") 