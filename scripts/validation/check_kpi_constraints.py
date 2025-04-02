#!/usr/bin/env python
"""
Script to check the allowed values for the trend field in the kpi_data table
"""

import os
import sys
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()

# Get Supabase credentials from environment variables
url = os.getenv('SUPABASE_URL', 'https://xdocqtlzgertsrmbocyt.supabase.co')
key = os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_KEY') or 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhkb2NxdGx6Z2VydHNybWJvY3l0Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczOTY0NTg4OSwiZXhwIjoyMDU1MjIxODg5fQ.RbhThoq-eBWNJWzILz9vPNnsNeEVIvHjl2aKkoUdeDM'

if not url or not key:
    print('Error: Missing Supabase credentials in environment variables')
    sys.exit(1)

# Connect to Supabase
print(f'Connecting to Supabase...')
supabase = create_client(url, key)

def check_trend_constraints():
    """Check the allowed values for the trend field in the kpi_data table"""
    print("Checking allowed values for the 'trend' field in the kpi_data table...")
    
    # Test various potential trend values
    test_values = ["up", "down", "stable", "increasing", "decreasing", "flat", "positive", "negative", "neutral"]
    allowed_values = []
    disallowed_values = []
    
    for trend in test_values:
        # Create test record
        test_record = {
            "name": f"Test KPI for trend '{trend}'",
            "value": "50.0",
            "trend": trend,
            "trend_value": "+0.5%",
            "status": "healthy",  # We know this is valid from previous tests
            "description": "Test description",
            "previous_value": "49.5"
        }
        
        try:
            # Try to insert the record
            result = supabase.table('kpi_data').insert(test_record).execute()
            
            # If successful, add to allowed values
            if result.data:
                allowed_values.append(trend)
                
                # Delete the test record
                id = result.data[0]['id']
                supabase.table('kpi_data').delete().eq('id', id).execute()
                
        except Exception as e:
            # If failed due to constraint violation, add to disallowed values
            if "violates check constraint" in str(e) and "kpi_data_trend_check" in str(e):
                disallowed_values.append(trend)
            else:
                print(f"Error testing '{trend}': {str(e)}")
    
    # Print results
    print(f"\nAllowed values for 'trend' field in kpi_data table:")
    if allowed_values:
        for value in allowed_values:
            print(f"- '{value}'")
    else:
        print("No allowed values found among those tested.")
    
    print(f"\nDisallowed values for 'trend' field in kpi_data table:")
    if disallowed_values:
        for value in disallowed_values:
            print(f"- '{value}'")
    else:
        print("No disallowed values found among those tested.")

if __name__ == "__main__":
    check_trend_constraints() 