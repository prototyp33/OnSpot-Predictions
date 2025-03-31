#!/usr/bin/env python
"""
Script to check the constraints in the Supabase database tables.
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

# Connect to PostgreSQL database directly to inspect schema
try:
    # Connect to Supabase
    print(f'Connecting to Supabase...')
    supabase = create_client(url, key)
    
    # Function to check enum constraints in a table
    def check_enum_constraints(table_name):
        print(f"\nChecking constraints for table '{table_name}'...")
        
        # This is a workaround to get constraints since the Supabase API doesn't directly expose schema info
        # We'll create a minimal record with invalid data to trigger constraint errors and analyze them
        
        # Try different status values for common status fields
        status_values = ["active", "inactive", "pending", "approved", "rejected", 
                        "open", "closed", "in_progress", "completed", "failed",
                        "green", "yellow", "red", "critical", "warning", "healthy",
                        "good", "bad", "ok", "error", "success", "neutral"]
        
        results = []
        
        for status in status_values:
            # Create a minimal test record
            test_record = {"name": "test_constraint", "status": status}
            
            # Add other required fields based on table
            if table_name == "business_sla":
                test_record.update({
                    "target": 99.0,
                    "actual": 98.0,
                    "financial_impact": "Low",
                })
            elif table_name == "risk_assessment":
                test_record.update({
                    "impact": "Low",
                    "likelihood": "Low",
                    "financial_impact": "Low",
                    "customer_impact": "Low",
                    "mitigation_status": "Planned",
                    "description": "Test description"
                })
            elif table_name == "kpi_data":
                test_record.update({
                    "value": "100",
                    "trend": "up",
                    "trend_value": "10%",
                    "description": "Test KPI",
                    "previous_value": "90"
                })
            
            try:
                # Try to insert the record
                result = supabase.table(table_name).insert(test_record).execute()
                
                # If it succeeds, delete it
                if result.data and len(result.data) > 0:
                    id_to_delete = result.data[0].get("id")
                    if id_to_delete:
                        supabase.table(table_name).delete().eq("id", id_to_delete).execute()
                    
                    print(f"  ✅ Status value '{status}' is allowed")
                    results.append({"status": status, "allowed": True})
            except Exception as e:
                error_str = str(e)
                constraint_error = "violates check constraint" in error_str
                
                if constraint_error:
                    print(f"  ❌ Status value '{status}' is not allowed")
                    results.append({"status": status, "allowed": False})
                else:
                    # If it's not a constraint error, print the actual error
                    print(f"  ❗ Error testing '{status}': {error_str}")
        
        allowed_values = [item["status"] for item in results if item["allowed"]]
        if allowed_values:
            print(f"\nAllowed status values for '{table_name}':")
            print(", ".join(allowed_values))
        else:
            print(f"\nNo allowed status values found for '{table_name}'")
            
        return allowed_values
    
    # Check constraints for problematic tables
    business_sla_values = check_enum_constraints("business_sla")
    risk_assessment_values = check_enum_constraints("risk_assessment")
    kpi_data_values = check_enum_constraints("kpi_data")
    
    print("\n=== SUMMARY OF ALLOWED STATUS VALUES ===")
    print(f"business_sla: {', '.join(business_sla_values) if business_sla_values else 'None'}")
    print(f"risk_assessment: {', '.join(risk_assessment_values) if risk_assessment_values else 'None'}")
    print(f"kpi_data: {', '.join(kpi_data_values) if kpi_data_values else 'None'}")
    
except Exception as e:
    print(f"Error checking constraints: {str(e)}") 