#!/usr/bin/env python
"""
Script to inspect the actual structure of tables in Supabase
by test-inserting minimal data and observing results.
"""

import os
import sys
import uuid
import logging
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get Supabase credentials
url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_KEY')

if not url or not key:
    logger.error('Missing Supabase credentials in environment variables')
    sys.exit(1)

# Connect to Supabase
logger.info(f'Connecting to Supabase...')
supabase = create_client(url, key)

# Tables to test
tables = [
    "ab_tests",
    "test_variants",
    "test_daily_metrics",
    "users",
    "user_roles"
]

def test_insert_ab_tests():
    """Test inserting into ab_tests table."""
    test_data = {
        "id": str(uuid.uuid4()),
        "name": "test_ab_test",
        "description": "Test description",
        "start_date": datetime.now().date().isoformat()
        # Add minimal fields to see which ones are required
    }
    
    try:
        logger.info(f"Testing insert into ab_tests with minimal data")
        result = supabase.table("ab_tests").insert(test_data).execute()
        logger.info(f"Success! Required fields: {list(test_data.keys())}")
        
        # Try to get the record back to see all fields
        inserted_id = result.data[0]['id'] if hasattr(result, 'data') and result.data else test_data['id']
        get_result = supabase.table("ab_tests").select("*").eq("id", inserted_id).execute()
        
        if hasattr(get_result, 'data') and get_result.data:
            logger.info(f"All fields: {list(get_result.data[0].keys())}")
        
        # Clean up test data
        supabase.table("ab_tests").delete().eq("id", inserted_id).execute()
        return True
    except Exception as e:
        logger.error(f"Error testing ab_tests: {str(e)}")
        
        # Try different combinations if it fails
        if "Could not find" in str(e):
            field = str(e).split("'")[1] if "'" in str(e) else "unknown"
            logger.info(f"Removing field {field} and trying again")
            if field in test_data:
                del test_data[field]
                return test_insert_ab_tests()
        return False

def test_insert_test_variants():
    """Test inserting into test_variants table."""
    test_data = {
        "id": str(uuid.uuid4()),
        "name": "test_variant",
        "description": "Test variant description",
        "test_id": str(uuid.uuid4()),  # This would normally be a real test ID
        "traffic_percentage": 50.0
    }
    
    try:
        logger.info(f"Testing insert into test_variants with minimal data")
        result = supabase.table("test_variants").insert(test_data).execute()
        logger.info(f"Success! Required fields: {list(test_data.keys())}")
        
        # Get the record back to see all fields
        inserted_id = result.data[0]['id'] if hasattr(result, 'data') and result.data else test_data['id']
        get_result = supabase.table("test_variants").select("*").eq("id", inserted_id).execute()
        
        if hasattr(get_result, 'data') and get_result.data:
            logger.info(f"All fields: {list(get_result.data[0].keys())}")
        
        # Clean up test data
        supabase.table("test_variants").delete().eq("id", inserted_id).execute()
        return True
    except Exception as e:
        logger.error(f"Error testing test_variants: {str(e)}")
        
        # Try different combinations if it fails
        if "Could not find" in str(e):
            field = str(e).split("'")[1] if "'" in str(e) else "unknown"
            logger.info(f"Removing field {field} and trying again")
            if field in test_data:
                del test_data[field]
                return test_insert_test_variants()
        return False

def test_insert_test_daily_metrics():
    """Test inserting into test_daily_metrics table."""
    test_data = {
        "id": str(uuid.uuid4()),
        "variant_id": str(uuid.uuid4()),  # This would normally be a real variant ID
        "date": datetime.now().date().isoformat(),
        "impressions": 1000
    }
    
    try:
        logger.info(f"Testing insert into test_daily_metrics with minimal data")
        result = supabase.table("test_daily_metrics").insert(test_data).execute()
        logger.info(f"Success! Required fields: {list(test_data.keys())}")
        
        # Get the record back to see all fields
        inserted_id = result.data[0]['id'] if hasattr(result, 'data') and result.data else test_data['id']
        get_result = supabase.table("test_daily_metrics").select("*").eq("id", inserted_id).execute()
        
        if hasattr(get_result, 'data') and get_result.data:
            logger.info(f"All fields: {list(get_result.data[0].keys())}")
        
        # Clean up test data
        supabase.table("test_daily_metrics").delete().eq("id", inserted_id).execute()
        return True
    except Exception as e:
        logger.error(f"Error testing test_daily_metrics: {str(e)}")
        
        # Try different combinations if it fails
        if "Could not find" in str(e):
            field = str(e).split("'")[1] if "'" in str(e) else "unknown"
            logger.info(f"Removing field {field} and trying again")
            if field in test_data:
                del test_data[field]
                return test_insert_test_daily_metrics()
        return False

def test_insert_users():
    """Test inserting into users table."""
    test_data = {
        "id": str(uuid.uuid4()),
        "email": "test@example.com",
        "name": "Test User"
    }
    
    try:
        logger.info(f"Testing insert into users with minimal data")
        result = supabase.table("users").insert(test_data).execute()
        logger.info(f"Success! Required fields: {list(test_data.keys())}")
        
        # Get the record back to see all fields
        inserted_id = result.data[0]['id'] if hasattr(result, 'data') and result.data else test_data['id']
        get_result = supabase.table("users").select("*").eq("id", inserted_id).execute()
        
        if hasattr(get_result, 'data') and get_result.data:
            logger.info(f"All fields: {list(get_result.data[0].keys())}")
        
        # Clean up test data
        supabase.table("users").delete().eq("id", inserted_id).execute()
        return True
    except Exception as e:
        logger.error(f"Error testing users: {str(e)}")
        
        # Try different combinations if it fails
        if "Could not find" in str(e):
            field = str(e).split("'")[1] if "'" in str(e) else "unknown"
            logger.info(f"Removing field {field} and trying again")
            if field in test_data:
                del test_data[field]
                return test_insert_users()
        return False

def test_insert_user_roles():
    """Test inserting into user_roles table."""
    test_data = {
        "id": str(uuid.uuid4()),
        "user_id": str(uuid.uuid4()),  # This would normally be a real user ID
        "role_id": str(uuid.uuid4())   # This would normally be a real role ID
    }
    
    try:
        logger.info(f"Testing insert into user_roles with minimal data")
        result = supabase.table("user_roles").insert(test_data).execute()
        logger.info(f"Success! Required fields: {list(test_data.keys())}")
        
        # Get the record back to see all fields
        inserted_id = result.data[0]['id'] if hasattr(result, 'data') and result.data else test_data['id']
        get_result = supabase.table("user_roles").select("*").eq("id", inserted_id).execute()
        
        if hasattr(get_result, 'data') and get_result.data:
            logger.info(f"All fields: {list(get_result.data[0].keys())}")
        
        # Clean up test data
        supabase.table("user_roles").delete().eq("id", inserted_id).execute()
        return True
    except Exception as e:
        logger.error(f"Error testing user_roles: {str(e)}")
        
        # Try different combinations if it fails
        if "Could not find" in str(e):
            field = str(e).split("'")[1] if "'" in str(e) else "unknown"
            logger.info(f"Removing field {field} and trying again")
            if field in test_data:
                del test_data[field]
                return test_insert_user_roles()
        return False

def main():
    """Run tests on all tables."""
    logger.info("Starting table inspection...")
    
    # Test each table
    test_insert_ab_tests()
    test_insert_test_variants()
    test_insert_test_daily_metrics()
    test_insert_users()
    test_insert_user_roles()
    
    logger.info("Table inspection complete")

if __name__ == "__main__":
    main() 