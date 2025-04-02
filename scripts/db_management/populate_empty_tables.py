#!/usr/bin/env python
"""
Script to populate empty tables in the Supabase database with sample data.
Uses batch insert for better performance where applicable.
"""

import os
import sys
import uuid
import json
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from supabase import create_client
import backoff
import time
import ssl
import httpx
from requests.exceptions import RequestException

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

# Retry mechanism for transient errors
@backoff.on_exception(
    backoff.expo,
    (RequestException, ssl.SSLError, httpx.HTTPError),
    max_tries=5,
    max_time=300,
    jitter=backoff.full_jitter
)
def batch_insert(table_name: str, records: List[Dict[str, Any]]) -> bool:
    """Insert batch of records with retry logic for transient errors."""
    try:
        if not records:
            logger.warning(f"No records to insert into {table_name}")
            return False
            
        logger.info(f"Inserting {len(records)} records into {table_name}")
        result = supabase.table(table_name).insert(records).execute()
        
        success_count = len(result.data) if hasattr(result, 'data') else 0
        logger.info(f"Successfully inserted {success_count} records into {table_name}")
        return success_count > 0
    except Exception as e:
        logger.error(f"Error inserting into {table_name}: {str(e)}")
        raise

def generate_business_metrics(count: int = 10) -> List[Dict[str, Any]]:
    """Generate sample business metrics data."""
    categories = ["revenue", "costs", "efficiency", "conversion", "retention"]
    location_ids = [f"location_{i}" for i in range(1, 6)]
    metrics = [
        "daily_revenue", "monthly_revenue", "occupancy_rate", 
        "customer_satisfaction", "retention_rate", "acquisition_cost"
    ]
    
    records = []
    now = datetime.now()
    
    for i in range(count):
        timestamp = (now - timedelta(days=i % 30)).isoformat()
        category = random.choice(categories)
        metric_name = random.choice(metrics)
        
        record = {
            "id": str(uuid.uuid4()),
            "metric_name": metric_name,
            "metric_value": round(random.uniform(10, 1000), 2),
            "timestamp": timestamp,
            "category": category,
            "location_id": random.choice(location_ids) if random.random() > 0.3 else None,
            "metadata": {
                "source": "sample_data",
                "confidence": round(random.uniform(0.5, 0.99), 2)
            }
        }
        records.append(record)
    
    return records

def generate_location_metrics(count: int = 10) -> List[Dict[str, Any]]:
    """Generate sample location metrics data."""
    location_ids = [f"location_{i}" for i in range(1, 6)]
    records = []
    now = datetime.now()
    
    for i in range(count):
        date = (now - timedelta(days=i % 30)).date().isoformat()
        location_id = random.choice(location_ids)
        
        record = {
            "id": str(uuid.uuid4()),
            "location_id": location_id,
            "date": date,
            "occupancy_accuracy": round(random.uniform(0.6, 0.95), 3),
            "utilization_rate": round(random.uniform(0.3, 0.9), 3),
            "revenue": round(random.uniform(500, 5000), 2),
            "opportunity_cost": round(random.uniform(50, 500), 2) if random.random() > 0.7 else None
        }
        records.append(record)
    
    return records

def generate_system_health(count: int = 10) -> List[Dict[str, Any]]:
    """Generate sample system health data."""
    components = ["api_server", "ml_pipeline", "data_ingestion", "monitoring", "notification_service"]
    statuses = ["operational", "degraded", "down"]
    alert_levels = ["info", "warning", "critical"]
    
    records = []
    now = datetime.now()
    
    for i in range(count):
        timestamp = (now - timedelta(hours=i % 24)).isoformat()
        component = random.choice(components)
        
        # Make most components operational most of the time
        status_weights = [0.8, 0.15, 0.05]
        status = random.choices(statuses, weights=status_weights)[0]
        
        # Alert level based on status
        if status == "operational":
            alert_level = "info"
        elif status == "degraded":
            alert_level = "warning"
        else:
            alert_level = "critical"
        
        record = {
            "id": str(uuid.uuid4()),
            "timestamp": timestamp,
            "component": component,
            "status": status,
            "metrics": {
                "cpu_usage": round(random.uniform(10, 95), 1),
                "memory_usage": round(random.uniform(20, 90), 1),
                "response_time_ms": round(random.uniform(50, 500), 1)
            },
            "alert_level": alert_level,
            "message": f"System {component} is {status}" if status != "operational" else None
        }
        records.append(record)
    
    return records

def generate_ab_tests(count: int = 5) -> List[Dict[str, Any]]:
    """Generate sample A/B test data."""
    test_names = [
        "homepage_redesign", "pricing_model", "recommendation_algorithm", 
        "notification_frequency", "onboarding_flow"
    ]
    
    records = []
    now = datetime.now()
    
    for i in range(count):
        start_date = (now - timedelta(days=random.randint(1, 90))).date().isoformat()
        end_date = (now + timedelta(days=random.randint(1, 90))).date().isoformat() if random.random() > 0.3 else None
        
        record = {
            "id": str(uuid.uuid4()),
            "name": test_names[i % len(test_names)],
            "description": f"Testing impact of changes to {test_names[i % len(test_names)]}",
            "start_date": start_date,
            "end_date": end_date,
            "status": "active" if not end_date else "completed",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat()
        }
        records.append(record)
    
    return records

def generate_test_variants(count: int = 10) -> List[Dict[str, Any]]:
    """Generate sample test variants for A/B tests."""
    # First get the A/B test IDs
    try:
        ab_tests_result = supabase.table('ab_tests').select('id, name').execute()
        ab_tests = ab_tests_result.data if hasattr(ab_tests_result, 'data') else []
        
        if not ab_tests:
            logger.warning("No A/B tests found, cannot generate variants")
            return []
        
        records = []
        
        for test in ab_tests:
            test_id = test['id']
            test_name = test['name']
            
            # Create control variant
            control = {
                "id": str(uuid.uuid4()),
                "test_id": test_id,
                "name": "control",
                "description": f"Control variant for {test_name}",
                "configuration": json.dumps({"is_control": True}),
                "traffic_percentage": 50.0,
                "created_at": datetime.now().isoformat()
            }
            records.append(control)
            
            # Create experimental variant
            experiment = {
                "id": str(uuid.uuid4()),
                "test_id": test_id,
                "name": "variant_a",
                "description": f"Experimental variant for {test_name}",
                "configuration": json.dumps({"key_param": "new_value"}),
                "traffic_percentage": 50.0,
                "created_at": datetime.now().isoformat()
            }
            records.append(experiment)
        
        return records
    except Exception as e:
        logger.error(f"Error generating test variants: {str(e)}")
        return []

def generate_test_daily_metrics(count: int = 30) -> List[Dict[str, Any]]:
    """Generate sample daily metrics for A/B tests."""
    try:
        # Get test variants
        variants_result = supabase.table('test_variants').select('id, test_id, name').execute()
        variants = variants_result.data if hasattr(variants_result, 'data') else []
        
        if not variants:
            logger.warning("No test variants found. Creating sample test_daily_metrics with dummy variants")
            # Create some test data with dummy variant IDs
            records = []
            now = datetime.now()
            
            # Create a few dummy variant IDs
            dummy_variant_ids = [str(uuid.uuid4()) for _ in range(2)]
            
            for i in range(count):
                for variant_id in dummy_variant_ids:
                    date = (now - timedelta(days=i)).date().isoformat()
                    
                    record = {
                        "id": str(uuid.uuid4()),
                        "variant_id": variant_id,
                        "date": date,
                        "impressions": random.randint(1000, 10000),
                        "clicks": random.randint(50, 500),
                        "conversions": random.randint(5, 50),
                        "revenue": round(random.uniform(100, 1000), 2),
                        "created_at": now.isoformat()
                    }
                    records.append(record)
            
            return records
        
        records = []
        now = datetime.now()
        
        for i in range(count):
            for variant in variants:
                variant_id = variant['id']
                date = (now - timedelta(days=i)).date().isoformat()
                
                record = {
                    "id": str(uuid.uuid4()),
                    "variant_id": variant_id,
                    "date": date,
                    "impressions": random.randint(1000, 10000),
                    "clicks": random.randint(50, 500),
                    "conversions": random.randint(5, 50),
                    "revenue": round(random.uniform(100, 1000), 2),
                    "created_at": now.isoformat()
                }
                records.append(record)
        
        return records
    except Exception as e:
        logger.error(f"Error generating test daily metrics: {str(e)}")
        return []

def generate_users(count: int = 5) -> List[Dict[str, Any]]:
    """Generate sample user data."""
    names = ["John Doe", "Jane Smith", "Alex Johnson", "Maria Garcia", "Raj Patel"]
    roles = ["admin", "analyst", "viewer", "editor", "manager"]
    
    records = []
    now = datetime.now()
    
    for i in range(count):
        record = {
            "id": str(uuid.uuid4()),
            "email": f"user{i+1}@example.com",
            "name": names[i % len(names)],
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "active": True
        }
        records.append(record)
    
    return records

def generate_user_roles(count: int = 10) -> List[Dict[str, Any]]:
    """Generate sample user roles data."""
    try:
        # Get users
        users_result = supabase.table('users').select('id').execute()
        users = users_result.data if hasattr(users_result, 'data') else []
        
        if not users:
            logger.warning("No users found, cannot generate user roles")
            return []
        
        # Get roles
        roles_result = supabase.table('roles').select('id').execute()
        roles = roles_result.data if hasattr(roles_result, 'data') else []
        
        if not roles:
            logger.warning("No roles found, cannot generate user roles")
            return []
        
        records = []
        
        # Assign roles to users
        for user in users:
            # Assign 1-2 roles per user
            num_roles = random.randint(1, min(2, len(roles)))
            selected_roles = random.sample(roles, num_roles)
            
            for role in selected_roles:
                record = {
                    "id": str(uuid.uuid4()),
                    "user_id": user['id'],
                    "role_id": role['id'],
                    "created_at": datetime.now().isoformat()
                }
                records.append(record)
        
        return records
    except Exception as e:
        logger.error(f"Error generating user roles: {str(e)}")
        return []

def populate_all_tables():
    """Populate all empty tables with sample data."""
    # List of tables to populate with their respective data generation functions
    tables_to_populate = [
        {
            "name": "business_metrics",
            "generator": generate_business_metrics,
            "count": 20
        },
        {
            "name": "location_metrics", 
            "generator": generate_location_metrics,
            "count": 20
        },
        {
            "name": "system_health",
            "generator": generate_system_health,
            "count": 15
        },
        {
            "name": "ab_tests",
            "generator": generate_ab_tests,
            "count": 5
        },
        {
            "name": "users",
            "generator": generate_users,
            "count": 5
        }
    ]
    
    # First populate the base tables
    for table in tables_to_populate:
        try:
            records = table["generator"](table["count"])
            if records:
                batch_insert(table["name"], records)
            else:
                logger.warning(f"No records generated for {table['name']}")
        except Exception as e:
            logger.error(f"Error populating {table['name']}: {str(e)}")
    
    # Then populate tables with dependencies
    try:
        # Test variants depend on AB tests
        test_variants = generate_test_variants()
        if test_variants:
            batch_insert("test_variants", test_variants)
        
        # Test daily metrics depend on test variants
        test_daily_metrics = generate_test_daily_metrics()
        if test_daily_metrics:
            batch_insert("test_daily_metrics", test_daily_metrics)
            
        # User roles depend on users and roles
        user_roles = generate_user_roles()
        if user_roles:
            batch_insert("user_roles", user_roles)
        
    except Exception as e:
        logger.error(f"Error populating dependent tables: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting to populate empty tables in Supabase...")
    populate_all_tables()
    logger.info("Finished populating tables.") 