#!/usr/bin/env python
"""
Final script to populate empty tables in the Supabase database.
Based on inspection of actual table schemas.
"""

import os
import sys
import uuid
import json
import random
import logging
from datetime import datetime, timedelta
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

def populate_business_metrics(count: int = 20):
    """Populate business_metrics table."""
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
    
    try:
        logger.info(f"Inserting {len(records)} records into business_metrics")
        result = supabase.table('business_metrics').insert(records).execute()
        success_count = len(result.data) if hasattr(result, 'data') else 0
        logger.info(f"Successfully inserted {success_count} records into business_metrics")
        return True
    except Exception as e:
        logger.error(f"Error inserting into business_metrics: {str(e)}")
        return False

def populate_location_metrics(count: int = 20):
    """Populate location_metrics table."""
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
    
    try:
        logger.info(f"Inserting {len(records)} records into location_metrics")
        result = supabase.table('location_metrics').insert(records).execute()
        success_count = len(result.data) if hasattr(result, 'data') else 0
        logger.info(f"Successfully inserted {success_count} records into location_metrics")
        return True
    except Exception as e:
        logger.error(f"Error inserting into location_metrics: {str(e)}")
        return False

def populate_system_health(count: int = 15):
    """Populate system_health table."""
    components = ["api_server", "ml_pipeline", "data_ingestion", "monitoring", "notification_service"]
    statuses = ["operational", "degraded", "down"]
    
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
    
    try:
        logger.info(f"Inserting {len(records)} records into system_health")
        result = supabase.table('system_health').insert(records).execute()
        success_count = len(result.data) if hasattr(result, 'data') else 0
        logger.info(f"Successfully inserted {success_count} records into system_health")
        return True
    except Exception as e:
        logger.error(f"Error inserting into system_health: {str(e)}")
        return False

def populate_ab_tests(count: int = 5):
    """Populate ab_tests table with required status field."""
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
            "status": "active" if not end_date else "completed"
        }
        records.append(record)
    
    try:
        logger.info(f"Inserting {len(records)} records into ab_tests")
        result = supabase.table('ab_tests').insert(records).execute()
        success_count = len(result.data) if hasattr(result, 'data') else 0
        logger.info(f"Successfully inserted {success_count} records into ab_tests")
        return True
    except Exception as e:
        logger.error(f"Error inserting into ab_tests: {str(e)}")
        return False

def populate_test_variants():
    """Populate test_variants table."""
    try:
        # Get the AB test IDs
        ab_tests_result = supabase.table('ab_tests').select('id, name').execute()
        ab_tests = ab_tests_result.data if hasattr(ab_tests_result, 'data') else []
        
        if not ab_tests:
            logger.warning("No A/B tests found, cannot generate variants")
            return False
        
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
                "traffic_percentage": 50
            }
            records.append(control)
            
            # Create experimental variant
            experiment = {
                "id": str(uuid.uuid4()),
                "test_id": test_id,
                "name": "variant_a",
                "description": f"Experimental variant for {test_name}",
                "traffic_percentage": 50
            }
            records.append(experiment)
        
        if not records:
            return False
            
        logger.info(f"Inserting {len(records)} records into test_variants")
        result = supabase.table('test_variants').insert(records).execute()
        success_count = len(result.data) if hasattr(result, 'data') else 0
        logger.info(f"Successfully inserted {success_count} records into test_variants")
        return True
    except Exception as e:
        logger.error(f"Error inserting into test_variants: {str(e)}")
        return False

def populate_test_daily_metrics():
    """Populate test_daily_metrics table with minimal valid fields."""
    try:
        # Get variants
        variants_result = supabase.table('test_variants').select('id').execute()
        variants = variants_result.data if hasattr(variants_result, 'data') else []
        
        if not variants:
            logger.warning("No variants found, cannot create test_daily_metrics")
            return False
        
        records = []
        now = datetime.now()
        
        for i in range(15):  # 15 days of history
            date = (now - timedelta(days=i)).date().isoformat()
            
            for variant in variants:
                record = {
                    "id": str(uuid.uuid4()),
                    "variant_id": variant['id'],
                    "date": date,
                    "created_at": now.isoformat()
                }
                records.append(record)
        
        if not records:
            return False
            
        logger.info(f"Inserting {len(records)} records into test_daily_metrics")
        result = supabase.table('test_daily_metrics').insert(records).execute()
        success_count = len(result.data) if hasattr(result, 'data') else 0
        logger.info(f"Successfully inserted {success_count} records into test_daily_metrics")
        return True
    except Exception as e:
        logger.error(f"Error inserting into test_daily_metrics: {str(e)}")
        return False

def populate_users(count: int = 5):
    """Populate users table with minimal valid fields."""
    try:
        records = []
        
        for i in range(count):
            # Add random string to email to avoid conflicts
            random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
            record = {
                "id": str(uuid.uuid4()),
                "email": f"user{i+1}_{random_suffix}@example.com"
            }
            records.append(record)
        
        logger.info(f"Inserting {len(records)} records into users")
        result = supabase.table('users').insert(records).execute()
        success_count = len(result.data) if hasattr(result, 'data') else 0
        logger.info(f"Successfully inserted {success_count} records into users")
        return True
    except Exception as e:
        logger.error(f"Error inserting into users: {str(e)}")
        return False

def populate_all():
    """Populate all empty tables."""
    results = {}
    
    # Populate the tables in order
    results["business_metrics"] = populate_business_metrics()
    results["location_metrics"] = populate_location_metrics() 
    results["system_health"] = populate_system_health()
    results["ab_tests"] = populate_ab_tests()
    results["users"] = populate_users()
    
    # Populate dependent tables if their parent tables were populated
    if results["ab_tests"]:
        results["test_variants"] = populate_test_variants()
        
        if results.get("test_variants", True):
            results["test_daily_metrics"] = populate_test_daily_metrics()
    
    # Skip user_roles for now as it requires more complex setup
    
    # Print summary
    logger.info("Population results:")
    for table, success in results.items():
        logger.info(f"  {table}: {'✅ Success' if success else '❌ Failed'}")
        
    return results

if __name__ == "__main__":
    logger.info("Starting to populate empty tables...")
    populate_all()
    logger.info("Finished populating tables") 