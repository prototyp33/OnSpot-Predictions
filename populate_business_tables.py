#!/usr/bin/env python
"""
Script to populate empty business tables in Supabase with realistic sample data.
"""

import os
import random
from datetime import datetime, timedelta
import uuid
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

# Connect to Supabase
print(f'Connecting to Supabase...')
supabase = create_client(url, key)

def populate_business_sla(count=5):
    """Populate business_sla table with sample data"""
    print(f'\nPopulating business_sla table with {count} records...')
    
    # Allowed status values from our constraint check: critical, warning, healthy
    valid_statuses = ["critical", "warning", "healthy"]
    
    # SLA data based on service metrics with valid status values
    sla_data = [
        {
            "name": "Model Prediction Accuracy",
            "target": 95.0,
            "actual": random.uniform(93.0, 97.0),
            "financial_impact": "High",
            "status": "healthy",  # Using valid lowercase status value
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        },
        {
            "name": "API Response Time",
            "target": 200.0,  # milliseconds
            "actual": random.uniform(180.0, 220.0),
            "financial_impact": "Medium",
            "status": "warning",  # Using valid lowercase status value
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        },
        {
            "name": "Data Processing Latency",
            "target": 15.0,  # minutes
            "actual": random.uniform(12.0, 18.0),
            "financial_impact": "Low",
            "status": "healthy",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        },
        {
            "name": "Availability",
            "target": 99.9,  # percentage
            "actual": random.uniform(99.7, 100.0),
            "financial_impact": "High",
            "status": "healthy",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        },
        {
            "name": "Error Rate",
            "target": 1.0,  # percentage
            "actual": random.uniform(0.5, 1.5),
            "financial_impact": "Medium",
            "status": "critical",  # Using valid lowercase status value
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    ]
    
    # Select the requested number of records
    data_to_insert = sla_data[:count]
    
    success_count = 0
    for record in data_to_insert:
        try:
            result = supabase.table('business_sla').insert(record).execute()
            if len(result.data) > 0:
                success_count += 1
        except Exception as e:
            print(f"Error inserting SLA record: {str(e)}")
    
    print(f"Successfully inserted {success_count} of {count} SLA records")
    return success_count

def populate_business_indicators(num_days=14):
    """Populate business_indicators table with time series data"""
    print(f'\nPopulating business_indicators table with {num_days} days of data...')
    
    success_count = 0
    for i in range(num_days):
        day_offset = num_days - i - 1
        date = datetime.now() - timedelta(days=day_offset)
        day_of_month = date.day
        
        # Random values with some consistent trends over time
        model_drift = max(0.01, 0.1 + (i * 0.01))  # increasing drift over time
        api_latency = 150 + random.uniform(-30, 50)  # variable latency
        error_rate = max(0.5, 1.0 + (i * 0.2 * random.uniform(-1, 1)))  # fluctuating error rate
        warning_threshold = 3.0
        critical_threshold = 5.0
        latency_warning = 200.0
        latency_critical = 500.0
        
        record = {
            "day": day_of_month,
            "model_drift": model_drift,
            "api_latency": api_latency,
            "error_rate": error_rate,
            "warning_threshold": warning_threshold,
            "critical_threshold": critical_threshold,
            "latency_warning": latency_warning,
            "latency_critical": latency_critical,
            "created_at": date.isoformat(),
            "updated_at": date.isoformat()
        }
        
        try:
            result = supabase.table('business_indicators').insert(record).execute()
            if len(result.data) > 0:
                success_count += 1
        except Exception as e:
            print(f"Error inserting indicator record for day {day_of_month}: {str(e)}")
    
    print(f"Successfully inserted {success_count} of {num_days} indicator records")
    return success_count

def populate_business_metrics_time_series(num_records=30):
    """Populate business_metrics_time_series table with time-based metrics"""
    print(f'\nPopulating business_metrics_time_series table with {num_records} records...')
    
    # Metrics to track over time
    metrics = ["user_engagement", "conversion_rate", "retention_rate", "usage_frequency"]
    
    success_count = 0
    for i in range(num_records):
        day_offset = num_records - i - 1
        date = datetime.now() - timedelta(days=day_offset)
        date_str = date.strftime("%Y-%m-%d")
        
        # Create different metrics with realistic values
        for metric in metrics:
            # Base value depends on metric type
            if metric == "user_engagement":
                base_value = 65.0  # percentage
            elif metric == "conversion_rate":
                base_value = 4.5   # percentage
            elif metric == "retention_rate":
                base_value = 85.0  # percentage
            else:  # usage_frequency
                base_value = 3.2   # times per week
                
            # Add some random variation and slight trend
            trend_factor = i * 0.05  # slight upward trend
            random_factor = random.uniform(-2.0, 2.0)
            value = base_value + trend_factor + random_factor
            
            record = {
                "date": date_str,
                "metric": metric,
                "value": value,
                "created_at": date.isoformat()
            }
            
            try:
                result = supabase.table('business_metrics_time_series').insert(record).execute()
                if len(result.data) > 0:
                    success_count += 1
            except Exception as e:
                print(f"Error inserting metrics time series record: {str(e)}")
    
    print(f"Successfully inserted {success_count} business metrics time series records")
    return success_count

def populate_risk_assessment(count=5):
    """Populate risk_assessment table with sample risk data"""
    print(f'\nPopulating risk_assessment table with {count} records...')
    
    # Allowed status values from our constraint check: critical, warning, healthy
    valid_statuses = ["critical", "warning", "healthy"]
    
    # Updated risk data with valid status values
    risk_data = [
        {
            "name": "Model Drift Risk",
            "impact": "High",
            "likelihood": "Medium",
            "financial_impact": "Significant",
            "customer_impact": "Moderate",
            "mitigation_status": "In Progress",
            "description": "Risk of model performance degradation over time due to changing data patterns",
            "status": "warning",  # Using valid lowercase status value
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        },
        {
            "name": "Data Quality Risk",
            "impact": "High",
            "likelihood": "Low",
            "financial_impact": "Moderate",
            "customer_impact": "High",
            "mitigation_status": "Mitigated",
            "description": "Risk of poor data quality affecting prediction accuracy",
            "status": "healthy",  # Using valid lowercase status value
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        },
        {
            "name": "Performance Scalability",
            "impact": "Medium",
            "likelihood": "Medium",
            "financial_impact": "Low",
            "customer_impact": "High",
            "mitigation_status": "Planned",
            "description": "Risk of performance degradation under high load conditions",
            "status": "warning",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        },
        {
            "name": "Security Vulnerability",
            "impact": "High",
            "likelihood": "Low",
            "financial_impact": "High",
            "customer_impact": "High",
            "mitigation_status": "In Progress",
            "description": "Risk of unauthorized access to sensitive prediction data",
            "status": "critical",  # Using valid lowercase status value
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        },
        {
            "name": "Algorithm Bias",
            "impact": "Medium",
            "likelihood": "Medium",
            "financial_impact": "Medium",
            "customer_impact": "High",
            "mitigation_status": "In Progress",
            "description": "Risk of biased predictions affecting certain user segments",
            "status": "warning",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    ]
    
    # Select the requested number of records
    data_to_insert = risk_data[:count]
    
    success_count = 0
    for record in data_to_insert:
        try:
            result = supabase.table('risk_assessment').insert(record).execute()
            if len(result.data) > 0:
                success_count += 1
        except Exception as e:
            print(f"Error inserting risk assessment record: {str(e)}")
    
    print(f"Successfully inserted {success_count} of {count} risk assessment records")
    return success_count

def populate_financial_data(count=10):
    """Populate financial_data table with sample financial metrics"""
    print(f'\nPopulating financial_data table with {count} records...')
    
    categories = ["Cost", "Revenue", "ROI", "Savings"]
    types = ["Direct", "Indirect", "Projected", "Actual"]
    
    financial_data = []
    for i in range(count):
        category = random.choice(categories)
        fin_type = random.choice(types)
        
        # Set values based on category
        if category == "Cost":
            name = f"{fin_type} Infrastructure Cost"
            value = random.uniform(5000, 50000)
            percentage = random.uniform(5, 25)
        elif category == "Revenue":
            name = f"{fin_type} Revenue Impact"
            value = random.uniform(10000, 100000)
            percentage = random.uniform(1, 15)
        elif category == "ROI":
            name = f"{fin_type} ROI"
            value = random.uniform(1.5, 5.0)
            percentage = random.uniform(50, 400)
        else:  # Savings
            name = f"{fin_type} Cost Savings"
            value = random.uniform(3000, 30000)
            percentage = random.uniform(5, 30)
        
        financial_data.append({
            "category": category,
            "type": fin_type,
            "name": name,
            "value": value,
            "percentage": percentage,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        })
    
    success_count = 0
    for record in financial_data:
        try:
            result = supabase.table('financial_data').insert(record).execute()
            if len(result.data) > 0:
                success_count += 1
        except Exception as e:
            print(f"Error inserting financial data record: {str(e)}")
    
    print(f"Successfully inserted {success_count} of {count} financial data records")
    return success_count

def populate_financial_time_series(num_days=30):
    """Populate financial_time_series table with time-based financial data"""
    print(f'\nPopulating financial_time_series table with {num_days} days of data...')
    
    metrics = ["daily_cost", "revenue_impact", "cost_savings", "roi"]
    
    success_count = 0
    for i in range(num_days):
        day_offset = num_days - i - 1
        date = datetime.now() - timedelta(days=day_offset)
        date_str = date.strftime("%Y-%m-%d")
        
        for metric in metrics:
            # Base value depends on metric type
            if metric == "daily_cost":
                base_value = 500
                variance = 100
            elif metric == "revenue_impact":
                base_value = 1200
                variance = 300
            elif metric == "cost_savings":
                base_value = 300
                variance = 100
            else:  # roi
                base_value = 2.5
                variance = 0.5
                
            # Add time trend (slight increase) and weekday effect
            weekday_factor = 1.0 if date.weekday() < 5 else 0.7  # lower on weekends
            trend_factor = 1.0 + (i * 0.002)  # slight upward trend
            random_factor = random.uniform(-variance, variance)
            
            value = (base_value * weekday_factor * trend_factor) + random_factor
            
            record = {
                "date": date_str,
                "metric": metric,
                "value": value,
                "created_at": date.isoformat()
            }
            
            try:
                result = supabase.table('financial_time_series').insert(record).execute()
                if len(result.data) > 0:
                    success_count += 1
            except Exception as e:
                print(f"Error inserting financial time series record: {str(e)}")
    
    expected_count = num_days * len(metrics)
    print(f"Successfully inserted {success_count} of {expected_count} financial time series records")
    return success_count

def populate_kpi_data(count=8):
    """Populate kpi_data table with key performance indicators"""
    print(f'\nPopulating kpi_data table with {count} KPI records...')
    
    # Allowed status values from our constraint check: critical, warning, healthy, neutral
    valid_statuses = ["critical", "warning", "healthy", "neutral"]
    
    # Allowed trend values from our constraint check: up, down, neutral
    valid_trends = ["up", "down", "neutral"]
    
    # Updated KPI data with valid status and trend values
    kpi_data = [
        {
            "name": "Prediction Accuracy",
            "value": str(random.uniform(92, 98)),
            "trend": "up",  # Valid trend value
            "trend_value": f"+{random.uniform(0.5, 2.5):.1f}%",
            "status": "healthy",  # Using valid lowercase status value
            "description": "Overall accuracy of the prediction model",
            "previous_value": str(random.uniform(90, 95)),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        },
        {
            "name": "Model ROI",
            "value": str(random.uniform(250, 350)),
            "trend": "up",  # Valid trend value
            "trend_value": f"+{random.uniform(5, 15):.1f}%",
            "status": "healthy",
            "description": "Return on investment from the predictive model",
            "previous_value": str(random.uniform(220, 280)),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        },
        {
            "name": "Cost Reduction",
            "value": str(random.uniform(15, 25)),
            "trend": "up",  # Valid trend value
            "trend_value": f"+{random.uniform(1, 5):.1f}%",
            "status": "healthy",
            "description": "Percentage reduction in operational costs",
            "previous_value": str(random.uniform(12, 18)),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        },
        {
            "name": "Decision Time",
            "value": str(random.uniform(30, 45)),
            "trend": "down",  # Valid trend value
            "trend_value": f"-{random.uniform(5, 15):.1f}%",
            "status": "healthy",
            "description": "Average time to make decisions (minutes)",
            "previous_value": str(random.uniform(40, 60)),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        },
        {
            "name": "Resource Utilization",
            "value": str(random.uniform(75, 85)),
            "trend": "up",  # Valid trend value
            "trend_value": f"+{random.uniform(2, 8):.1f}%",
            "status": "healthy",
            "description": "Percentage of resource utilization",
            "previous_value": str(random.uniform(65, 75)),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        },
        {
            "name": "User Satisfaction",
            "value": str(random.uniform(80, 90)),
            "trend": "neutral",  # Valid trend value instead of 'stable'
            "trend_value": f"{random.uniform(-1, 1):.1f}%",
            "status": "neutral",  # Using valid lowercase status value
            "description": "User satisfaction score",
            "previous_value": str(random.uniform(78, 88)),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        },
        {
            "name": "Time to Market",
            "value": str(random.uniform(20, 30)),
            "trend": "down",  # Valid trend value
            "trend_value": f"-{random.uniform(5, 15):.1f}%",
            "status": "healthy",
            "description": "Time to deploy new features (days)",
            "previous_value": str(random.uniform(25, 35)),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        },
        {
            "name": "Data Processing Efficiency",
            "value": str(random.uniform(85, 95)),
            "trend": "up",  # Valid trend value
            "trend_value": f"+{random.uniform(1, 5):.1f}%",
            "status": "healthy",
            "description": "Efficiency of data processing pipeline",
            "previous_value": str(random.uniform(80, 90)),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    ]
    
    # Select the requested number of records
    data_to_insert = kpi_data[:count]
    
    success_count = 0
    for record in data_to_insert:
        try:
            result = supabase.table('kpi_data').insert(record).execute()
            if len(result.data) > 0:
                success_count += 1
        except Exception as e:
            print(f"Error inserting KPI data record: {str(e)}")
    
    print(f"Successfully inserted {success_count} of {count} KPI records")
    return success_count

def main():
    """Main function to populate all empty tables"""
    try:
        print("Starting data population for empty Supabase tables...")
        
        # Dictionary to track results
        results = {}
        
        # Populate each table
        results['business_sla'] = populate_business_sla(5)
        results['business_indicators'] = populate_business_indicators(14) 
        results['business_metrics_time_series'] = populate_business_metrics_time_series(30)
        results['risk_assessment'] = populate_risk_assessment(5)
        results['financial_data'] = populate_financial_data(10)
        results['financial_time_series'] = populate_financial_time_series(30)
        results['kpi_data'] = populate_kpi_data(8)
        
        # Print summary
        print("\n=== POPULATION SUMMARY ===")
        total_records = sum(results.values())
        print(f"Total records inserted: {total_records}")
        for table, count in results.items():
            print(f"- {table}: {count} records")
        
        print("\nData population complete!")
        
    except Exception as e:
        print(f"Error during data population: {str(e)}")

if __name__ == "__main__":
    main() 