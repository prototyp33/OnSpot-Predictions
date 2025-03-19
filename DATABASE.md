# OnSpot Predictive Model - Database Schema

This document explains the database schema for the OnSpot Predictive Model application, including tables for monitoring, drift detection, A/B testing, and business impact metrics.

## Database Tables

### Existing Tables

These tables are already part of the system:

- `raw_parking_data`: Raw data from parking sensors
- `cleaned_parking_data`: Preprocessed data with weather information
- `feature_engineered_data`: Data with additional features for modeling
- `model_metrics`: Performance metrics for ML models
- `data_quality_metrics`: Data quality measurements
- `health_metrics`: System health information

### New Monitoring Tables

The following tables have been added to enhance the monitoring system:

1. **drift_analysis**
   - Tracks feature drift over time for each model
   - Includes metrics like KS-statistic, p-value, mean differences
   - Stores categorical feature changes in JSON fields

2. **retraining_events**
   - Records when models are retrained and why
   - Tracks performance before and after retraining
   - Enables audit trails of model updates

3. **business_metrics**
   - Stores financial and operational impact metrics
   - Categories may include revenue, costs, efficiency, etc.
   - Can be linked to locations to track regional performance

4. **location_metrics**
   - Location-specific metrics including occupancy accuracy
   - Tracks revenue and opportunity costs per location
   - Helps identify which areas benefit most from the model

5. **system_health**
   - Monitors infrastructure components
   - Tracks statuses, alerts, and performance metrics
   - Enables system-level monitoring

6. **A/B Testing Tables**
   - `ab_tests`: Test configurations and results
   - `test_variants`: Different versions being tested
   - `test_daily_metrics`: Daily performance of each variant

7. **User Management**
   - `users`: User accounts for dashboard access
   - `roles`: Access control roles with permissions
   - `user_roles`: User-role assignments

## Setting Up the Database

### Prerequisites

- PostgreSQL database (or Supabase)
- Database connection string in the format: `postgresql://username:password@host:port/database`

### Applying Migrations

1. Set your database connection string:

```bash
export DATABASE_URL="postgresql://your_username:your_password@localhost:5432/your_database"
```

2. Run the migration script to create the new tables:

```bash
python scripts/apply_migrations.py --file sql/create_monitoring_tables.sql
```

### Using the Database Client

The application includes a `DatabaseClient` class in `scripts/db_integration.py` that provides methods for interacting with the database:

```python
from db_integration import DatabaseClient

# Initialize the client
with DatabaseClient() as db:
    # Store drift analysis
    db.store_drift_analysis(
        model_id="model_v1",
        drift_metrics=drift_metrics,
        baseline_timestamp=datetime.now()
    )
    
    # Store business metrics
    db.store_business_metric(
        metric_name="revenue_impact",
        metric_value=12500.0,
        category="revenue",
        location_id="downtown_1"
    )
```

## Integration with Monitoring System

The monitoring system has been updated to automatically store drift detection results in the database. The `check_for_drift` method in `MonitoringPipeline` now calls `store_drift_analysis` when drift is detected.

### Enabling Database Integration

To enable database integration:

1. Ensure the `DATABASE_URL` environment variable is set
2. Make sure `psycopg2` is installed: `pip install psycopg2-binary`
3. Run the monitoring pipeline as usual: `python scripts/automated_monitoring.py`

## Querying the Data

### Example Queries

#### Get Drift Metrics for a Model

```sql
SELECT 
    feature_name, 
    drift_score, 
    timestamp,
    p_value,
    mean_difference,
    std_difference
FROM drift_analysis
WHERE model_id = 'model_v1'
ORDER BY timestamp DESC;
```

#### Track Retraining Events

```sql
SELECT 
    model_id, 
    timestamp, 
    reason, 
    success
FROM retraining_events
ORDER BY timestamp DESC;
```

#### Business Impact by Location

```sql
SELECT 
    location_id,
    AVG(revenue) as avg_revenue,
    AVG(opportunity_cost) as avg_opportunity_cost,
    AVG(occupancy_accuracy) as avg_accuracy
FROM location_metrics
GROUP BY location_id
ORDER BY avg_revenue DESC;
```

## Dashboard Integration

These database tables are designed to be used with the OnSpot dashboard, providing data for:

- Model Performance page: `model_metrics` and `drift_analysis`
- Business Impact page: `business_metrics` and `location_metrics`
- System Health page: `health_metrics` and `system_health`
- A/B Testing page: `ab_tests`, `test_variants`, and `test_daily_metrics`

## Maintenance

### Regular Cleanup

For tables that accumulate a lot of data, consider implementing a retention policy:

```sql
-- Example: Delete drift_analysis data older than 1 year
DELETE FROM drift_analysis
WHERE timestamp < NOW() - INTERVAL '1 year';
```

### Table Optimization

For production use, consider adding additional indexes based on your query patterns, and regularly running VACUUM and ANALYZE commands to maintain performance. 