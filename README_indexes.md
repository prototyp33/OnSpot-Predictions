# Supabase Database Indexing

This documentation describes the database indexing implementation for the OnSpot Predictive Model project. 
Proper indexing is essential for query performance, especially as data volumes grow.

## Overview

We've implemented a systematic approach to database indexing that:

1. Analyzes the database schema and query patterns
2. Creates appropriate indexes for common query patterns
3. Balances query performance with index maintenance overhead
4. Prioritizes indexing for the most frequently accessed tables and queries

## Files in this Implementation

* `analyze_query_patterns.py` - Script that scans the codebase to detect query patterns and recommend indexes
* `create_indexes.sql` - Generated SQL file containing all index creation statements
* `apply_indexes.py` - Script to apply the indexes to the Supabase database

## Index Design Strategy

Our indexing strategy focuses on:

### 1. Single-Column Indexes

We've added single-column indexes for:
* Primary keys
* Foreign keys
* Common filter columns (e.g., timestamps, status, categories)
* Common join columns

### 2. Composite Indexes

We've added composite (multi-column) indexes for:
* Common combinations of filters used together
* Typical time-series query patterns (e.g., location_id + timestamp)
* Columns frequently used in GROUP BY or ORDER BY clauses

### 3. Index Types

Most indexes use the default B-tree index type, which is well-suited for:
* Equality comparisons (col = value)
* Range queries (col > value)
* Prefix searches (col LIKE 'prefix%')
* Sorting (ORDER BY col)

## Indexed Tables

We've added indexes to the following tables:

| Table | Key Indexed Columns | Purpose |
|-------|---------------------|---------|
| business_metrics | timestamp, metric_name, category | Time-series analysis and metrics filtering |
| location_metrics | location_id, date | Location-based and date-range queries |
| system_health | component, status, timestamp | System monitoring and status filtering |
| ab_tests | status, start_date, end_date | Test management and reporting |
| test_variants | test_id, name | Test variant analysis |
| test_daily_metrics | variant_id, date | Test metrics and time-series analysis |
| users | email | User lookups and authentication |
| user_roles | user_id, role_id | Role-based access control |
| drift_analysis | model_id, timestamp | Model drift monitoring |
| retraining_events | model_id, timestamp, success | Model retraining history |
| predictions | model_id, location_id, timestamp | Prediction analysis and retrieval |
| models | model_type, training_date | Model metadata retrieval |
| raw_parking_data | location_id, timestamp | Raw data access |
| cleaned_parking_data | location_id, timestamp | Cleaned data access |
| feature_engineered_data | location_id, timestamp, day_of_week, etc. | Feature filtering for ML |

## How to Apply Indexes

### Option 1: Using the Python Script

Run the `apply_indexes.py` script:

```bash
python apply_indexes.py
```

This script will:
1. Parse the SQL file into individual statements
2. Connect to your Supabase database
3. Execute each index creation statement
4. Report on successes and failures

### Option 2: Supabase SQL Editor

If the script approach doesn't work (which may happen if the RPC function isn't available):

1. Open your Supabase project dashboard
2. Go to the SQL Editor
3. Create a new query and paste the contents of `create_indexes.sql`
4. Execute the query

## Monitoring Index Performance

After adding these indexes, you should monitor:

1. Query performance improvements
2. Index usage statistics
3. Insert/update overhead

The `scripts/supabase_monitor.py` and related modules can help track query performance improvements.

## Future Index Maintenance

Indexes should be periodically reviewed as:
* Data volumes grow
* Query patterns change
* New tables are added

Use the `analyze_query_patterns.py` script periodically to generate updated index recommendations as your application evolves.

## Index Size Considerations

Indexes improve query performance but require storage space and maintenance overhead. For large tables, consider:

* Removing unused indexes
* Consolidating indexes where possible
* Using partial indexes for specialized queries 