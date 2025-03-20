# Schema Migration Guide for Developers

This guide provides instructions for developers on how to update their code to work with the new schema organization in the OnSpot Predictive Model database.

## Overview of Changes

We've reorganized the database tables from the default `public` schema into logical schemas:

1. `core` - Core data pipeline tables
2. `monitoring` - System monitoring and model lifecycle tables
3. `analytics` - Business intelligence and reporting tables
4. `experimentation` - A/B testing tables
5. `auth` - User management tables

## Backward Compatibility

For backward compatibility, we've created views in the `public` schema that point to the new tables. This means that **existing code will continue to work** without modification, but we recommend updating your code to explicitly use the new schemas for better clarity and to avoid potential issues in the future.

## How to Update Your Code

### 1. Update Supabase Queries

#### Old code:
```python
# Using table() method
result = supabase.table('models').select('*').execute()

# Using from() method
result = supabase.from_('models').select('*').execute()
```

#### New code:
```python
# Using from_() method with schema prefix
result = supabase.from_('core.models').select('*').execute()
```

### 2. Update SQL Queries

#### Old code:
```sql
SELECT * FROM models WHERE id = '123';

INSERT INTO predictions (model_id, value) VALUES ('123', 42);
```

#### New code:
```sql
SELECT * FROM core.models WHERE id = '123';

INSERT INTO core.predictions (model_id, value) VALUES ('123', 42);
```

### 3. Table to Schema Mapping

Use this table to determine which schema each table belongs to:

| Table Name | New Schema |
|------------|------------|
| models | core |
| predictions | core |
| raw_parking_data | core |
| cleaned_parking_data | core |
| feature_engineered_data | core |
| drift_analysis | monitoring |
| retraining_events | monitoring |
| system_health | monitoring |
| business_metrics | analytics |
| location_metrics | analytics |
| ab_tests | experimentation |
| test_variants | experimentation |
| test_daily_metrics | experimentation |
| users | auth |
| user_roles | auth |

## Automated Code Update Tool

We've provided a Python script to help you automatically update your codebase:

```bash
# Run in dry-run mode to see what would change without modifying files
python scripts/update_application_code.py --directory ./your/code/directory --dry-run

# Run without --dry-run to apply the changes
python scripts/update_application_code.py --directory ./your/code/directory
```

Options:
- `--directory`: Directory to scan (default: current directory)
- `--extensions`: Comma-separated file extensions to process (default: .py)
- `--dry-run`: Print changes without modifying files
- `--verbose`: Print more detailed information

## Testing Your Updates

After updating your code, thoroughly test all database interactions:

1. Run your application in development mode
2. Test each feature that involves database access
3. Check logs for any SQL errors
4. Verify that data is being correctly written to and read from the database

## Common Issues

### 1. Schema Name Not Found

If you see errors like:
```
ERROR: schema "core" does not exist
```

Make sure the schemas have been created in your database. Run the schema creation SQL:

```sql
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS experimentation;
CREATE SCHEMA IF NOT EXISTS auth;
```

### 2. Permission Denied

If you see errors like:
```
ERROR: permission denied for schema core
```

Make sure the appropriate permissions have been granted:

```sql
GRANT USAGE ON SCHEMA core TO authenticated;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA core TO authenticated;
```

### 3. Foreign Key Issues

If you encounter foreign key constraint issues, check that all foreign key references have been updated to point to tables in their new schemas.

## Best Practices Going Forward

1. **Always use schema prefixes** when referencing tables in SQL queries
2. **Use the `from_()` method** in Supabase queries instead of `table()`
3. **Document schema information** in new code and comments
4. **Verify schema existence** when setting up new development environments

## Questions or Issues?

If you encounter any issues not covered in this guide, please contact the database team for assistance. 