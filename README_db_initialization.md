# Database Initialization and Testing

This directory contains scripts for initializing and testing the OnSpot Predictive Model database schema in Supabase.

## Overview

The database initialization process creates all necessary tables, indexes, relationships, and security policies for the OnSpot Predictive Model. It also populates the database with sample data for testing purposes.

## Prerequisites

Before running these scripts, make sure you have:

1. A Supabase project set up
2. Python 3.8+ installed
3. Required Python packages: `supabase`, `python-dotenv`
4. Supabase credentials in a `.env` file (see below)

## Environment Setup

Create a `.env` file in the project root with the following content:

```
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_API_KEY=your-supabase-api-key
```

## Scripts

### 1. Create RPC Functions

First, you need to create the necessary RPC functions in your Supabase database:

1. Navigate to the Supabase dashboard
2. Go to the SQL Editor
3. Copy the contents of `scripts/create_rpc_functions.sql`
4. Execute the SQL in the Supabase SQL Editor

These functions enable the initialization and testing scripts to interact with the database.

### 2. Initialize the Database

Run the initialization script to create all tables and insert sample data:

```bash
python scripts/execute_schema_initialization.py
```

Options:
- `--sql-file PATH`: Path to the SQL initialization file (default: `scripts/initialize_database_schema.sql`)
- `--skip-init`: Skip SQL initialization and only insert sample data
- `--skip-sample-data`: Skip sample data insertion

### 3. Test the Database Schema

After initialization, you can test the database schema:

```bash
python scripts/test_database_schema.py
```

Options:
- `--tables TABLE_NAMES`: Comma-separated list of tables to test (default: all tables)
- `--skip-insert`: Skip INSERT tests (useful for read-only access)
- `--output FILE`: Path to save test results (default: `schema_test_results.json`)

## Initialization Process

The initialization process:

1. Creates logical schemas if they don't exist (`core`, `monitoring`, `analytics`, `experimentation`, `auth`)
2. Creates all required tables with appropriate constraints and relationships
3. Creates indexes for performance optimization
4. Sets up Row Level Security (RLS) policies
5. Adds comments to tables for documentation
6. Inserts sample data for testing

## Schema Organization

Tables are organized into logical schemas:

- **core**: `models`, `raw_parking_data`, `cleaned_parking_data`, `feature_engineered_data`, `training_data`, `predictions`
- **monitoring**: `drift_analysis`, `retraining_events`, `system_health`
- **analytics**: `business_metrics`, `location_metrics`, `metrics`
- **experimentation**: `ab_tests`, `test_variants`, `test_daily_metrics`
- **auth**: `users`, `user_roles`, `roles`

## Troubleshooting

### Common Issues:

1. **Missing Tables**:
   - Check Supabase SQL Editor logs for errors during initialization
   - Verify that execute_sql function was created successfully
   - Try running initialization script with verbose logging: `python -m scripts.execute_schema_initialization --verbose`

2. **Permission Denied**:
   - Ensure your Supabase API key has sufficient permissions
   - Check RLS policies if operations fail after initialization

3. **Duplicate Key Errors**:
   - Use `--skip-sample-data` if you're re-running the initialization but want to keep existing data

## Schema Evolution

As your application evolves, you may need to update the database schema. Create migration scripts in the `scripts/migrations/` directory and follow a similar pattern to the initialization script.

## Security Considerations

The initialization script sets up Row Level Security (RLS) policies for authenticated users. Review these policies and adjust them according to your security requirements.

The `execute_sql` function created in Supabase is marked with `SECURITY DEFINER`, which means it runs with the privileges of the user who created it. This is necessary for schema initialization but represents a potential security risk. Consider dropping this function after initialization is complete. 