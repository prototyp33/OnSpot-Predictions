# OnSpot Predictive Model - Database Schema Migration

This README describes the process for migrating the database from a flat structure with all tables in the `public` schema to a more organized structure with logical schemas.

## Overview

The schema migration reorganizes the database tables into the following logical schemas:

1. `core` - Core data pipeline tables (models, predictions, etc.)
2. `monitoring` - System monitoring and model lifecycle tables
3. `analytics` - Business intelligence and reporting tables
4. `experimentation` - A/B testing tables
5. `auth` - User management tables

## Security and Performance Enhancements

The migration includes important security and performance improvements:

1. **Row Level Security (RLS)** is enabled on all tables to ensure proper access control
2. **Indexes on foreign key columns** are created to improve join performance
3. **Additional performance indexes** on frequently queried columns and common filtering patterns

## Prerequisites

- Python 3.7+
- Supabase access credentials
- Required Python packages:
  - supabase
  - python-dotenv

Install the required packages with:

```bash
pip install supabase python-dotenv
```

## Migration Options

You have two options for executing the migration:

### Option 1: Using the Python Script

1. Ensure your `.env` file contains the Supabase credentials:

```
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-service-role-key
```

2. Run the migration script:

```bash
python scripts/apply_schema_migration.py
```

The script will:
- Connect to Supabase
- Load the migration SQL file
- Execute the migration
- Log the progress and results

### Option 2: Manual Execution in Supabase SQL Editor

1. Open the Supabase dashboard
2. Navigate to the SQL Editor
3. Open the `scripts/migrate_schemas.sql` file from this repository
4. Copy the SQL contents
5. Paste into the SQL Editor and execute

## What the Migration Does

The migration script:

1. Creates new schemas (`core`, `monitoring`, `analytics`, `experimentation`, `auth`)
2. For each table:
   - If the table exists in the `public` schema, it copies it to the new schema
   - If the table doesn't exist, it creates it with the proper structure in the new schema
3. Sets up foreign key constraints between tables
4. Creates views in the `public` schema for backward compatibility
5. Sets appropriate permissions on the schemas and tables
6. Enables Row Level Security (RLS) on all tables with appropriate policies
7. Creates indexes on foreign key columns and commonly queried fields

## Row Level Security Policies

The migration sets up the following RLS policies:

- **Core tables**: Authenticated users can read and insert data
- **Monitoring tables**: Authenticated users can read and insert data
- **Analytics tables**: Authenticated users can read, insert, and update data
- **Experimentation tables**: Authenticated users can read, insert, and update data
- **Auth tables**: Users can only read and update their own data, service role has full access

## Post-Migration Steps

After the migration is complete:

1. Update application code to use the new schema-qualified table names
   - Use `python scripts/update_application_code.py` to automate this process
   - See `documentation/schema_update_guide.md` for detailed guidance

2. Test the application thoroughly to ensure everything works correctly

3. Once verified, you can optionally drop the original tables from the `public` schema
   (the migration script doesn't do this automatically for safety)

## Troubleshooting

If you encounter issues during migration:

1. Check the Supabase logs for detailed error messages
2. Verify that you have appropriate permissions to create schemas and tables
3. If a specific table fails to migrate, you may need to adjust its structure or constraints
4. For RLS policy errors, ensure the auth functions are available in your Supabase instance

For detailed guidance on updating your code, see `documentation/schema_update_guide.md`. 