# Database Schema Organization Proposal

This document outlines a proposal to organize the OnSpot Predictive Model database tables into logical schemas for better structure, management, and security.

## Current Structure

Currently, all tables reside in the default `public` schema in PostgreSQL/Supabase. While this works for smaller applications, as the system grows, a more organized approach will provide several benefits:

- **Improved organization** - Related tables are grouped together
- **Better access control** - Permissions can be managed at the schema level
- **Clearer ownership** - Different teams can own different schemas
- **Simplified queries** - Context is provided by schema names
- **Better documentation** - Structure is self-documenting

## Proposed Schema Organization

We propose organizing the tables into the following schemas:

### 1. `core` Schema

Tables related to the core data pipeline and model management:

| Table Name | Current Schema | New Schema |
|------------|---------------|------------|
| models | public | core |
| predictions | public | core |
| raw_parking_data | public | core |
| cleaned_parking_data | public | core |
| feature_engineered_data | public | core |

### 2. `monitoring` Schema

Tables related to system monitoring and model lifecycle:

| Table Name | Current Schema | New Schema |
|------------|---------------|------------|
| drift_analysis | public | monitoring |
| retraining_events | public | monitoring |
| system_health | public | monitoring |

### 3. `analytics` Schema

Tables used for business intelligence and reporting:

| Table Name | Current Schema | New Schema |
|------------|---------------|------------|
| business_metrics | public | analytics |
| location_metrics | public | analytics |

### 4. `experimentation` Schema

Tables used for A/B testing and experimental features:

| Table Name | Current Schema | New Schema |
|------------|---------------|------------|
| ab_tests | public | experimentation |
| test_variants | public | experimentation |
| test_daily_metrics | public | experimentation |

### 5. `auth` Schema

Tables used for user management:

| Table Name | Current Schema | New Schema |
|------------|---------------|------------|
| users | public | auth |
| user_roles | public | auth |

## Implementation Plan

The migration to the new schema structure should be performed in a controlled manner to minimize disruption. Here's a proposed implementation plan:

### 1. Schema Creation

```sql
-- Create new schemas
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS experimentation;
CREATE SCHEMA IF NOT EXISTS auth;

-- Set appropriate permissions
GRANT USAGE ON SCHEMA core TO authenticated;
GRANT USAGE ON SCHEMA monitoring TO authenticated;
GRANT USAGE ON SCHEMA analytics TO authenticated;
GRANT USAGE ON SCHEMA experimentation TO authenticated;
GRANT USAGE ON SCHEMA auth TO authenticated;
```

### 2. Table Migration

For each table, we'll need to:
1. Create the new table in the appropriate schema
2. Copy data from the old table to the new table
3. Update foreign key references
4. Drop the old table (after verification)

Example for the `models` table:

```sql
-- Create table in new schema
CREATE TABLE core.models (
    LIKE public.models INCLUDING ALL
);

-- Copy data
INSERT INTO core.models
SELECT * FROM public.models;

-- Verify data
SELECT COUNT(*) FROM public.models;
SELECT COUNT(*) FROM core.models;

-- Update references in other tables
-- (This would involve updating foreign key references in related tables)

-- When ready, drop the old table
DROP TABLE public.models;
```

### 3. Foreign Key Updates

The most complex part of the migration will be updating foreign key references. Since tables will now be in different schemas, all foreign key constraints will need to be updated.

Example for updating a foreign key in `predictions`:

```sql
-- First, drop the existing foreign key constraint
ALTER TABLE core.predictions DROP CONSTRAINT predictions_model_id_fkey;

-- Then add it back, pointing to the table in the core schema
ALTER TABLE core.predictions ADD CONSTRAINT predictions_model_id_fkey 
    FOREIGN KEY (model_id) REFERENCES core.models(id);
```

### 4. View Creation

To minimize disruption to existing applications, we can create views in the public schema that point to the new tables:

```sql
CREATE OR REPLACE VIEW public.models AS
    SELECT * FROM core.models;
    
CREATE OR REPLACE VIEW public.predictions AS
    SELECT * FROM core.predictions;
    
-- ... and so on for all tables
```

### 5. Code Updates

Application code that directly references the tables will need to be updated to use the new schema qualifiers. For example:

```python
# Old query
supabase.table('models').select('*').execute()

# New query
supabase.from_('core.models').select('*').execute()
```

## Search Path Configuration

To simplify queries, we can configure the PostgreSQL search path to include our new schemas:

```sql
-- Set search path for the database
ALTER DATABASE onspot_predictive_model SET search_path TO "$user", public, core, monitoring, analytics, experimentation, auth;
```

This allows queries to reference tables without fully qualifying the schema name, while still providing the organization benefits.

## Benefits of the New Structure

The proposed schema organization provides several benefits:

1. **Logical Grouping**: Tables are grouped by their function in the system
2. **Improved Security**: Access control can be managed at the schema level
3. **Better Maintenance**: Database administrators can focus on specific schemas for maintenance tasks
4. **Enhanced Documentation**: The schema structure itself documents the system architecture
5. **Team Ownership**: Different teams can be responsible for different schemas
6. **Query Context**: Queries become more self-documenting through schema qualification

## Considerations and Risks

Some considerations and risks to be aware of:

1. **Migration Complexity**: Moving tables between schemas requires careful planning and execution
2. **Application Impact**: Any code that directly references tables will need updates
3. **Permission Management**: New permissions will need to be set up for the new schemas
4. **Testing Required**: Thorough testing is needed to ensure all functionality works after migration

## Conclusion

Organizing the database into logical schemas will provide significant long-term benefits for the OnSpot Predictive Model system. While there is some short-term complexity in the migration, the improved structure will make the system more maintainable, secure, and understandable for all stakeholders. 