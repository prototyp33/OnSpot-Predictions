# OnSpot Predictive Model Database Documentation

## Overview

This directory contains comprehensive documentation for the OnSpot Predictive Model database. The documentation is organized into multiple files, each focusing on a specific aspect of the database design and implementation.

## Contents

1. **[Database Schema](database_schema.md)** - Comprehensive overview of all tables, their columns, and relationships.

2. **[Database Relationships](database_relationships.md)** - Visual representation of relationships between tables and data flow diagrams.

3. **[Data Dictionary](data_dictionary.md)** - Detailed information about data types, constraints, and sample values.

## Database Architecture

The OnSpot Predictive Model database is designed to support a parking prediction system with the following key components:

1. **Core Data Pipeline**
   - Collection of raw parking data
   - Data cleaning and validation
   - Feature engineering
   - Model training and storage
   - Prediction generation and evaluation

2. **Monitoring & Maintenance**
   - Model drift detection
   - System health monitoring
   - Model retraining events

3. **Business Intelligence**
   - Business metrics tracking
   - Location-specific performance metrics

4. **Experimentation**
   - A/B testing framework
   - Test variants management
   - Performance metrics for variants

5. **User Management**
   - User account storage
   - Role-based access control

## Database Organization

The database tables are organized into logical groups:

### Core Tables
- `models` - Metadata about machine learning models
- `predictions` - Parking occupancy predictions
- `raw_parking_data` - Raw data from sensors and sources
- `cleaned_parking_data` - Validated and cleaned data
- `feature_engineered_data` - Data with engineered features for ML

### Monitoring Tables
- `drift_analysis` - Model drift detection records
- `retraining_events` - Model retraining history
- `system_health` - System health monitoring

### Business Intelligence Tables
- `business_metrics` - General business performance metrics
- `location_metrics` - Location-specific performance metrics

### A/B Testing Tables
- `ab_tests` - Information about A/B tests
- `test_variants` - Different variants in A/B tests
- `test_daily_metrics` - Daily performance metrics for test variants

### User Management Tables
- `users` - User account information
- `user_roles` - Role assignments for users

## Database Design Principles

The database is designed with the following principles in mind:

1. **Data Integrity** - Foreign key relationships ensure referential integrity between related tables.

2. **Auditability** - Most tables include creation timestamps and relevant metadata for audit trails.

3. **Performance** - Indexes have been added to frequently queried columns to optimize performance.

4. **Flexibility** - JSON/JSONB columns allow for storing flexible data structures where appropriate.

5. **Scalability** - The design separates concerns to allow for easy scaling of different components.

## Using This Documentation

- **For Developers**: Start with the Database Schema document to understand the overall structure, then refer to the Relationships document to understand how tables connect.

- **For Data Scientists**: The Data Dictionary provides detailed information about the data stored in each column, including formats and examples.

- **For Database Administrators**: The Relationships document will help understand the dependencies between tables for maintenance tasks.

## Maintenance

This documentation should be updated whenever:

1. New tables are added to the database
2. Existing table schemas are modified
3. Relationships between tables change
4. New use cases or patterns emerge that affect the interpretation of the data

## Additional Resources

For implementation details about how the database is used in code, refer to:

- The `supabase/` directory for Supabase-specific setup
- The `models/` directory for how models interact with the database
- The `analytics/` directory for reporting and analytical queries 