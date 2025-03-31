# Supabase Monitoring System

This document describes the Supabase monitoring system implemented for the OnSpot Predictive Model project. The monitoring system provides comprehensive visibility into Supabase database performance, health metrics, and schema organization.

## Overview

The monitoring system consists of several components that work together to provide a complete view of Supabase health and performance:

1. **Core Monitoring** - Tracks database operations, response times, and errors
2. **Metrics Extension** - Integrates with external systems and provides health analysis
3. **Dashboard** - Web-based UI for viewing performance metrics and health status
4. **System Health** - Collects and tracks overall system performance metrics
5. **Schema Migration** - Tools for organizing tables into logical schemas

## Installation

### Requirements

The monitoring system has the following dependencies:

```
# Core dependencies
psycopg2-binary
python-dotenv
requests

# Optional dependencies
fastapi
uvicorn
jinja2
psutil
```

To install all dependencies:

```bash
pip install psycopg2-binary python-dotenv requests fastapi uvicorn jinja2 psutil
```

## Usage

### Monitoring Supabase Operations

The monitoring system can be used to track Supabase operations by decorating functions that interact with the database:

```python
from scripts.supabase_monitor import monitor_select, monitor_insert

class UserRepository:
    @monitor_select
    def get_user(self, user_id):
        # Database operation to get user
        pass
    
    @monitor_insert
    def create_user(self, user_data):
        # Database operation to create user
        pass
```

### Starting the Dashboard

To start the monitoring dashboard:

```python
from scripts.supabase_dashboard import SupabaseDashboard

dashboard = SupabaseDashboard()
dashboard.start(open_browser=True)
```

### Running System Health Monitoring

To collect system health metrics:

```bash
# Run once and exit
python scripts/system_health.py

# Run as a daemon
python scripts/system_health.py --daemon

# Generate a report
python scripts/system_health.py --report --days 7 --out health_report.json
```

### Managing Schema Migration

To organize tables into logical schemas:

```bash
# Dry run (no changes)
python scripts/execute_schema_migration.py --dry-run

# Execute migration
python scripts/execute_schema_migration.py

# Update application code to use new schemas
python scripts/update_application_code.py
```

## Components

### Core Monitoring (`scripts/supabase_monitor.py`)

This module provides:

- Query performance tracking
- Error rate monitoring
- Operation counts by type
- Latency metrics (average, P95, P99)

### Metrics Extension (`scripts/supabase_metrics_extension.py`)

This module provides:

- Health checks and alerting
- Integration with external monitoring systems
- Historical metric collection and analysis

### Dashboard (`scripts/supabase_dashboard.py`)

This module provides:

- Web-based UI for monitoring Supabase
- Real-time charts and metrics
- Query history and error visibility

### System Health (`scripts/system_health.py`)

This module provides:

- Overall system performance monitoring
- API endpoint health checks
- Resource usage tracking (CPU, memory, disk)
- Health reporting and alerting

### Schema Migration (`scripts/execute_schema_migration.py`)

This module provides:

- Execution of schema migration SQL
- Transaction handling and rollback on errors
- Detailed reporting of migration results

## Directory Structure

```
scripts/
├── supabase_monitor.py          # Core monitoring
├── supabase_metrics_extension.py # Metrics extension
├── supabase_dashboard.py        # Dashboard
├── supabase_setup.py            # Setup and configuration
├── system_health.py             # System health monitoring
├── execute_schema_migration.py  # Schema migration execution
├── migrate_schemas.sql          # SQL migration file
└── update_application_code.py   # Code update for schema migration
```

## Schema Organization

The monitoring system supports the organization of database tables into logical schemas for better structure, security, and maintenance:

- **Core Schema** (`core`) - Models, predictions, and parking data
- **Monitoring Schema** (`monitoring`) - Drift analysis and retraining events
- **Analytics Schema** (`analytics`) - Business and location metrics
- **Experimentation Schema** (`experimentation`) - A/B tests and variants
- **Auth Schema** (`auth`) - Users and roles

## Configuration

Configuration is handled via environment variables or configuration files:

### Environment Variables

```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SUPABASE_DB_HOST=your_db_host
SUPABASE_DB_NAME=your_db_name
SUPABASE_DB_USER=your_db_user
SUPABASE_DB_PASSWORD=your_db_password
SUPABASE_DB_PORT=your_db_port
```

### Configuration Files

Custom configuration can be provided via JSON files:

```json
{
  "collection_interval": 300,
  "retention_days": 30,
  "metrics_file": "system_health_metrics.json",
  "thresholds": {
    "cpu_percent": 80,
    "memory_percent": 85,
    "disk_percent": 90,
    "api_response_time": 2000
  }
}
```

## Extending the System

The monitoring system is designed to be extensible:

1. **Custom Metrics** - Add new metrics by extending the monitoring classes
2. **Additional Dashboards** - Create specialized dashboards for specific monitoring needs
3. **Integration** - Integrate with external monitoring systems via the metrics extension

## Troubleshooting

Common issues and solutions:

1. **Missing dependencies** - Ensure all required packages are installed
2. **Database connection errors** - Check environment variables and database credentials
3. **Dashboard access issues** - Ensure the port is not in use and firewalls allow access

## License

This monitoring system is part of the OnSpot Predictive Model project and is subject to its licensing terms. 