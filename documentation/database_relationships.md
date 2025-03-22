# Database Relationship Diagram

This document describes the relationships between tables in the OnSpot Predictive Model database.

## Entity Relationship Diagram

```
+------------------+       +-------------------+       +------------------+
|      models      |------>|    predictions    |       | raw_parking_data |
+------------------+       +-------------------+       +------------------+
| id (PK)          |       | id (PK)           |       | id (PK)          |
| model_type       |       | model_id (FK)     |       | location_id      |
| version          |       | location_id       |       | timestamp        |
| training_date    |       | timestamp         |       | occupancy        |
| location_id      |       | predicted_occupancy|       | data_source     |
| parameters       |       | actual_occupancy  |       | raw_payload      |
| metrics          |       | created_at        |       | created_at       |
| status           |       +-------------------+       +------------------+
| created_at       |                                          |
| updated_at       |                                          |
+------------------+                                          |
        |                                                     |
        |              +----------------------+               |
        |              |   drift_analysis     |               v
        +------------->+----------------------+      +------------------+
        |              | id (PK)              |      | cleaned_parking_data|
        |              | model_id (FK)        |      +------------------+
        |              | timestamp            |      | id (PK)          |
        |              | baseline_timestamp   |      | raw_data_id (FK) |
        |              | feature_drift_scores |      | location_id      |
        |              | overall_drift_score  |      | timestamp        |
        |              | drift_detected       |      | occupancy        |
        |              | recommended_action   |      | is_anomaly       |
        |              | created_at           |      | cleaning_notes   |
        |              +----------------------+      | created_at       |
        |                                            +------------------+
        |                                                    |
        |              +----------------------+              |
        |              |  retraining_events   |              v
        +------------->+----------------------+     +---------------------+
                       | id (PK)              |     | feature_engineered_data|
                       | model_id (FK)        |     +---------------------+
                       | new_model_id (FK)    |     | id (PK)             |
                       | timestamp            |     | cleaned_data_id (FK)|
                       | trigger              |     | location_id         |
                       | success              |     | timestamp           |
                       | performance_change   |     | occupancy           |
                       | notes                |     | day_of_week         |
                       | created_at           |     | hour_of_day         |
                       +----------------------+     | weather_features    |
                                                    | etc...              |
                                                    | created_at          |
                                                    +---------------------+

+------------------+       +-------------------+
|     ab_tests     |------>|   test_variants   |
+------------------+       +-------------------+
| id (PK)          |       | id (PK)           |
| name             |       | test_id (FK)      |
| description      |       | name              |
| status           |       | description       |
| start_date       |       | configuration     |
| end_date         |       | allocation_percentage|
| hypothesis       |       | created_at        |
| success_criteria |       | updated_at        |
| created_at       |       +-------------------+
| updated_at       |               |
+------------------+               |
                                   v
                          +-------------------+
                          | test_daily_metrics|
                          +-------------------+
                          | id (PK)           |
                          | variant_id (FK)   |
                          | date              |
                          | impressions       |
                          | clicks            |
                          | conversions       |
                          | revenue           |
                          | custom_metrics    |
                          | created_at        |
                          +-------------------+

+------------------+       +-------------------+
|      users       |------>|    user_roles     |
+------------------+       +-------------------+
| id (PK)          |       | user_id (FK)      |
| email            |       | role              |
| first_name       |       | assigned_at       |
| last_name        |       | assigned_by (FK)  |
| created_at       |       | created_at        |
| updated_at       |       +-------------------+
+------------------+

+------------------+       +-------------------+
| business_metrics |       | location_metrics  |
+------------------+       +-------------------+
| id (PK)          |       | id (PK)           |
| timestamp        |       | location_id       |
| metric_name      |       | date              |
| metric_value     |       | average_occupancy |
| category         |       | peak_occupancy    |
| dimensions       |       | peak_time         |
| notes            |       | total_vehicles    |
| created_at       |       | prediction_accuracy|
+------------------+       | revenue           |
                           | created_at        |
                           +-------------------+

+------------------+
|  system_health   |
+------------------+
| id (PK)          |
| timestamp        |
| component        |
| status           |
| response_time_ms |
| error_count      |
| warning_count    |
| metrics          |
| notes            |
| created_at       |
+------------------+
```

## Key Relationships

### Core Data Flow

1. **Data Collection & Processing**:
   - `raw_parking_data` → `cleaned_parking_data` → `feature_engineered_data`
   - Raw data is collected, cleaned, and then features are engineered for model training

2. **Prediction Pipeline**:
   - `models` generate → `predictions`
   - Trained models produce predictions which are stored and evaluated

3. **Model Lifecycle Management**:
   - `models` are monitored through → `drift_analysis`
   - When drift is detected, `retraining_events` occur which may produce new `models`

### User Management

- `users` are assigned `user_roles` for access control
- `user_roles` may reference the user who assigned the role (`assigned_by`)

### A/B Testing

- `ab_tests` contain multiple → `test_variants`
- `test_variants` generate performance data in → `test_daily_metrics`

## Foreign Key Constraints

| Table | Column | References | Purpose |
|-------|--------|------------|---------|
| predictions | model_id | models.id | Link predictions to their source model |
| drift_analysis | model_id | models.id | Associate drift analysis with a model |
| retraining_events | model_id | models.id | Link retraining event to original model |
| retraining_events | new_model_id | models.id | Link to newly created model (if successful) |
| cleaned_parking_data | raw_data_id | raw_parking_data.id | Track data lineage from raw to cleaned |
| feature_engineered_data | cleaned_data_id | cleaned_parking_data.id | Track data lineage from cleaned to featured |
| test_variants | test_id | ab_tests.id | Associate variants with their parent test |
| test_daily_metrics | variant_id | test_variants.id | Associate metrics with their variant |
| user_roles | user_id | users.id | Link roles to users |
| user_roles | assigned_by | users.id | Track which user assigned the role |

## Data Flow Diagram

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Raw Data     │     │ Cleaned Data  │     │ Feature       │     │ Machine       │
│  Collection   │────>│ Processing    │────>│ Engineering   │────>│ Learning      │
└───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
       │                                                                   │
       │                                                                   │
       v                                                                   v
┌───────────────┐                                               ┌───────────────┐
│ Raw Parking   │                                               │ Model         │
│ Data Storage  │                                               │ Training      │
└───────────────┘                                               └───────────────┘
                                                                       │
┌───────────────┐     ┌───────────────┐                               │
│ Metrics &     │     │ Visualization │                               v
│ Reporting     │<────│ & Dashboard   │<─────────────┐        ┌───────────────┐
└───────────────┘     └───────────────┘              │        │ Model         │
       ^                                             │        │ Storage       │
       │                                             │        └───────────────┘
       │                                             │                │
┌───────────────┐                           ┌───────────────┐        │
│ A/B Test      │                           │ Predictions   │<───────┘
│ Evaluation    │                           │ Generation    │
└───────────────┘                           └───────────────┘
       ^                                             │
       │                                             v
┌───────────────┐                           ┌───────────────┐
│ Test Variants │                           │ Drift         │
│ Tracking      │                           │ Analysis      │
└───────────────┘                           └───────────────┘
                                                     │
                                                     v
                                            ┌───────────────┐
                                            │ Retraining    │
                                            │ Events        │
                                            └───────────────┘
``` 