# OnSpot Predictive Model Data Dictionary

This document provides detailed information about the data types, constraints, and sample values for each table and column in the OnSpot Predictive Model database.

## Data Type Conventions

| Type | Format | Description | Example |
|------|--------|-------------|---------|
| uuid | 32-hex string | Unique identifier | `a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11` |
| text | String | Variable length text | `"Random Forest"` |
| timestamp | ISO 8601 | Date and time with timezone | `2023-10-15T14:30:00Z` |
| date | ISO 8601 | Date only | `2023-10-15` |
| time | ISO 8601 | Time only | `14:30:00` |
| numeric | Decimal | Numeric value with decimal precision | `75.5` |
| integer | Integer | Whole number | `42` |
| boolean | True/False | Boolean value | `true` |
| jsonb | JSON | Binary JSON data | `{"key": "value"}` |

## Core Tables

### models

| Column Name | Data Type | Constraints | Sample Value | Description |
|-------------|-----------|-------------|--------------|-------------|
| id | uuid | PRIMARY KEY | `a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11` | Unique identifier for the model |
| model_type | text | NOT NULL | `"random_forest"` | Type of the model |
| version | text | NOT NULL | `"1.2.0"` | Version identifier of the model |
| training_date | timestamp | NOT NULL | `2023-10-15T14:30:00Z` | Date and time when the model was trained |
| location_id | uuid | NULLABLE | `b1febc99-9c0b-4ef8-bb6d-6bb9bd380a22` | Reference to location the model is trained for |
| parameters | jsonb | NULLABLE | `{"n_estimators": 100, "max_depth": 5}` | Model hyperparameters and configuration |
| metrics | jsonb | NULLABLE | `{"accuracy": 0.85, "rmse": 0.12}` | Performance metrics of the model |
| status | text | NOT NULL | `"active"` | Current status of the model |
| created_at | timestamp | NOT NULL, DEFAULT NOW() | `2023-10-15T14:30:00Z` | Timestamp when the record was created |
| updated_at | timestamp | NOT NULL, DEFAULT NOW() | `2023-10-16T09:15:00Z` | Timestamp when the record was last updated |

### predictions

| Column Name | Data Type | Constraints | Sample Value | Description |
|-------------|-----------|-------------|--------------|-------------|
| id | uuid | PRIMARY KEY | `c2eebc99-9c0b-4ef8-bb6d-6bb9bd380a33` | Unique identifier for the prediction |
| model_id | uuid | FOREIGN KEY, NOT NULL | `a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11` | Reference to the model that generated this prediction |
| location_id | uuid | NOT NULL | `b1febc99-9c0b-4ef8-bb6d-6bb9bd380a22` | Identifier for the parking location |
| timestamp | timestamp | NOT NULL | `2023-10-16T12:00:00Z` | Date and time for which the prediction is made |
| predicted_occupancy | numeric | NOT NULL | `75.5` | Predicted parking occupancy (percentage) |
| prediction_interval_lower | numeric | NULLABLE | `70.2` | Lower bound of the prediction interval |
| prediction_interval_upper | numeric | NULLABLE | `80.8` | Upper bound of the prediction interval |
| features_used | jsonb | NULLABLE | `{"day_of_week": 1, "hour": 12, "is_holiday": false}` | Features used to generate this prediction |
| actual_occupancy | numeric | NULLABLE | `73.0` | Actual observed occupancy (updated after the fact) |
| created_at | timestamp | NOT NULL, DEFAULT NOW() | `2023-10-15T08:00:00Z` | Timestamp when the prediction was generated |

### raw_parking_data

| Column Name | Data Type | Constraints | Sample Value | Description |
|-------------|-----------|-------------|--------------|-------------|
| id | uuid | PRIMARY KEY | `d3eebc99-9c0b-4ef8-bb6d-6bb9bd380a44` | Unique identifier for the data point |
| location_id | uuid | NOT NULL | `b1febc99-9c0b-4ef8-bb6d-6bb9bd380a22` | Identifier for the parking location |
| timestamp | timestamp | NOT NULL | `2023-10-15T12:15:00Z` | Date and time when the data was collected |
| occupancy | numeric | NULLABLE | `72.0` | Raw occupancy count or percentage |
| data_source | text | NOT NULL | `"sensor"` | Source of the data |
| raw_payload | jsonb | NULLABLE | `{"sensor_id": "S123", "battery": 85, "occupancy": 72}` | Complete raw data as received |
| created_at | timestamp | NOT NULL, DEFAULT NOW() | `2023-10-15T12:16:02Z` | Timestamp when the record was created |

### cleaned_parking_data

| Column Name | Data Type | Constraints | Sample Value | Description |
|-------------|-----------|-------------|--------------|-------------|
| id | uuid | PRIMARY KEY | `e4eebc99-9c0b-4ef8-bb6d-6bb9bd380a55` | Unique identifier for the data point |
| raw_data_id | uuid | FOREIGN KEY, NOT NULL | `d3eebc99-9c0b-4ef8-bb6d-6bb9bd380a44` | Reference to the original raw data |
| location_id | uuid | NOT NULL | `b1febc99-9c0b-4ef8-bb6d-6bb9bd380a22` | Identifier for the parking location |
| timestamp | timestamp | NOT NULL | `2023-10-15T12:15:00Z` | Date and time the data represents |
| occupancy | numeric | NOT NULL | `72.0` | Cleaned and validated occupancy value |
| is_anomaly | boolean | NOT NULL, DEFAULT FALSE | `false` | Flag indicating if the data point was identified as an anomaly |
| cleaning_notes | text | NULLABLE | `"No anomalies detected"` | Notes on any cleaning operations performed |
| created_at | timestamp | NOT NULL, DEFAULT NOW() | `2023-10-15T12:20:00Z` | Timestamp when the record was created |

### feature_engineered_data

| Column Name | Data Type | Constraints | Sample Value | Description |
|-------------|-----------|-------------|--------------|-------------|
| id | uuid | PRIMARY KEY | `f5eebc99-9c0b-4ef8-bb6d-6bb9bd380a66` | Unique identifier for the data point |
| cleaned_data_id | uuid | FOREIGN KEY, NOT NULL | `e4eebc99-9c0b-4ef8-bb6d-6bb9bd380a55` | Reference to the cleaned data |
| location_id | uuid | NOT NULL | `b1febc99-9c0b-4ef8-bb6d-6bb9bd380a22` | Identifier for the parking location |
| timestamp | timestamp | NOT NULL | `2023-10-15T12:15:00Z` | Date and time the data represents |
| occupancy | numeric | NOT NULL | `72.0` | Target variable - parking occupancy |
| day_of_week | integer | NOT NULL | `1` | Day of week (0-6, where 0 is Monday) |
| hour_of_day | integer | NOT NULL | `12` | Hour of day (0-23) |
| is_weekend | boolean | NOT NULL | `false` | Whether the day is a weekend |
| is_holiday | boolean | NOT NULL | `false` | Whether the day is a public holiday |
| weather_temperature | numeric | NULLABLE | `22.5` | Temperature at the time (°C) |
| weather_precipitation | numeric | NULLABLE | `0.0` | Precipitation amount at the time (mm) |
| weather_conditions | text | NULLABLE | `"Clear"` | General weather conditions |
| events_nearby | jsonb | NULLABLE | `{"events": [{"name": "Concert", "distance_km": 1.2}]}` | Information about nearby events |
| time_features | jsonb | NULLABLE | `{"is_rush_hour": true, "is_lunch_time": true}` | Additional derived time features |
| lag_features | jsonb | NULLABLE | `{"occupancy_1h_ago": 68.5, "occupancy_1d_ago": 70.2}` | Lagged values of occupancy |
| created_at | timestamp | NOT NULL, DEFAULT NOW() | `2023-10-15T12:30:00Z` | Timestamp when the record was created |

## Monitoring Tables

### drift_analysis

| Column Name | Data Type | Constraints | Sample Value | Description |
|-------------|-----------|-------------|--------------|-------------|
| id | uuid | PRIMARY KEY | `g6eebc99-9c0b-4ef8-bb6d-6bb9bd380a77` | Unique identifier for the drift analysis |
| model_id | uuid | FOREIGN KEY, NOT NULL | `a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11` | Reference to the model being analyzed |
| timestamp | timestamp | NOT NULL | `2023-10-20T08:00:00Z` | When the drift analysis was performed |
| baseline_timestamp | timestamp | NOT NULL | `2023-10-01T00:00:00Z` | Reference timestamp for the baseline distribution |
| feature_drift_scores | jsonb | NULLABLE | `{"day_of_week": 0.02, "weather_temperature": 0.15}` | Drift scores for individual features |
| overall_drift_score | numeric | NOT NULL | `0.08` | Aggregated drift score |
| drift_detected | boolean | NOT NULL | `false` | Whether significant drift was detected |
| recommended_action | text | NULLABLE | `"No action needed"` | Recommended action based on drift analysis |
| created_at | timestamp | NOT NULL, DEFAULT NOW() | `2023-10-20T08:05:00Z` | Timestamp when the record was created |

### retraining_events

| Column Name | Data Type | Constraints | Sample Value | Description |
|-------------|-----------|-------------|--------------|-------------|
| id | uuid | PRIMARY KEY | `h7eebc99-9c0b-4ef8-bb6d-6bb9bd380a88` | Unique identifier for the retraining event |
| model_id | uuid | FOREIGN KEY, NOT NULL | `a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11` | Reference to the original model |
| new_model_id | uuid | FOREIGN KEY, NULLABLE | `i8eebc99-9c0b-4ef8-bb6d-6bb9bd380a99` | Reference to the newly trained model |
| timestamp | timestamp | NOT NULL | `2023-11-01T09:30:00Z` | When the retraining was performed |
| trigger | text | NOT NULL | `"scheduled"` | What triggered the retraining |
| success | boolean | NOT NULL | `true` | Whether the retraining was successful |
| performance_change | numeric | NULLABLE | `2.5` | Percentage change in model performance |
| notes | text | NULLABLE | `"Slight improvement in prediction accuracy"` | Additional notes about the retraining |
| created_at | timestamp | NOT NULL, DEFAULT NOW() | `2023-11-01T10:00:00Z` | Timestamp when the record was created |

### system_health

| Column Name | Data Type | Constraints | Sample Value | Description |
|-------------|-----------|-------------|--------------|-------------|
| id | uuid | PRIMARY KEY | `j9eebc99-9c0b-4ef8-bb6d-6bb9bd380aaa` | Unique identifier for the health record |
| timestamp | timestamp | NOT NULL | `2023-10-15T08:00:00Z` | When the health check was performed |
| component | text | NOT NULL | `"prediction_service"` | System component being monitored |
| status | text | NOT NULL | `"healthy"` | Status of the component |
| response_time_ms | integer | NULLABLE | `120` | Response time in milliseconds |
| error_count | integer | NOT NULL, DEFAULT 0 | `0` | Number of errors encountered |
| warning_count | integer | NOT NULL, DEFAULT 0 | `2` | Number of warnings encountered |
| metrics | jsonb | NULLABLE | `{"cpu_usage": 45.2, "memory_usage": 512}` | Additional performance metrics |
| notes | text | NULLABLE | `"Normal operation"` | Additional context or notes |
| created_at | timestamp | NOT NULL, DEFAULT NOW() | `2023-10-15T08:01:00Z` | Timestamp when the record was created |

## Business Intelligence Tables

### business_metrics

| Column Name | Data Type | Constraints | Sample Value | Description |
|-------------|-----------|-------------|--------------|-------------|
| id | uuid | PRIMARY KEY | `k0eebc99-9c0b-4ef8-bb6d-6bb9bd380abb` | Unique identifier for the metric |
| timestamp | timestamp | NOT NULL | `2023-10-15T00:00:00Z` | Date and time the metric represents |
| metric_name | text | NOT NULL | `"total_predictions"` | Name of the business metric |
| metric_value | numeric | NOT NULL | `15023` | Value of the business metric |
| category | text | NOT NULL | `"usage"` | Business category the metric belongs to |
| dimensions | jsonb | NULLABLE | `{"platform": "mobile", "user_type": "premium"}` | Additional dimensions for multi-dimensional analysis |
| notes | text | NULLABLE | `"Record high usage"` | Any additional context or notes |
| created_at | timestamp | NOT NULL, DEFAULT NOW() | `2023-10-16T08:00:00Z` | Timestamp when the record was created |

### location_metrics

| Column Name | Data Type | Constraints | Sample Value | Description |
|-------------|-----------|-------------|--------------|-------------|
| id | uuid | PRIMARY KEY | `l1eebc99-9c0b-4ef8-bb6d-6bb9bd380acc` | Unique identifier for the location metric |
| location_id | uuid | NOT NULL | `b1febc99-9c0b-4ef8-bb6d-6bb9bd380a22` | Identifier for the parking location |
| date | date | NOT NULL | `2023-10-15` | Date the metrics represent |
| average_occupancy | numeric | NOT NULL | `68.5` | Average occupancy for the day |
| peak_occupancy | numeric | NOT NULL | `92.3` | Peak occupancy for the day |
| peak_time | time | NULLABLE | `17:30:00` | Time when peak occupancy occurred |
| total_vehicles | integer | NULLABLE | `450` | Estimated total vehicles using the location |
| prediction_accuracy | numeric | NULLABLE | `94.2` | Accuracy of predictions for this location and day |
| revenue | numeric | NULLABLE | `1250.50` | Estimated or actual revenue generated |
| created_at | timestamp | NOT NULL, DEFAULT NOW() | `2023-10-16T08:00:00Z` | Timestamp when the record was created |

## A/B Testing Tables

### ab_tests

| Column Name | Data Type | Constraints | Sample Value | Description |
|-------------|-----------|-------------|--------------|-------------|
| id | uuid | PRIMARY KEY | `m2eebc99-9c0b-4ef8-bb6d-6bb9bd380add` | Unique identifier for the A/B test |
| name | text | NOT NULL | `"prediction_algorithm_test"` | Name of the A/B test |
| description | text | NULLABLE | `"Testing new prediction algorithm against baseline"` | Detailed description of what is being tested |
| status | text | NOT NULL | `"running"` | Current status of the test |
| start_date | date | NOT NULL | `2023-10-01` | Date when the test started |
| end_date | date | NULLABLE | `2023-10-31` | Date when the test ended or is scheduled to end |
| hypothesis | text | NULLABLE | `"The new algorithm will improve prediction accuracy by 5%"` | The hypothesis being tested |
| success_criteria | jsonb | NULLABLE | `{"min_improvement": 5, "confidence_level": 95}` | Criteria for determining test success |
| created_at | timestamp | NOT NULL, DEFAULT NOW() | `2023-09-25T15:00:00Z` | Timestamp when the record was created |
| updated_at | timestamp | NOT NULL, DEFAULT NOW() | `2023-09-25T15:00:00Z` | Timestamp when the record was last updated |

### test_variants

| Column Name | Data Type | Constraints | Sample Value | Description |
|-------------|-----------|-------------|--------------|-------------|
| id | uuid | PRIMARY KEY | `n3eebc99-9c0b-4ef8-bb6d-6bb9bd380aee` | Unique identifier for the variant |
| test_id | uuid | FOREIGN KEY, NOT NULL | `m2eebc99-9c0b-4ef8-bb6d-6bb9bd380add` | Reference to the A/B test |
| name | text | NOT NULL | `"Algorithm B"` | Name of the variant |
| description | text | NULLABLE | `"New prediction algorithm with enhanced features"` | Detailed description of the variant |
| configuration | jsonb | NULLABLE | `{"algorithm": "gradient_boosting", "features": ["day_of_week", "weather"]}` | Technical configuration specific to this variant |
| allocation_percentage | numeric | NOT NULL | `50.0` | Percentage of traffic allocated to this variant |
| created_at | timestamp | NOT NULL, DEFAULT NOW() | `2023-09-25T15:05:00Z` | Timestamp when the record was created |
| updated_at | timestamp | NOT NULL, DEFAULT NOW() | `2023-09-25T15:05:00Z` | Timestamp when the record was last updated |

### test_daily_metrics

| Column Name | Data Type | Constraints | Sample Value | Description |
|-------------|-----------|-------------|--------------|-------------|
| id | uuid | PRIMARY KEY | `o4eebc99-9c0b-4ef8-bb6d-6bb9bd380aff` | Unique identifier for the daily metric |
| variant_id | uuid | FOREIGN KEY, NOT NULL | `n3eebc99-9c0b-4ef8-bb6d-6bb9bd380aee` | Reference to the test variant |
| date | date | NOT NULL | `2023-10-15` | Date the metrics represent |
| impressions | integer | NOT NULL | `5280` | Number of impressions or exposures for the variant |
| clicks | integer | NOT NULL | `423` | Number of clicks or interactions |
| conversions | integer | NOT NULL | `86` | Number of conversions or desired actions |
| revenue | numeric | NULLABLE | `256.75` | Revenue generated by this variant on this day |
| custom_metrics | jsonb | NULLABLE | `{"avg_prediction_error": 0.04, "user_satisfaction": 4.2}` | Additional custom metrics specific to the test |
| created_at | timestamp | NOT NULL, DEFAULT NOW() | `2023-10-16T08:00:00Z` | Timestamp when the record was created |

## User Management Tables

### users

| Column Name | Data Type | Constraints | Sample Value | Description |
|-------------|-----------|-------------|--------------|-------------|
| id | uuid | PRIMARY KEY | `p5eebc99-9c0b-4ef8-bb6d-6bb9bd380baa` | Unique identifier for the user |
| email | text | NOT NULL, UNIQUE | `"user@example.com"` | User's email address |
| first_name | text | NOT NULL | `"Jane"` | User's first name |
| last_name | text | NOT NULL | `"Doe"` | User's last name |
| created_at | timestamp | NOT NULL, DEFAULT NOW() | `2023-09-01T10:00:00Z` | Timestamp when the user account was created |
| updated_at | timestamp | NOT NULL, DEFAULT NOW() | `2023-09-01T10:00:00Z` | Timestamp when the user account was last updated |

### user_roles

| Column Name | Data Type | Constraints | Sample Value | Description |
|-------------|-----------|-------------|--------------|-------------|
| user_id | uuid | FOREIGN KEY, NOT NULL | `p5eebc99-9c0b-4ef8-bb6d-6bb9bd380baa` | Reference to the user |
| role | text | NOT NULL | `"admin"` | Role assigned to the user |
| assigned_at | timestamp | NOT NULL, DEFAULT NOW() | `2023-09-01T10:05:00Z` | When the role was assigned to the user |
| assigned_by | uuid | FOREIGN KEY, NULLABLE | `q6eebc99-9c0b-4ef8-bb6d-6bb9bd380bbb` | User who assigned this role |
| created_at | timestamp | NOT NULL, DEFAULT NOW() | `2023-09-01T10:05:00Z` | Timestamp when the record was created | 