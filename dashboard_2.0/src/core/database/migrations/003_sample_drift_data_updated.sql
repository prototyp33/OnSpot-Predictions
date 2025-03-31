-- Insert sample drift analysis data with updated structure
INSERT INTO drift_analysis (
    model_id,
    feature_name,
    drift_score,
    p_value,
    mean_difference,
    std_difference,
    distribution_difference,
    new_categories,
    missing_categories,
    timestamp,
    baseline_timestamp,
    metadata
)
VALUES
    -- Overall model drift
    ('model_1', 'overall', 0.35, 0.042, 0.15, 0.08, 0.32, '[]'::jsonb, '[]'::jsonb, 
     CURRENT_TIMESTAMP, CURRENT_TIMESTAMP - INTERVAL '24 hours',
     '{"sample_size": 1000, "alert_threshold": 0.5}'::jsonb),
    
    -- Feature-specific drift scores
    ('model_1', 'temperature', 0.42, 0.031, 0.22, 0.12, 0.38, 
     '["extreme_heat"]'::jsonb, '[]'::jsonb,
     CURRENT_TIMESTAMP, CURRENT_TIMESTAMP - INTERVAL '24 hours',
     '{"unit": "celsius", "range": [-10, 45]}'::jsonb),
    
    ('model_1', 'humidity', 0.28, 0.067, 0.14, 0.09, 0.25,
     '[]'::jsonb, '[]'::jsonb,
     CURRENT_TIMESTAMP, CURRENT_TIMESTAMP - INTERVAL '24 hours',
     '{"unit": "percent", "range": [0, 100]}'::jsonb),
    
    ('model_1', 'pressure', 0.15, 0.089, 0.08, 0.05, 0.12,
     '[]'::jsonb, '[]'::jsonb,
     CURRENT_TIMESTAMP, CURRENT_TIMESTAMP - INTERVAL '24 hours',
     '{"unit": "hPa", "range": [980, 1030]}'::jsonb),
    
    ('model_1', 'wind_speed', 0.55, 0.012, 0.35, 0.18, 0.48,
     '["storm_conditions"]'::jsonb, '[]'::jsonb,
     CURRENT_TIMESTAMP, CURRENT_TIMESTAMP - INTERVAL '24 hours',
     '{"unit": "m/s", "range": [0, 50]}'::jsonb),
    
    ('model_1', 'precipitation', 0.33, 0.045, 0.18, 0.11, 0.30,
     '["heavy_rain"]'::jsonb, '[]'::jsonb,
     CURRENT_TIMESTAMP, CURRENT_TIMESTAMP - INTERVAL '24 hours',
     '{"unit": "mm", "range": [0, 100]}'::jsonb),
    
    ('model_1', 'cloud_cover', 0.22, 0.078, 0.11, 0.07, 0.18,
     '[]'::jsonb, '[]'::jsonb,
     CURRENT_TIMESTAMP, CURRENT_TIMESTAMP - INTERVAL '24 hours',
     '{"unit": "percent", "range": [0, 100]}'::jsonb),
    
    ('model_1', 'visibility', 0.45, 0.028, 0.28, 0.15, 0.40,
     '["fog_conditions"]'::jsonb, '[]'::jsonb,
     CURRENT_TIMESTAMP, CURRENT_TIMESTAMP - INTERVAL '24 hours',
     '{"unit": "km", "range": [0, 20]}'::jsonb),
    
    ('model_1', 'solar_radiation', 0.38, 0.039, 0.21, 0.13, 0.35,
     '[]'::jsonb, '[]'::jsonb,
     CURRENT_TIMESTAMP, CURRENT_TIMESTAMP - INTERVAL '24 hours',
     '{"unit": "W/m²", "range": [0, 1000]}'::jsonb),
    
    ('model_1', 'air_quality', 0.29, 0.058, 0.16, 0.10, 0.26,
     '[]'::jsonb, '[]'::jsonb,
     CURRENT_TIMESTAMP, CURRENT_TIMESTAMP - INTERVAL '24 hours',
     '{"unit": "AQI", "range": [0, 500]}'::jsonb);

-- Insert historical data for trend analysis
WITH base_features AS (
    SELECT DISTINCT 
        model_id,
        feature_name,
        drift_score,
        p_value,
        mean_difference,
        std_difference,
        distribution_difference,
        new_categories,
        missing_categories,
        metadata
    FROM drift_analysis
    WHERE feature_name != 'overall'
)
INSERT INTO drift_analysis (
    model_id,
    feature_name,
    drift_score,
    p_value,
    mean_difference,
    std_difference,
    distribution_difference,
    new_categories,
    missing_categories,
    timestamp,
    baseline_timestamp,
    metadata
)
SELECT 
    model_id,
    feature_name,
    -- Add some random variation to the metrics while keeping them within reasonable bounds
    GREATEST(0.1, LEAST(0.9, drift_score + (random() * 0.2 - 0.1))),
    GREATEST(0.001, LEAST(0.1, p_value + (random() * 0.02 - 0.01))),
    GREATEST(0.05, LEAST(0.5, mean_difference + (random() * 0.1 - 0.05))),
    GREATEST(0.02, LEAST(0.2, std_difference + (random() * 0.05 - 0.025))),
    GREATEST(0.1, LEAST(0.9, distribution_difference + (random() * 0.2 - 0.1))),
    new_categories,
    missing_categories,
    CURRENT_TIMESTAMP - (INTERVAL '1 hour' * generate_series(2, 24)),
    CURRENT_TIMESTAMP - INTERVAL '24 hours',
    metadata
FROM base_features; 