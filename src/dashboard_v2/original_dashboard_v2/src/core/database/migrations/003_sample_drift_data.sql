-- Insert sample drift analysis data
INSERT INTO drift_analysis (model_id, feature_name, drift_score, distribution_difference, timestamp)
VALUES
    ('model_1', NULL, 0.35, NULL, CURRENT_TIMESTAMP - INTERVAL '1 hour'),
    ('model_1', 'temperature', 0.42, 0.38, CURRENT_TIMESTAMP - INTERVAL '1 hour'),
    ('model_1', 'humidity', 0.28, 0.25, CURRENT_TIMESTAMP - INTERVAL '1 hour'),
    ('model_1', 'pressure', 0.15, 0.12, CURRENT_TIMESTAMP - INTERVAL '1 hour'),
    ('model_1', 'wind_speed', 0.55, 0.48, CURRENT_TIMESTAMP - INTERVAL '1 hour'),
    ('model_1', 'precipitation', 0.33, 0.30, CURRENT_TIMESTAMP - INTERVAL '1 hour'),
    ('model_1', 'cloud_cover', 0.22, 0.18, CURRENT_TIMESTAMP - INTERVAL '1 hour'),
    ('model_1', 'visibility', 0.45, 0.40, CURRENT_TIMESTAMP - INTERVAL '1 hour'),
    ('model_1', 'solar_radiation', 0.38, 0.35, CURRENT_TIMESTAMP - INTERVAL '1 hour'),
    ('model_1', 'air_quality', 0.29, 0.26, CURRENT_TIMESTAMP - INTERVAL '1 hour');

-- Insert historical data for trend analysis
INSERT INTO drift_analysis (model_id, feature_name, drift_score, distribution_difference, timestamp)
SELECT 
    'model_1',
    feature_name,
    GREATEST(0.1, LEAST(0.9, drift_score + (random() * 0.2 - 0.1))), -- Vary drift score by ±0.1
    GREATEST(0.1, LEAST(0.9, distribution_difference + (random() * 0.2 - 0.1))), -- Vary distribution difference by ±0.1
    CURRENT_TIMESTAMP - (INTERVAL '1 hour' * generate_series(2, 24))
FROM drift_analysis
WHERE feature_name IS NOT NULL
ORDER BY feature_name; 