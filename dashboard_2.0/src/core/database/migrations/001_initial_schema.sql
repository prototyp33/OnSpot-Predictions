-- Initial schema for model monitoring system

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create metrics table
CREATE TABLE model_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id VARCHAR(255) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    sample_size INTEGER NOT NULL,
    confidence_interval_lower DOUBLE PRECISION,
    confidence_interval_upper DOUBLE PRECISION,
    prediction_id VARCHAR(255),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create data quality metrics table
CREATE TABLE data_quality_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    missing_rate DOUBLE PRECISION NOT NULL,
    out_of_range_rate DOUBLE PRECISION NOT NULL,
    correlation_changes JSONB,
    distribution_metrics JSONB,
    sample_size INTEGER NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create health metrics table
CREATE TABLE health_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    prediction_latency_ms DOUBLE PRECISION NOT NULL,
    error_rate DOUBLE PRECISION NOT NULL,
    memory_usage_mb DOUBLE PRECISION,
    cpu_usage_percent DOUBLE PRECISION,
    request_count INTEGER NOT NULL,
    additional_metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_model_metrics_model_id ON model_metrics(model_id);
CREATE INDEX idx_model_metrics_timestamp ON model_metrics(timestamp);
CREATE INDEX idx_model_metrics_metric_name ON model_metrics(metric_name);
CREATE INDEX idx_model_metrics_created_at ON model_metrics(created_at);

CREATE INDEX idx_data_quality_model_id ON data_quality_metrics(model_id);
CREATE INDEX idx_data_quality_timestamp ON data_quality_metrics(timestamp);
CREATE INDEX idx_data_quality_created_at ON data_quality_metrics(created_at);

CREATE INDEX idx_health_metrics_model_id ON health_metrics(model_id);
CREATE INDEX idx_health_metrics_timestamp ON health_metrics(timestamp);
CREATE INDEX idx_health_metrics_created_at ON health_metrics(created_at);

-- Add foreign key constraints if needed
-- ALTER TABLE model_metrics ADD CONSTRAINT fk_model_id FOREIGN KEY (model_id) REFERENCES models(id);

-- Create trigger for updating updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_model_metrics_updated_at
    BEFORE UPDATE ON model_metrics
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_data_quality_metrics_updated_at
    BEFORE UPDATE ON data_quality_metrics
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_health_metrics_updated_at
    BEFORE UPDATE ON health_metrics
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column(); 