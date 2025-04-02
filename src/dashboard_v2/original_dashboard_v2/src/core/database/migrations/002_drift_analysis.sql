-- Add drift analysis table
CREATE TABLE drift_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id VARCHAR(255) NOT NULL,
    feature_name VARCHAR(255),
    drift_score DOUBLE PRECISION NOT NULL,
    distribution_difference DOUBLE PRECISION,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_drift_analysis_model_id ON drift_analysis(model_id);
CREATE INDEX idx_drift_analysis_timestamp ON drift_analysis(timestamp);
CREATE INDEX idx_drift_analysis_feature_name ON drift_analysis(feature_name);
CREATE INDEX idx_drift_analysis_drift_score ON drift_analysis(drift_score);

-- Add trigger for updating updated_at
CREATE TRIGGER update_drift_analysis_updated_at
    BEFORE UPDATE ON drift_analysis
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column(); 