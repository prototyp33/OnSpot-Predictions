-- Migration file for adding monitoring tables
-- Extends the existing schema with tables for drift analysis, business metrics, 
-- system health monitoring, A/B testing, and user management

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create drift analysis table to track feature drift over time
CREATE TABLE IF NOT EXISTS drift_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id VARCHAR(255) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    drift_score DOUBLE PRECISION NOT NULL,
    p_value DOUBLE PRECISION,
    mean_difference DOUBLE PRECISION,
    std_difference DOUBLE PRECISION,
    distribution_difference DOUBLE PRECISION,
    new_categories JSONB,
    missing_categories JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    baseline_timestamp TIMESTAMP WITH TIME ZONE,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create retraining events table to track model retraining
CREATE TABLE IF NOT EXISTS retraining_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    reason VARCHAR(255) NOT NULL,
    success BOOLEAN NOT NULL DEFAULT FALSE,
    metrics_before JSONB,
    metrics_after JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create business metrics table to track financial and operational impacts
CREATE TABLE IF NOT EXISTS business_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    category VARCHAR(50) NOT NULL,
    location_id VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create location metrics table for detailed location-based tracking
CREATE TABLE IF NOT EXISTS location_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    location_id VARCHAR(100) NOT NULL,
    date DATE NOT NULL,
    occupancy_accuracy DOUBLE PRECISION,
    utilization_rate DOUBLE PRECISION,
    revenue DOUBLE PRECISION,
    opportunity_cost DOUBLE PRECISION,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create system health table for monitoring infrastructure
CREATE TABLE IF NOT EXISTS system_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    component VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    metrics JSONB,
    alert_level VARCHAR(20),
    message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create tables for A/B testing
CREATE TABLE IF NOT EXISTS ab_tests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    start_date TIMESTAMP WITH TIME ZONE,
    end_date TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) NOT NULL,
    winning_variant VARCHAR(50),
    significance_level DOUBLE PRECISION,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS test_variants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    test_id UUID REFERENCES ab_tests(id),
    name VARCHAR(50) NOT NULL,
    description TEXT,
    traffic_percentage INTEGER,
    conversion_count INTEGER DEFAULT 0,
    total_traffic INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS test_daily_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    test_id UUID REFERENCES ab_tests(id),
    variant_id UUID REFERENCES test_variants(id),
    date DATE NOT NULL,
    conversion_rate DOUBLE PRECISION,
    traffic_count INTEGER,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create tables for user management (if not using auth schema already)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS roles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(50) UNIQUE NOT NULL,
    permissions JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_roles (
    user_id UUID REFERENCES users(id),
    role_id UUID REFERENCES roles(id),
    PRIMARY KEY (user_id, role_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create necessary indexes for better query performance
-- Drift analysis indexes
CREATE INDEX IF NOT EXISTS idx_drift_model_id ON drift_analysis(model_id);
CREATE INDEX IF NOT EXISTS idx_drift_feature ON drift_analysis(feature_name);
CREATE INDEX IF NOT EXISTS idx_drift_timestamp ON drift_analysis(timestamp);

-- Retraining events indexes
CREATE INDEX IF NOT EXISTS idx_retraining_model_id ON retraining_events(model_id);
CREATE INDEX IF NOT EXISTS idx_retraining_timestamp ON retraining_events(timestamp);

-- Business metrics indexes
CREATE INDEX IF NOT EXISTS idx_business_metric_name ON business_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_business_category ON business_metrics(category);
CREATE INDEX IF NOT EXISTS idx_business_timestamp ON business_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_business_location ON business_metrics(location_id);

-- Location metrics indexes
CREATE INDEX IF NOT EXISTS idx_location_metrics_id ON location_metrics(location_id);
CREATE INDEX IF NOT EXISTS idx_location_metrics_date ON location_metrics(date);

-- System health indexes
CREATE INDEX IF NOT EXISTS idx_system_health_component ON system_health(component);
CREATE INDEX IF NOT EXISTS idx_system_health_status ON system_health(status);
CREATE INDEX IF NOT EXISTS idx_system_health_timestamp ON system_health(timestamp);

-- A/B tests indexes
CREATE INDEX IF NOT EXISTS idx_abtest_status ON ab_tests(status);
CREATE INDEX IF NOT EXISTS idx_abtest_dates ON ab_tests(start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_variant_test_id ON test_variants(test_id);
CREATE INDEX IF NOT EXISTS idx_test_metrics_test_id ON test_daily_metrics(test_id);
CREATE INDEX IF NOT EXISTS idx_test_metrics_variant_id ON test_daily_metrics(variant_id);
CREATE INDEX IF NOT EXISTS idx_test_metrics_date ON test_daily_metrics(date);

-- User management indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_roles_name ON roles(name);

-- Create function for updating 'updated_at' fields
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for tables with updated_at
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Add RLS policies
ALTER TABLE drift_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE retraining_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE business_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE location_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_health ENABLE ROW LEVEL SECURITY;
ALTER TABLE ab_tests ENABLE ROW LEVEL SECURITY;
ALTER TABLE test_variants ENABLE ROW LEVEL SECURITY;
ALTER TABLE test_daily_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE roles ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_roles ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY "Enable read access for all users" ON drift_analysis
    FOR SELECT USING (true);

CREATE POLICY "Enable read access for all users" ON retraining_events
    FOR SELECT USING (true);

CREATE POLICY "Enable read access for all users" ON business_metrics
    FOR SELECT USING (true);

CREATE POLICY "Enable read access for all users" ON location_metrics
    FOR SELECT USING (true);

CREATE POLICY "Enable read access for all users" ON system_health
    FOR SELECT USING (true);

CREATE POLICY "Enable read access for all users" ON ab_tests
    FOR SELECT USING (true);

CREATE POLICY "Enable read access for all users" ON test_variants
    FOR SELECT USING (true);

CREATE POLICY "Enable read access for all users" ON test_daily_metrics
    FOR SELECT USING (true);

-- Service role policies for all tables
CREATE POLICY "Enable all operations for service role" ON drift_analysis
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "Enable all operations for service role" ON retraining_events
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "Enable all operations for service role" ON business_metrics
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "Enable all operations for service role" ON location_metrics
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "Enable all operations for service role" ON system_health
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "Enable all operations for service role" ON ab_tests
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "Enable all operations for service role" ON test_variants
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "Enable all operations for service role" ON test_daily_metrics
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

-- User role policies
CREATE POLICY "Enable users to see their own data" ON users
    FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "Enable service role full access to users" ON users
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

-- Add default roles
INSERT INTO roles (name, permissions) VALUES
('admin', '{"can_view": true, "can_edit": true, "can_delete": true, "can_manage_users": true}'::jsonb),
('analyst', '{"can_view": true, "can_edit": true, "can_delete": false, "can_manage_users": false}'::jsonb),
('viewer', '{"can_view": true, "can_edit": false, "can_delete": false, "can_manage_users": false}'::jsonb)
ON CONFLICT (name) DO NOTHING; 