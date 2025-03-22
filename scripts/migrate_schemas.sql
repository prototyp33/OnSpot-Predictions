-- OnSpot Predictive Model - Schema Migration Script
-- This script migrates tables from the public schema to dedicated schemas
-- If tables don't exist in public schema, they will be created in the new schemas

-- Step 1: Create the new schemas
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS experimentation;
CREATE SCHEMA IF NOT EXISTS auth;

-- Step 2: Set permissions
GRANT USAGE ON SCHEMA core TO authenticated;
GRANT USAGE ON SCHEMA monitoring TO authenticated;
GRANT USAGE ON SCHEMA analytics TO authenticated;
GRANT USAGE ON SCHEMA experimentation TO authenticated;
GRANT USAGE ON SCHEMA auth TO authenticated;

-- Step 3: Set search path (makes transition easier)
ALTER DATABASE postgres SET search_path TO "$user", public, core, monitoring, analytics, experimentation, auth;

--------------------------------------------------------------------------
-- Function to check if a table exists
--------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION table_exists(schema text, tablename text) 
RETURNS BOOLEAN AS $$
DECLARE
    exists BOOLEAN;
BEGIN
    SELECT count(*) > 0 INTO exists
    FROM information_schema.tables
    WHERE table_schema = schema
    AND table_name = tablename;
    RETURN exists;
END;
$$ LANGUAGE plpgsql;

--------------------------------------------------------------------------
-- CORE schema tables
--------------------------------------------------------------------------

-- Create core.models table
DO $$
BEGIN
    IF (SELECT table_exists('public', 'models')) THEN
        -- Migrate existing table
        EXECUTE 'CREATE TABLE IF NOT EXISTS core.models (LIKE public.models INCLUDING ALL)';
        EXECUTE 'INSERT INTO core.models SELECT * FROM public.models';
        RAISE NOTICE 'Migrated table: public.models → core.models';
    ELSE
        -- Create new table
        CREATE TABLE IF NOT EXISTS core.models (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_id TEXT NOT NULL,
            model_type TEXT NOT NULL,
            training_date TIMESTAMP WITH TIME ZONE NOT NULL,
            parameters JSONB,
            metrics JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        RAISE NOTICE 'Created new table: core.models';
    END IF;
END $$;

-- Create core.predictions table
DO $$
BEGIN
    IF (SELECT table_exists('public', 'predictions')) THEN
        -- Migrate existing table
        EXECUTE 'CREATE TABLE IF NOT EXISTS core.predictions (LIKE public.predictions INCLUDING ALL)';
        EXECUTE 'INSERT INTO core.predictions SELECT * FROM public.predictions';
        RAISE NOTICE 'Migrated table: public.predictions → core.predictions';
    ELSE
        -- Create new table
        CREATE TABLE IF NOT EXISTS core.predictions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            location_id TEXT NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            predicted_occupancy FLOAT NOT NULL,
            actual_occupancy FLOAT,
            model_id UUID,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        RAISE NOTICE 'Created new table: core.predictions';
    END IF;
END $$;

-- Create core.raw_parking_data table
DO $$
BEGIN
    IF (SELECT table_exists('public', 'raw_parking_data')) THEN
        -- Migrate existing table
        EXECUTE 'CREATE TABLE IF NOT EXISTS core.raw_parking_data (LIKE public.raw_parking_data INCLUDING ALL)';
        EXECUTE 'INSERT INTO core.raw_parking_data SELECT * FROM public.raw_parking_data';
        RAISE NOTICE 'Migrated table: public.raw_parking_data → core.raw_parking_data';
    ELSE
        -- Create new table
        CREATE TABLE IF NOT EXISTS core.raw_parking_data (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            location_id TEXT NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            occupancy FLOAT,
            latitude FLOAT,
            longitude FLOAT,
            area_type TEXT,
            source TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        RAISE NOTICE 'Created new table: core.raw_parking_data';
    END IF;
END $$;

-- Create core.cleaned_parking_data table
DO $$
BEGIN
    IF (SELECT table_exists('public', 'cleaned_parking_data')) THEN
        -- Migrate existing table
        EXECUTE 'CREATE TABLE IF NOT EXISTS core.cleaned_parking_data (LIKE public.cleaned_parking_data INCLUDING ALL)';
        EXECUTE 'INSERT INTO core.cleaned_parking_data SELECT * FROM public.cleaned_parking_data';
        RAISE NOTICE 'Migrated table: public.cleaned_parking_data → core.cleaned_parking_data';
    ELSE
        -- Create new table
        CREATE TABLE IF NOT EXISTS core.cleaned_parking_data (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            location_id TEXT NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            occupancy FLOAT,
            temperature FLOAT,
            humidity FLOAT,
            precipitation FLOAT,
            wind_speed FLOAT,
            raw_data_id UUID,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        RAISE NOTICE 'Created new table: core.cleaned_parking_data';
    END IF;
END $$;

-- Create core.feature_engineered_data table
DO $$
BEGIN
    IF (SELECT table_exists('public', 'feature_engineered_data')) THEN
        -- Migrate existing table
        EXECUTE 'CREATE TABLE IF NOT EXISTS core.feature_engineered_data (LIKE public.feature_engineered_data INCLUDING ALL)';
        EXECUTE 'INSERT INTO core.feature_engineered_data SELECT * FROM public.feature_engineered_data';
        RAISE NOTICE 'Migrated table: public.feature_engineered_data → core.feature_engineered_data';
    ELSE
        -- Create new table
        CREATE TABLE IF NOT EXISTS core.feature_engineered_data (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            location_id TEXT NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            occupancy FLOAT,
            temperature FLOAT,
            humidity FLOAT,
            precipitation FLOAT,
            wind_speed FLOAT,
            day_of_week INTEGER,
            hour_of_day INTEGER,
            is_weekend BOOLEAN,
            is_holiday BOOLEAN,
            time_of_day_sin FLOAT,
            time_of_day_cos FLOAT,
            day_of_week_sin FLOAT,
            day_of_week_cos FLOAT,
            cleaned_data_id UUID,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        RAISE NOTICE 'Created new table: core.feature_engineered_data';
    END IF;
END $$;

--------------------------------------------------------------------------
-- MONITORING schema tables
--------------------------------------------------------------------------

-- Create monitoring.drift_analysis table
DO $$
BEGIN
    IF (SELECT table_exists('public', 'drift_analysis')) THEN
        -- Migrate existing table
        EXECUTE 'CREATE TABLE IF NOT EXISTS monitoring.drift_analysis (LIKE public.drift_analysis INCLUDING ALL)';
        EXECUTE 'INSERT INTO monitoring.drift_analysis SELECT * FROM public.drift_analysis';
        RAISE NOTICE 'Migrated table: public.drift_analysis → monitoring.drift_analysis';
    ELSE
        -- Create new table
        CREATE TABLE IF NOT EXISTS monitoring.drift_analysis (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_id UUID NOT NULL,
            feature_name TEXT NOT NULL,
            drift_score FLOAT NOT NULL,
            p_value FLOAT,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT now(),
            baseline_timestamp TIMESTAMP WITH TIME ZONE,
            metrics JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        RAISE NOTICE 'Created new table: monitoring.drift_analysis';
    END IF;
END $$;

-- Create monitoring.retraining_events table
DO $$
BEGIN
    IF (SELECT table_exists('public', 'retraining_events')) THEN
        -- Migrate existing table
        EXECUTE 'CREATE TABLE IF NOT EXISTS monitoring.retraining_events (LIKE public.retraining_events INCLUDING ALL)';
        EXECUTE 'INSERT INTO monitoring.retraining_events SELECT * FROM public.retraining_events';
        RAISE NOTICE 'Migrated table: public.retraining_events → monitoring.retraining_events';
    ELSE
        -- Create new table
        CREATE TABLE IF NOT EXISTS monitoring.retraining_events (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_id UUID NOT NULL,
            new_model_id UUID,
            reason TEXT NOT NULL,
            success BOOLEAN DEFAULT FALSE,
            metrics_before JSONB,
            metrics_after JSONB,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT now(),
            duration_seconds FLOAT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        RAISE NOTICE 'Created new table: monitoring.retraining_events';
    END IF;
END $$;

-- Create monitoring.system_health table
DO $$
BEGIN
    IF (SELECT table_exists('public', 'system_health')) THEN
        -- Migrate existing table
        EXECUTE 'CREATE TABLE IF NOT EXISTS monitoring.system_health (LIKE public.system_health INCLUDING ALL)';
        EXECUTE 'INSERT INTO monitoring.system_health SELECT * FROM public.system_health';
        RAISE NOTICE 'Migrated table: public.system_health → monitoring.system_health';
    ELSE
        -- Create new table
        CREATE TABLE IF NOT EXISTS monitoring.system_health (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            component TEXT NOT NULL,
            status TEXT NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT now(),
            metrics JSONB,
            alert_level TEXT,
            message TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        RAISE NOTICE 'Created new table: monitoring.system_health';
    END IF;
END $$;

--------------------------------------------------------------------------
-- ANALYTICS schema tables
--------------------------------------------------------------------------

-- Create analytics.business_metrics table
DO $$
BEGIN
    IF (SELECT table_exists('public', 'business_metrics')) THEN
        -- Migrate existing table
        EXECUTE 'CREATE TABLE IF NOT EXISTS analytics.business_metrics (LIKE public.business_metrics INCLUDING ALL)';
        EXECUTE 'INSERT INTO analytics.business_metrics SELECT * FROM public.business_metrics';
        RAISE NOTICE 'Migrated table: public.business_metrics → analytics.business_metrics';
    ELSE
        -- Create new table
        CREATE TABLE IF NOT EXISTS analytics.business_metrics (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            metric_name TEXT NOT NULL,
            metric_value FLOAT NOT NULL,
            category TEXT NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT now(),
            location_id TEXT,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        RAISE NOTICE 'Created new table: analytics.business_metrics';
    END IF;
END $$;

-- Create analytics.location_metrics table
DO $$
BEGIN
    IF (SELECT table_exists('public', 'location_metrics')) THEN
        -- Migrate existing table
        EXECUTE 'CREATE TABLE IF NOT EXISTS analytics.location_metrics (LIKE public.location_metrics INCLUDING ALL)';
        EXECUTE 'INSERT INTO analytics.location_metrics SELECT * FROM public.location_metrics';
        RAISE NOTICE 'Migrated table: public.location_metrics → analytics.location_metrics';
    ELSE
        -- Create new table
        CREATE TABLE IF NOT EXISTS analytics.location_metrics (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            location_id TEXT NOT NULL,
            date DATE NOT NULL,
            occupancy_accuracy FLOAT,
            utilization_rate FLOAT,
            revenue FLOAT,
            opportunity_cost FLOAT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        RAISE NOTICE 'Created new table: analytics.location_metrics';
    END IF;
END $$;

--------------------------------------------------------------------------
-- EXPERIMENTATION schema tables
--------------------------------------------------------------------------

-- Create experimentation.ab_tests table
DO $$
BEGIN
    IF (SELECT table_exists('public', 'ab_tests')) THEN
        -- Migrate existing table
        EXECUTE 'CREATE TABLE IF NOT EXISTS experimentation.ab_tests (LIKE public.ab_tests INCLUDING ALL)';
        EXECUTE 'INSERT INTO experimentation.ab_tests SELECT * FROM public.ab_tests';
        RAISE NOTICE 'Migrated table: public.ab_tests → experimentation.ab_tests';
    ELSE
        -- Create new table
        CREATE TABLE IF NOT EXISTS experimentation.ab_tests (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name TEXT NOT NULL,
            description TEXT,
            status TEXT NOT NULL,
            start_date TIMESTAMP WITH TIME ZONE,
            end_date TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        RAISE NOTICE 'Created new table: experimentation.ab_tests';
    END IF;
END $$;

-- Create experimentation.test_variants table
DO $$
BEGIN
    IF (SELECT table_exists('public', 'test_variants')) THEN
        -- Migrate existing table
        EXECUTE 'CREATE TABLE IF NOT EXISTS experimentation.test_variants (LIKE public.test_variants INCLUDING ALL)';
        EXECUTE 'INSERT INTO experimentation.test_variants SELECT * FROM public.test_variants';
        RAISE NOTICE 'Migrated table: public.test_variants → experimentation.test_variants';
    ELSE
        -- Create new table
        CREATE TABLE IF NOT EXISTS experimentation.test_variants (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            test_id UUID NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            allocation_percentage INTEGER NOT NULL,
            parameters JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        RAISE NOTICE 'Created new table: experimentation.test_variants';
    END IF;
END $$;

-- Create experimentation.test_daily_metrics table
DO $$
BEGIN
    IF (SELECT table_exists('public', 'test_daily_metrics')) THEN
        -- Migrate existing table
        EXECUTE 'CREATE TABLE IF NOT EXISTS experimentation.test_daily_metrics (LIKE public.test_daily_metrics INCLUDING ALL)';
        EXECUTE 'INSERT INTO experimentation.test_daily_metrics SELECT * FROM public.test_daily_metrics';
        RAISE NOTICE 'Migrated table: public.test_daily_metrics → experimentation.test_daily_metrics';
    ELSE
        -- Create new table
        CREATE TABLE IF NOT EXISTS experimentation.test_daily_metrics (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            variant_id UUID NOT NULL,
            date DATE NOT NULL,
            impressions INTEGER NOT NULL DEFAULT 0,
            clicks INTEGER NOT NULL DEFAULT 0,
            conversions INTEGER NOT NULL DEFAULT 0,
            revenue FLOAT NOT NULL DEFAULT 0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        RAISE NOTICE 'Created new table: experimentation.test_daily_metrics';
    END IF;
END $$;

--------------------------------------------------------------------------
-- AUTH schema tables
--------------------------------------------------------------------------

-- Create auth.users table
DO $$
BEGIN
    IF (SELECT table_exists('public', 'users')) THEN
        -- Migrate existing table
        EXECUTE 'CREATE TABLE IF NOT EXISTS auth.users (LIKE public.users INCLUDING ALL)';
        EXECUTE 'INSERT INTO auth.users SELECT * FROM public.users';
        RAISE NOTICE 'Migrated table: public.users → auth.users';
    ELSE
        -- Create new table
        CREATE TABLE IF NOT EXISTS auth.users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            email TEXT UNIQUE NOT NULL,
            name TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            last_login TIMESTAMP WITH TIME ZONE
        );
        RAISE NOTICE 'Created new table: auth.users';
    END IF;
END $$;

-- Create auth.user_roles table
DO $$
BEGIN
    IF (SELECT table_exists('public', 'user_roles')) THEN
        -- Migrate existing table
        EXECUTE 'CREATE TABLE IF NOT EXISTS auth.user_roles (LIKE public.user_roles INCLUDING ALL)';
        EXECUTE 'INSERT INTO auth.user_roles SELECT * FROM public.user_roles';
        RAISE NOTICE 'Migrated table: public.user_roles → auth.user_roles';
    ELSE
        -- Create new table
        CREATE TABLE IF NOT EXISTS auth.user_roles (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL,
            role TEXT NOT NULL,
            assigned_by UUID,
            assigned_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        RAISE NOTICE 'Created new table: auth.user_roles';
    END IF;
END $$;

--------------------------------------------------------------------------
-- Set up foreign key constraints and indexes on foreign keys
--------------------------------------------------------------------------

-- Core schema foreign keys
ALTER TABLE core.predictions 
    ADD CONSTRAINT IF NOT EXISTS predictions_model_id_fkey 
    FOREIGN KEY (model_id) REFERENCES core.models(id);
    
-- Add index on predictions.model_id
CREATE INDEX IF NOT EXISTS idx_predictions_model_id ON core.predictions(model_id);

ALTER TABLE core.cleaned_parking_data 
    ADD CONSTRAINT IF NOT EXISTS cleaned_parking_data_raw_data_id_fkey 
    FOREIGN KEY (raw_data_id) REFERENCES core.raw_parking_data(id);
    
-- Add index on cleaned_parking_data.raw_data_id
CREATE INDEX IF NOT EXISTS idx_cleaned_parking_data_raw_data_id ON core.cleaned_parking_data(raw_data_id);

ALTER TABLE core.feature_engineered_data 
    ADD CONSTRAINT IF NOT EXISTS feature_engineered_data_cleaned_data_id_fkey 
    FOREIGN KEY (cleaned_data_id) REFERENCES core.cleaned_parking_data(id);
    
-- Add index on feature_engineered_data.cleaned_data_id
CREATE INDEX IF NOT EXISTS idx_feature_engineered_data_cleaned_data_id ON core.feature_engineered_data(cleaned_data_id);

-- Monitoring schema foreign keys
ALTER TABLE monitoring.drift_analysis 
    ADD CONSTRAINT IF NOT EXISTS drift_analysis_model_id_fkey 
    FOREIGN KEY (model_id) REFERENCES core.models(id);
    
-- Add index on drift_analysis.model_id
CREATE INDEX IF NOT EXISTS idx_drift_analysis_model_id ON monitoring.drift_analysis(model_id);

ALTER TABLE monitoring.retraining_events 
    ADD CONSTRAINT IF NOT EXISTS retraining_events_model_id_fkey 
    FOREIGN KEY (model_id) REFERENCES core.models(id);
    
-- Add index on retraining_events.model_id
CREATE INDEX IF NOT EXISTS idx_retraining_events_model_id ON monitoring.retraining_events(model_id);

ALTER TABLE monitoring.retraining_events 
    ADD CONSTRAINT IF NOT EXISTS retraining_events_new_model_id_fkey 
    FOREIGN KEY (new_model_id) REFERENCES core.models(id);
    
-- Add index on retraining_events.new_model_id
CREATE INDEX IF NOT EXISTS idx_retraining_events_new_model_id ON monitoring.retraining_events(new_model_id);

-- Experimentation schema foreign keys
ALTER TABLE experimentation.test_variants 
    ADD CONSTRAINT IF NOT EXISTS test_variants_test_id_fkey 
    FOREIGN KEY (test_id) REFERENCES experimentation.ab_tests(id);
    
-- Add index on test_variants.test_id
CREATE INDEX IF NOT EXISTS idx_test_variants_test_id ON experimentation.test_variants(test_id);

ALTER TABLE experimentation.test_daily_metrics 
    ADD CONSTRAINT IF NOT EXISTS test_daily_metrics_variant_id_fkey 
    FOREIGN KEY (variant_id) REFERENCES experimentation.test_variants(id);
    
-- Add index on test_daily_metrics.variant_id
CREATE INDEX IF NOT EXISTS idx_test_daily_metrics_variant_id ON experimentation.test_daily_metrics(variant_id);

-- Auth schema foreign keys
ALTER TABLE auth.user_roles 
    ADD CONSTRAINT IF NOT EXISTS user_roles_user_id_fkey 
    FOREIGN KEY (user_id) REFERENCES auth.users(id);
    
-- Add index on user_roles.user_id
CREATE INDEX IF NOT EXISTS idx_user_roles_user_id ON auth.user_roles(user_id);

ALTER TABLE auth.user_roles 
    ADD CONSTRAINT IF NOT EXISTS user_roles_assigned_by_fkey 
    FOREIGN KEY (assigned_by) REFERENCES auth.users(id);
    
-- Add index on user_roles.assigned_by
CREATE INDEX IF NOT EXISTS idx_user_roles_assigned_by ON auth.user_roles(assigned_by);

-- Additional performance indexes
CREATE INDEX IF NOT EXISTS idx_raw_parking_data_location_timestamp ON core.raw_parking_data(location_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_cleaned_parking_data_location_timestamp ON core.cleaned_parking_data(location_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_feature_engineered_data_location_timestamp ON core.feature_engineered_data(location_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_location_timestamp ON core.predictions(location_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_location_metrics_location_date ON analytics.location_metrics(location_id, date);
CREATE INDEX IF NOT EXISTS idx_drift_analysis_feature_timestamp ON monitoring.drift_analysis(feature_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_test_daily_metrics_date ON experimentation.test_daily_metrics(date);

--------------------------------------------------------------------------
-- Create views for backward compatibility
--------------------------------------------------------------------------

-- Create views for core schema
CREATE OR REPLACE VIEW public.models AS SELECT * FROM core.models;
CREATE OR REPLACE VIEW public.predictions AS SELECT * FROM core.predictions;
CREATE OR REPLACE VIEW public.raw_parking_data AS SELECT * FROM core.raw_parking_data;
CREATE OR REPLACE VIEW public.cleaned_parking_data AS SELECT * FROM core.cleaned_parking_data;
CREATE OR REPLACE VIEW public.feature_engineered_data AS SELECT * FROM core.feature_engineered_data;

-- Create views for monitoring schema
CREATE OR REPLACE VIEW public.drift_analysis AS SELECT * FROM monitoring.drift_analysis;
CREATE OR REPLACE VIEW public.retraining_events AS SELECT * FROM monitoring.retraining_events;
CREATE OR REPLACE VIEW public.system_health AS SELECT * FROM monitoring.system_health;

-- Create views for analytics schema
CREATE OR REPLACE VIEW public.business_metrics AS SELECT * FROM analytics.business_metrics;
CREATE OR REPLACE VIEW public.location_metrics AS SELECT * FROM analytics.location_metrics;

-- Create views for experimentation schema
CREATE OR REPLACE VIEW public.ab_tests AS SELECT * FROM experimentation.ab_tests;
CREATE OR REPLACE VIEW public.test_variants AS SELECT * FROM experimentation.test_variants;
CREATE OR REPLACE VIEW public.test_daily_metrics AS SELECT * FROM experimentation.test_daily_metrics;

-- Create views for auth schema
CREATE OR REPLACE VIEW public.users AS SELECT * FROM auth.users;
CREATE OR REPLACE VIEW public.user_roles AS SELECT * FROM auth.user_roles;

--------------------------------------------------------------------------
-- Set up Row Level Security (RLS)
--------------------------------------------------------------------------

-- Enable RLS on all tables
ALTER TABLE core.models ENABLE ROW LEVEL SECURITY;
ALTER TABLE core.predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE core.raw_parking_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE core.cleaned_parking_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE core.feature_engineered_data ENABLE ROW LEVEL SECURITY;

ALTER TABLE monitoring.drift_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE monitoring.retraining_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE monitoring.system_health ENABLE ROW LEVEL SECURITY;

ALTER TABLE analytics.business_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics.location_metrics ENABLE ROW LEVEL SECURITY;

ALTER TABLE experimentation.ab_tests ENABLE ROW LEVEL SECURITY;
ALTER TABLE experimentation.test_variants ENABLE ROW LEVEL SECURITY;
ALTER TABLE experimentation.test_daily_metrics ENABLE ROW LEVEL SECURITY;

ALTER TABLE auth.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE auth.user_roles ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
-- Core Schema Policies
CREATE POLICY "Authenticated users can read models" ON core.models
    FOR SELECT USING (auth.role() = 'authenticated');
    
CREATE POLICY "Authenticated users can insert models" ON core.models
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');
    
CREATE POLICY "Authenticated users can update models" ON core.models
    FOR UPDATE USING (auth.role() = 'authenticated') WITH CHECK (auth.role() = 'authenticated');

CREATE POLICY "Authenticated users can read predictions" ON core.predictions
    FOR SELECT USING (auth.role() = 'authenticated');
    
CREATE POLICY "Authenticated users can insert predictions" ON core.predictions
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');

CREATE POLICY "Authenticated users can read raw_parking_data" ON core.raw_parking_data
    FOR SELECT USING (auth.role() = 'authenticated');
    
CREATE POLICY "Authenticated users can insert raw_parking_data" ON core.raw_parking_data
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');

CREATE POLICY "Authenticated users can read cleaned_parking_data" ON core.cleaned_parking_data
    FOR SELECT USING (auth.role() = 'authenticated');
    
CREATE POLICY "Authenticated users can insert cleaned_parking_data" ON core.cleaned_parking_data
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');

CREATE POLICY "Authenticated users can read feature_engineered_data" ON core.feature_engineered_data
    FOR SELECT USING (auth.role() = 'authenticated');
    
CREATE POLICY "Authenticated users can insert feature_engineered_data" ON core.feature_engineered_data
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');

-- Monitoring Schema Policies
CREATE POLICY "Authenticated users can read drift_analysis" ON monitoring.drift_analysis
    FOR SELECT USING (auth.role() = 'authenticated');
    
CREATE POLICY "Authenticated users can insert drift_analysis" ON monitoring.drift_analysis
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');

CREATE POLICY "Authenticated users can read retraining_events" ON monitoring.retraining_events
    FOR SELECT USING (auth.role() = 'authenticated');
    
CREATE POLICY "Authenticated users can insert retraining_events" ON monitoring.retraining_events
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');
    
CREATE POLICY "Authenticated users can update retraining_events" ON monitoring.retraining_events
    FOR UPDATE USING (auth.role() = 'authenticated') WITH CHECK (auth.role() = 'authenticated');

CREATE POLICY "Authenticated users can read system_health" ON monitoring.system_health
    FOR SELECT USING (auth.role() = 'authenticated');
    
CREATE POLICY "Authenticated users can insert system_health" ON monitoring.system_health
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');

-- Analytics Schema Policies
CREATE POLICY "Authenticated users can read business_metrics" ON analytics.business_metrics
    FOR SELECT USING (auth.role() = 'authenticated');
    
CREATE POLICY "Authenticated users can insert business_metrics" ON analytics.business_metrics
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');

CREATE POLICY "Authenticated users can read location_metrics" ON analytics.location_metrics
    FOR SELECT USING (auth.role() = 'authenticated');
    
CREATE POLICY "Authenticated users can insert location_metrics" ON analytics.location_metrics
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');
    
CREATE POLICY "Authenticated users can update location_metrics" ON analytics.location_metrics
    FOR UPDATE USING (auth.role() = 'authenticated') WITH CHECK (auth.role() = 'authenticated');

-- Experimentation Schema Policies
CREATE POLICY "Authenticated users can read ab_tests" ON experimentation.ab_tests
    FOR SELECT USING (auth.role() = 'authenticated');
    
CREATE POLICY "Authenticated users can insert ab_tests" ON experimentation.ab_tests
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');
    
CREATE POLICY "Authenticated users can update ab_tests" ON experimentation.ab_tests
    FOR UPDATE USING (auth.role() = 'authenticated') WITH CHECK (auth.role() = 'authenticated');

CREATE POLICY "Authenticated users can read test_variants" ON experimentation.test_variants
    FOR SELECT USING (auth.role() = 'authenticated');
    
CREATE POLICY "Authenticated users can insert test_variants" ON experimentation.test_variants
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');
    
CREATE POLICY "Authenticated users can update test_variants" ON experimentation.test_variants
    FOR UPDATE USING (auth.role() = 'authenticated') WITH CHECK (auth.role() = 'authenticated');

CREATE POLICY "Authenticated users can read test_daily_metrics" ON experimentation.test_daily_metrics
    FOR SELECT USING (auth.role() = 'authenticated');
    
CREATE POLICY "Authenticated users can insert test_daily_metrics" ON experimentation.test_daily_metrics
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');
    
CREATE POLICY "Authenticated users can update test_daily_metrics" ON experimentation.test_daily_metrics
    FOR UPDATE USING (auth.role() = 'authenticated') WITH CHECK (auth.role() = 'authenticated');

-- Auth Schema Policies
CREATE POLICY "Users can read their own data" ON auth.users
    FOR SELECT USING (auth.uid() = id OR auth.role() = 'service_role');
    
CREATE POLICY "Service role can insert users" ON auth.users
    FOR INSERT WITH CHECK (auth.role() = 'service_role');
    
CREATE POLICY "Users can update their own data" ON auth.users
    FOR UPDATE USING (auth.uid() = id OR auth.role() = 'service_role') 
    WITH CHECK (auth.uid() = id OR auth.role() = 'service_role');

CREATE POLICY "Users can read their own roles" ON auth.user_roles
    FOR SELECT USING (auth.uid() = user_id OR auth.role() = 'service_role');
    
CREATE POLICY "Service role can manage roles" ON auth.user_roles
    FOR ALL USING (auth.role() = 'service_role');

--------------------------------------------------------------------------
-- Set permissions
--------------------------------------------------------------------------

-- Grant permissions on all tables
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA core TO authenticated;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO authenticated;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO authenticated;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA experimentation TO authenticated;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA auth TO authenticated;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA core GRANT ALL PRIVILEGES ON TABLES TO authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA monitoring GRANT ALL PRIVILEGES ON TABLES TO authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics GRANT ALL PRIVILEGES ON TABLES TO authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA experimentation GRANT ALL PRIVILEGES ON TABLES TO authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA auth GRANT ALL PRIVILEGES ON TABLES TO authenticated;

-- Drop helper function
DROP FUNCTION table_exists;

-- Final information message
DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '================================================';
    RAISE NOTICE 'Schema migration completed successfully!';
    RAISE NOTICE '';
    RAISE NOTICE 'Tables have been organized into the following schemas:';
    RAISE NOTICE '- core: models, predictions, raw_parking_data, cleaned_parking_data, feature_engineered_data';
    RAISE NOTICE '- monitoring: drift_analysis, retraining_events, system_health';
    RAISE NOTICE '- analytics: business_metrics, location_metrics';
    RAISE NOTICE '- experimentation: ab_tests, test_variants, test_daily_metrics';
    RAISE NOTICE '- auth: users, user_roles';
    RAISE NOTICE '';
    RAISE NOTICE 'Security enhancements:';
    RAISE NOTICE '- Row Level Security (RLS) has been enabled for all tables';
    RAISE NOTICE '- Indexes have been created on foreign key columns for better performance';
    RAISE NOTICE '';
    RAISE NOTICE 'Views have been created in the public schema for backward compatibility.';
    RAISE NOTICE '================================================';
END $$; 