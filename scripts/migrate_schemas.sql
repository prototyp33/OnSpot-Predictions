-- OnSpot Predictive Model - Schema Migration Script
-- This script migrates tables from the public schema to dedicated schemas

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
-- Migrate CORE schema tables
--------------------------------------------------------------------------

-- models table
CREATE TABLE IF NOT EXISTS core.models (LIKE public.models INCLUDING ALL);
INSERT INTO core.models SELECT * FROM public.models;

-- predictions table
CREATE TABLE IF NOT EXISTS core.predictions (LIKE public.predictions INCLUDING ALL);
INSERT INTO core.predictions SELECT * FROM public.predictions;

-- raw_parking_data table
CREATE TABLE IF NOT EXISTS core.raw_parking_data (LIKE public.raw_parking_data INCLUDING ALL);
INSERT INTO core.raw_parking_data SELECT * FROM public.raw_parking_data;

-- cleaned_parking_data table 
CREATE TABLE IF NOT EXISTS core.cleaned_parking_data (LIKE public.cleaned_parking_data INCLUDING ALL);
INSERT INTO core.cleaned_parking_data SELECT * FROM public.cleaned_parking_data;

-- feature_engineered_data table
CREATE TABLE IF NOT EXISTS core.feature_engineered_data (LIKE public.feature_engineered_data INCLUDING ALL);
INSERT INTO core.feature_engineered_data SELECT * FROM public.feature_engineered_data;

--------------------------------------------------------------------------
-- Migrate MONITORING schema tables
--------------------------------------------------------------------------

-- drift_analysis table
CREATE TABLE IF NOT EXISTS monitoring.drift_analysis (LIKE public.drift_analysis INCLUDING ALL);
INSERT INTO monitoring.drift_analysis SELECT * FROM public.drift_analysis;

-- retraining_events table
CREATE TABLE IF NOT EXISTS monitoring.retraining_events (LIKE public.retraining_events INCLUDING ALL);
INSERT INTO monitoring.retraining_events SELECT * FROM public.retraining_events;

-- system_health table
CREATE TABLE IF NOT EXISTS monitoring.system_health (LIKE public.system_health INCLUDING ALL);
INSERT INTO monitoring.system_health SELECT * FROM public.system_health;

--------------------------------------------------------------------------
-- Migrate ANALYTICS schema tables
--------------------------------------------------------------------------

-- business_metrics table
CREATE TABLE IF NOT EXISTS analytics.business_metrics (LIKE public.business_metrics INCLUDING ALL);
INSERT INTO analytics.business_metrics SELECT * FROM public.business_metrics;

-- location_metrics table
CREATE TABLE IF NOT EXISTS analytics.location_metrics (LIKE public.location_metrics INCLUDING ALL);
INSERT INTO analytics.location_metrics SELECT * FROM public.location_metrics;

--------------------------------------------------------------------------
-- Migrate EXPERIMENTATION schema tables
--------------------------------------------------------------------------

-- ab_tests table
CREATE TABLE IF NOT EXISTS experimentation.ab_tests (LIKE public.ab_tests INCLUDING ALL);
INSERT INTO experimentation.ab_tests SELECT * FROM public.ab_tests;

-- test_variants table
CREATE TABLE IF NOT EXISTS experimentation.test_variants (LIKE public.test_variants INCLUDING ALL);
INSERT INTO experimentation.test_variants SELECT * FROM public.test_variants;

-- test_daily_metrics table
CREATE TABLE IF NOT EXISTS experimentation.test_daily_metrics (LIKE public.test_daily_metrics INCLUDING ALL);
INSERT INTO experimentation.test_daily_metrics SELECT * FROM public.test_daily_metrics;

--------------------------------------------------------------------------
-- Migrate AUTH schema tables
--------------------------------------------------------------------------

-- users table
CREATE TABLE IF NOT EXISTS auth.users (LIKE public.users INCLUDING ALL);
INSERT INTO auth.users SELECT * FROM public.users;

-- user_roles table
CREATE TABLE IF NOT EXISTS auth.user_roles (LIKE public.user_roles INCLUDING ALL);
INSERT INTO auth.user_roles SELECT * FROM public.user_roles;

--------------------------------------------------------------------------
-- Step 4: Update Foreign Key Constraints
--------------------------------------------------------------------------

-- Core schema foreign keys
ALTER TABLE core.predictions DROP CONSTRAINT IF EXISTS predictions_model_id_fkey;
ALTER TABLE core.predictions ADD CONSTRAINT predictions_model_id_fkey 
    FOREIGN KEY (model_id) REFERENCES core.models(id);

ALTER TABLE core.cleaned_parking_data DROP CONSTRAINT IF EXISTS cleaned_parking_data_raw_data_id_fkey;
ALTER TABLE core.cleaned_parking_data ADD CONSTRAINT cleaned_parking_data_raw_data_id_fkey 
    FOREIGN KEY (raw_data_id) REFERENCES core.raw_parking_data(id);

ALTER TABLE core.feature_engineered_data DROP CONSTRAINT IF EXISTS feature_engineered_data_cleaned_data_id_fkey;
ALTER TABLE core.feature_engineered_data ADD CONSTRAINT feature_engineered_data_cleaned_data_id_fkey 
    FOREIGN KEY (cleaned_data_id) REFERENCES core.cleaned_parking_data(id);

-- Monitoring schema foreign keys
ALTER TABLE monitoring.drift_analysis DROP CONSTRAINT IF EXISTS drift_analysis_model_id_fkey;
ALTER TABLE monitoring.drift_analysis ADD CONSTRAINT drift_analysis_model_id_fkey 
    FOREIGN KEY (model_id) REFERENCES core.models(id);

ALTER TABLE monitoring.retraining_events DROP CONSTRAINT IF EXISTS retraining_events_model_id_fkey;
ALTER TABLE monitoring.retraining_events ADD CONSTRAINT retraining_events_model_id_fkey 
    FOREIGN KEY (model_id) REFERENCES core.models(id);

ALTER TABLE monitoring.retraining_events DROP CONSTRAINT IF EXISTS retraining_events_new_model_id_fkey;
ALTER TABLE monitoring.retraining_events ADD CONSTRAINT retraining_events_new_model_id_fkey 
    FOREIGN KEY (new_model_id) REFERENCES core.models(id);

-- Experimentation schema foreign keys
ALTER TABLE experimentation.test_variants DROP CONSTRAINT IF EXISTS test_variants_test_id_fkey;
ALTER TABLE experimentation.test_variants ADD CONSTRAINT test_variants_test_id_fkey 
    FOREIGN KEY (test_id) REFERENCES experimentation.ab_tests(id);

ALTER TABLE experimentation.test_daily_metrics DROP CONSTRAINT IF EXISTS test_daily_metrics_variant_id_fkey;
ALTER TABLE experimentation.test_daily_metrics ADD CONSTRAINT test_daily_metrics_variant_id_fkey 
    FOREIGN KEY (variant_id) REFERENCES experimentation.test_variants(id);

-- Auth schema foreign keys
ALTER TABLE auth.user_roles DROP CONSTRAINT IF EXISTS user_roles_user_id_fkey;
ALTER TABLE auth.user_roles ADD CONSTRAINT user_roles_user_id_fkey 
    FOREIGN KEY (user_id) REFERENCES auth.users(id);

ALTER TABLE auth.user_roles DROP CONSTRAINT IF EXISTS user_roles_assigned_by_fkey;
ALTER TABLE auth.user_roles ADD CONSTRAINT user_roles_assigned_by_fkey 
    FOREIGN KEY (assigned_by) REFERENCES auth.users(id);

--------------------------------------------------------------------------
-- Step 5: Create Views for Backward Compatibility
--------------------------------------------------------------------------

-- Core schema views
CREATE OR REPLACE VIEW public.models AS SELECT * FROM core.models;
CREATE OR REPLACE VIEW public.predictions AS SELECT * FROM core.predictions;
CREATE OR REPLACE VIEW public.raw_parking_data AS SELECT * FROM core.raw_parking_data;
CREATE OR REPLACE VIEW public.cleaned_parking_data AS SELECT * FROM core.cleaned_parking_data;
CREATE OR REPLACE VIEW public.feature_engineered_data AS SELECT * FROM core.feature_engineered_data;

-- Monitoring schema views
CREATE OR REPLACE VIEW public.drift_analysis AS SELECT * FROM monitoring.drift_analysis;
CREATE OR REPLACE VIEW public.retraining_events AS SELECT * FROM monitoring.retraining_events;
CREATE OR REPLACE VIEW public.system_health AS SELECT * FROM monitoring.system_health;

-- Analytics schema views
CREATE OR REPLACE VIEW public.business_metrics AS SELECT * FROM analytics.business_metrics;
CREATE OR REPLACE VIEW public.location_metrics AS SELECT * FROM analytics.location_metrics;

-- Experimentation schema views
CREATE OR REPLACE VIEW public.ab_tests AS SELECT * FROM experimentation.ab_tests;
CREATE OR REPLACE VIEW public.test_variants AS SELECT * FROM experimentation.test_variants;
CREATE OR REPLACE VIEW public.test_daily_metrics AS SELECT * FROM experimentation.test_daily_metrics;

-- Auth schema views
CREATE OR REPLACE VIEW public.users AS SELECT * FROM auth.users;
CREATE OR REPLACE VIEW public.user_roles AS SELECT * FROM auth.user_roles;

--------------------------------------------------------------------------
-- Step 6: Set Schema Permissions on Tables
--------------------------------------------------------------------------

-- Core schema permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA core TO authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA core GRANT ALL PRIVILEGES ON TABLES TO authenticated;

-- Monitoring schema permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA monitoring GRANT ALL PRIVILEGES ON TABLES TO authenticated;

-- Analytics schema permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics GRANT ALL PRIVILEGES ON TABLES TO authenticated;

-- Experimentation schema permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA experimentation TO authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA experimentation GRANT ALL PRIVILEGES ON TABLES TO authenticated;

-- Auth schema permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA auth TO authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA auth GRANT ALL PRIVILEGES ON TABLES TO authenticated;

--------------------------------------------------------------------------
-- VERIFICATION: Uncomment this section after verifying the migration worked
--------------------------------------------------------------------------

/* 
-- Drop original tables after verification
DROP TABLE IF EXISTS public.models;
DROP TABLE IF EXISTS public.predictions;
DROP TABLE IF EXISTS public.raw_parking_data;
DROP TABLE IF EXISTS public.cleaned_parking_data;
DROP TABLE IF EXISTS public.feature_engineered_data;
DROP TABLE IF EXISTS public.drift_analysis;
DROP TABLE IF EXISTS public.retraining_events;
DROP TABLE IF EXISTS public.system_health;
DROP TABLE IF EXISTS public.business_metrics;
DROP TABLE IF EXISTS public.location_metrics;
DROP TABLE IF EXISTS public.ab_tests;
DROP TABLE IF EXISTS public.test_variants;
DROP TABLE IF EXISTS public.test_daily_metrics;
DROP TABLE IF EXISTS public.users;
DROP TABLE IF EXISTS public.user_roles;
*/

-- IMPORTANT: Keep the views in the public schema for backward compatibility
-- Only drop the original tables after thorough verification and application updates 