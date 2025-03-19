-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create raw parking data table
CREATE TABLE IF NOT EXISTS raw_parking_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    location_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    occupancy INTEGER NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    area_type TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create cleaned parking data table
CREATE TABLE IF NOT EXISTS cleaned_parking_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    location_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    occupancy INTEGER NOT NULL,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    precipitation DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create feature engineered data table
CREATE TABLE IF NOT EXISTS feature_engineered_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    location_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    occupancy INTEGER NOT NULL,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    precipitation DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION,
    day_of_week INTEGER,
    hour_of_day INTEGER,
    is_weekend BOOLEAN,
    is_holiday BOOLEAN,
    time_of_day_sin DOUBLE PRECISION,
    time_of_day_cos DOUBLE PRECISION,
    day_of_week_sin DOUBLE PRECISION,
    day_of_week_cos DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_raw_location_id ON raw_parking_data(location_id);
CREATE INDEX IF NOT EXISTS idx_raw_timestamp ON raw_parking_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_cleaned_location_id ON cleaned_parking_data(location_id);
CREATE INDEX IF NOT EXISTS idx_cleaned_timestamp ON cleaned_parking_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_feature_location_id ON feature_engineered_data(location_id);
CREATE INDEX IF NOT EXISTS idx_feature_timestamp ON feature_engineered_data(timestamp);

-- Temporarily disable RLS for data upload
ALTER TABLE raw_parking_data DISABLE ROW LEVEL SECURITY;
ALTER TABLE cleaned_parking_data DISABLE ROW LEVEL SECURITY;
ALTER TABLE feature_engineered_data DISABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist
DROP POLICY IF EXISTS "Enable read access for authenticated users" ON raw_parking_data;
DROP POLICY IF EXISTS "Enable insert for service role" ON raw_parking_data;
DROP POLICY IF EXISTS "Enable read access for authenticated users" ON cleaned_parking_data;
DROP POLICY IF EXISTS "Enable insert for service role" ON cleaned_parking_data;
DROP POLICY IF EXISTS "Enable read access for authenticated users" ON feature_engineered_data;
DROP POLICY IF EXISTS "Enable insert for service role" ON feature_engineered_data;

-- Create policies for raw_parking_data
CREATE POLICY "Enable read access for all users" ON raw_parking_data
    FOR SELECT USING (true);

CREATE POLICY "Enable all operations for service role" ON raw_parking_data
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

-- Create policies for cleaned_parking_data
CREATE POLICY "Enable read access for all users" ON cleaned_parking_data
    FOR SELECT USING (true);

CREATE POLICY "Enable all operations for service role" ON cleaned_parking_data
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

-- Create policies for feature_engineered_data
CREATE POLICY "Enable read access for all users" ON feature_engineered_data
    FOR SELECT USING (true);

CREATE POLICY "Enable all operations for service role" ON feature_engineered_data
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role'); 