-- SQL functions to support database schema testing
-- These RPC functions can be executed in the Supabase SQL Editor

-- Drop all existing functions to avoid return type conflicts
DO $$
BEGIN
    -- Drop functions if they exist
    DROP FUNCTION IF EXISTS execute_sql(text);
    DROP FUNCTION IF EXISTS get_tables();
    DROP FUNCTION IF EXISTS get_table_schema(text);
    DROP FUNCTION IF EXISTS get_foreign_keys();
    DROP FUNCTION IF EXISTS verify_schema_completeness();
    
    RAISE NOTICE 'All existing functions dropped or did not exist';
END $$;

-- Function to execute raw SQL (needed for schema initialization)
CREATE OR REPLACE FUNCTION execute_sql(sql_statement text) 
RETURNS json
LANGUAGE plpgsql 
SECURITY DEFINER AS $$
BEGIN
  EXECUTE sql_statement;
  RETURN json_build_object('success', true);
EXCEPTION WHEN OTHERS THEN
  RETURN json_build_object('success', false, 'error', SQLERRM);
END;
$$;

-- Function to get all tables in the database
CREATE OR REPLACE FUNCTION get_tables() 
RETURNS json
LANGUAGE plpgsql 
SECURITY DEFINER AS $$
DECLARE
  result json;
BEGIN
  SELECT json_agg(table_name)
  INTO result
  FROM information_schema.tables
  WHERE table_schema IN ('public', 'core', 'monitoring', 'analytics', 'experimentation', 'auth')
  AND table_type = 'BASE TABLE';
  
  RETURN result;
EXCEPTION WHEN OTHERS THEN
  RETURN json_build_object('error', SQLERRM);
END;
$$;

-- Function to get a table's schema
CREATE OR REPLACE FUNCTION get_table_schema(table_name text) 
RETURNS json
LANGUAGE plpgsql 
SECURITY DEFINER AS $$
DECLARE
  result json;
BEGIN
  SELECT json_agg(
    json_build_object(
      'column_name', column_name,
      'data_type', data_type,
      'is_nullable', is_nullable,
      'column_default', column_default
    )
  )
  INTO result
  FROM information_schema.columns
  WHERE table_schema = COALESCE(
    (SELECT table_schema FROM information_schema.tables 
     WHERE table_name = get_table_schema.table_name
     AND table_schema IN ('public', 'core', 'monitoring', 'analytics', 'experimentation', 'auth')
     LIMIT 1),
    'public'
  )
  AND table_name = get_table_schema.table_name;
  
  RETURN result;
EXCEPTION WHEN OTHERS THEN
  RETURN json_build_object('error', SQLERRM);
END;
$$;

-- Function to get foreign key relationships
CREATE OR REPLACE FUNCTION get_foreign_keys() 
RETURNS json
LANGUAGE plpgsql 
SECURITY DEFINER AS $$
DECLARE
  result json;
BEGIN
  WITH fk_relations AS (
    SELECT
      kcu.table_schema || '.' || kcu.table_name AS source_table,
      kcu.column_name AS source_column,
      ccu.table_schema || '.' || ccu.table_name AS target_table,
      ccu.column_name AS target_column
    FROM information_schema.table_constraints AS tc
    JOIN information_schema.key_column_usage AS kcu
      ON tc.constraint_name = kcu.constraint_name
      AND tc.table_schema = kcu.table_schema
    JOIN information_schema.constraint_column_usage AS ccu
      ON ccu.constraint_name = tc.constraint_name
      AND ccu.table_schema = tc.table_schema
    WHERE tc.constraint_type = 'FOREIGN KEY'
    AND kcu.table_schema IN ('public', 'core', 'monitoring', 'analytics', 'experimentation', 'auth')
  )
  SELECT json_agg(
    json_build_object(
      'source_table', source_table,
      'source_column', source_column,
      'target_table', target_table,
      'target_column', target_column
    )
  )
  INTO result
  FROM fk_relations;
  
  RETURN result;
EXCEPTION WHEN OTHERS THEN
  RETURN json_build_object('error', SQLERRM);
END;
$$;

-- Function to verify if a database schema is complete
CREATE OR REPLACE FUNCTION verify_schema_completeness() 
RETURNS json
LANGUAGE plpgsql 
SECURITY DEFINER AS $$
DECLARE
  expected_tables text[] := ARRAY[
    'models', 'raw_parking_data', 'cleaned_parking_data', 'feature_engineered_data',
    'training_data', 'predictions', 'drift_analysis', 'retraining_events',
    'system_health', 'business_metrics', 'location_metrics', 'metrics',
    'ab_tests', 'test_variants', 'test_daily_metrics', 'users', 'user_roles', 'roles'
  ];
  missing_tables text[] := '{}';
  current_table text;
  result json;
BEGIN
  FOREACH current_table IN ARRAY expected_tables
  LOOP
    IF NOT EXISTS (
      SELECT 1 
      FROM information_schema.tables 
      WHERE table_schema IN ('public', 'core', 'monitoring', 'analytics', 'experimentation', 'auth')
      AND table_name = current_table
    ) THEN
      missing_tables := missing_tables || current_table;
    END IF;
  END LOOP;
  
  IF array_length(missing_tables, 1) > 0 THEN
    RETURN json_build_object(
      'complete', false,
      'missing_tables', missing_tables
    );
  ELSE
    RETURN json_build_object(
      'complete', true,
      'tables_count', array_length(expected_tables, 1)
    );
  END IF;
EXCEPTION WHEN OTHERS THEN
  RETURN json_build_object('error', SQLERRM);
END;
$$;

-- Grant execute permissions to authenticated users
GRANT EXECUTE ON FUNCTION get_tables() TO authenticated;
GRANT EXECUTE ON FUNCTION get_table_schema(text) TO authenticated;
GRANT EXECUTE ON FUNCTION get_foreign_keys() TO authenticated;
GRANT EXECUTE ON FUNCTION verify_schema_completeness() TO authenticated;
GRANT EXECUTE ON FUNCTION execute_sql(text) TO authenticated; 