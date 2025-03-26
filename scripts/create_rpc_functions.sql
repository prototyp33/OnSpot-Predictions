-- SQL functions to support database schema testing
-- These RPC functions can be executed in the Supabase SQL Editor

-- Function to get all tables in the database
CREATE OR REPLACE FUNCTION get_tables()
RETURNS TABLE (
    table_schema text,
    table_name text,
    table_type text
) LANGUAGE SQL SECURITY DEFINER AS $$
    SELECT 
        table_schema, 
        table_name, 
        table_type
    FROM 
        information_schema.tables 
    WHERE 
        table_schema NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
        AND table_type = 'BASE TABLE'
    ORDER BY 
        table_schema, 
        table_name;
$$;

-- Function to get column information for a specific table
CREATE OR REPLACE FUNCTION get_table_columns(p_table_name text)
RETURNS TABLE (
    column_name text,
    data_type text,
    is_nullable boolean,
    column_default text,
    is_identity boolean,
    is_primary boolean
) LANGUAGE SQL SECURITY DEFINER AS $$
    WITH pk_columns AS (
        SELECT 
            a.attname as column_name
        FROM 
            pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
        WHERE 
            i.indrelid = (p_table_name)::regclass
            AND i.indisprimary
    )
    SELECT 
        c.column_name,
        c.data_type,
        CASE WHEN c.is_nullable = 'YES' THEN true ELSE false END as is_nullable,
        c.column_default,
        CASE WHEN c.is_identity = 'YES' THEN true ELSE false END as is_identity,
        CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END as is_primary
    FROM 
        information_schema.columns c
    LEFT JOIN 
        pk_columns pk ON c.column_name = pk.column_name
    WHERE 
        c.table_name = p_table_name
    ORDER BY 
        c.ordinal_position;
$$;

-- Function to get foreign key relationships
CREATE OR REPLACE FUNCTION get_foreign_keys()
RETURNS TABLE (
    constraint_name text,
    source_schema text,
    source_table text,
    source_column text,
    target_schema text,
    target_table text,
    target_column text
) LANGUAGE SQL SECURITY DEFINER AS $$
    SELECT
        tc.constraint_name,
        tc.table_schema AS source_schema,
        tc.table_name AS source_table,
        kcu.column_name AS source_column,
        ccu.table_schema AS target_schema,
        ccu.table_name AS target_table,
        ccu.column_name AS target_column
    FROM 
        information_schema.table_constraints AS tc 
        JOIN information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
    WHERE tc.constraint_type = 'FOREIGN KEY'
    ORDER BY
        tc.table_schema,
        tc.table_name;
$$;

-- Grant execute permissions to authenticated users
GRANT EXECUTE ON FUNCTION get_tables() TO authenticated;
GRANT EXECUTE ON FUNCTION get_table_columns(text) TO authenticated;
GRANT EXECUTE ON FUNCTION get_foreign_keys() TO authenticated; 