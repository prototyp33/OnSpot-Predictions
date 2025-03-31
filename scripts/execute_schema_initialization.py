#!/usr/bin/env python
"""
Database Schema Initialization Script

This script executes the SQL initialization script and populates the database with
sample data to ensure all tables are created and properly linked.
"""

import os
import json
import argparse
import logging
import time
from dotenv import load_dotenv
from supabase import create_client, Client
from postgrest.exceptions import APIError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Supabase configuration - check multiple possible variable names
SUPABASE_URL = os.getenv('SUPABASE_URL')
# Try multiple possible key names
SUPABASE_KEY = os.getenv('SUPABASE_API_KEY')  # First try our script's expected name
if not SUPABASE_KEY:
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')  # Then try the name in the user's .env
if not SUPABASE_KEY:
    SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY')  # Finally try the service key name


def connect_to_supabase() -> Client:
    """Connect to Supabase and return the client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase credentials not found in environment variables")
    
    logger.info("Connecting to Supabase...")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase


def execute_sql_file(supabase: Client, sql_file_path: str):
    """Execute SQL statements from the specified file."""
    if not os.path.exists(sql_file_path):
        raise FileNotFoundError(f"SQL file not found: {sql_file_path}")
    
    logger.info(f"Reading SQL file: {sql_file_path}")
    with open(sql_file_path, 'r') as file:
        sql_content = file.read()
    
    # Split SQL statements by semicolon and execute them individually
    statements = sql_content.split(';')
    executed_count = 0
    failed_count = 0
    
    for stmt in statements:
        stmt = stmt.strip()
        if not stmt:  # Skip empty statements
            continue
        
        try:
            logger.info(f"Executing SQL statement: {stmt[:50]}...")  # Log first 50 chars
            # Execute the SQL statement through Supabase's REST API using RPC
            # Note: This requires a SQL function in Supabase that can execute raw SQL
            response = supabase.rpc(
                'execute_sql', 
                {'sql_statement': stmt}
            ).execute()
            
            # Check if the response indicates an error
            if 'error' in response.data:
                failed_count += 1
                logger.error(f"SQL execution failed: {response.data['error']}")
            else:
                executed_count += 1
                logger.info(f"SQL statement executed successfully")
                
            time.sleep(0.5)  # Small delay to avoid overwhelming the API
            
        except APIError as e:
            failed_count += 1
            error_msg = str(e)
            logger.error(f"Error executing SQL statement: {error_msg}")
            
            # Provide specific guidance based on error type
            if "function execute_sql" in error_msg and "does not exist" in error_msg:
                logger.error(
                    "The execute_sql function is missing. You need to create this SQL function in Supabase.\n"
                    "Run the scripts/create_rpc_functions.sql script in the Supabase SQL Editor first."
                )
            elif "permission denied" in error_msg.lower():
                logger.error(
                    "Permission denied. Make sure your Supabase API key has sufficient permissions.\n"
                    "You may need to use a service role key instead of an anonymous key."
                )
            elif "duplicate key" in error_msg.lower():
                logger.warning(
                    "Duplicate key error. This object may already exist in the database.\n"
                    "This is not necessarily an error if you're re-running the initialization."
                )
            elif "relation" in error_msg.lower() and "already exists" in error_msg.lower():
                logger.warning(
                    "The relation already exists. This is not necessarily an error\n"
                    "if you're re-running the initialization."
                )
    
    logger.info(f"SQL execution complete. {executed_count} statements executed, {failed_count} failed.")
    
    if failed_count > 0:
        logger.warning(
            "Some SQL statements failed. This might be expected if objects already exist.\n"
            "You can continue with sample data insertion to test if the schema is functional."
        )


def create_execute_sql_function(supabase: Client):
    """Create the execute_sql function in Supabase if it doesn't exist."""
    logger.info("Checking for execute_sql function...")
    
    # First try the verify_schema_completeness function which is less likely to exist
    # and should be created by the create_rpc_functions.sql script
    try:
        test_response = supabase.rpc('verify_schema_completeness').execute()
        logger.info("RPC functions already set up correctly.")
        return True
    except APIError as e:
        if "function verify_schema_completeness" in str(e) and "does not exist" in str(e):
            logger.warning("RPC helper functions are not set up yet.")
            logger.info("Please run the scripts/create_rpc_functions.sql script in the Supabase SQL Editor.")
            logger.info("The script will create all necessary RPC functions, including execute_sql.")
            
            # Provide detailed instructions for setting up RPC functions
            instructions = """
            ========================================================================
            INSTRUCTIONS FOR SETTING UP RPC FUNCTIONS
            ========================================================================
            
            1. Open your Supabase project dashboard
            2. Go to the 'SQL Editor' section
            3. Create a new query
            4. Copy the entire contents of scripts/create_rpc_functions.sql into the editor
            5. Run the query to create all necessary RPC functions
            6. After the functions are created, return to this script and run it again
            
            These functions are required for the database initialization and testing.
            ========================================================================
            """
            logger.info(instructions)
            return False
        else:
            logger.error(f"Unexpected error checking for RPC functions: {e}")
            return False
    
    # We shouldn't get here if the above check works properly, but just in case
    return False


def insert_sample_data(supabase: Client):
    """Insert sample data into the database tables."""
    logger.info("Inserting sample data into tables...")
    
    # Sample data for users table
    try:
        # Check if users table is accessible
        users_response = supabase.table('users').select('*').limit(1).execute()
        logger.info("Users table exists, inserting sample data...")
        
        # Insert a sample user if none exists
        if not users_response.data:
            sample_user = {
                'email': 'test@example.com',
                'name': 'Test User',
                'active': True,
                'metadata': json.dumps({'role': 'admin'})
            }
            user_response = supabase.table('users').insert(sample_user).execute()
            user_id = user_response.data[0]['id'] if user_response.data else None
            logger.info(f"Created sample user with ID: {user_id}")
        else:
            user_id = users_response.data[0]['id']
            logger.info(f"Using existing user with ID: {user_id}")
            
        # Associate a role with the user
        role_data = {
            'user_id': user_id,
            'role': 'admin'
        }
        try:
            supabase.table('user_roles').insert(role_data).execute()
            logger.info(f"Created user role association for user {user_id}")
        except APIError as e:
            if "duplicate key" in str(e).lower():
                logger.info(f"User role association already exists for user {user_id}")
            else:
                logger.error(f"Error creating user role: {e}")
        
        # Create sample model
        model_data = {
            'model_id': 'model_2024_test',
            'model_type': 'RandomForest',
            'training_date': '2024-01-01T00:00:00Z',
            'parameters': json.dumps({'n_estimators': 100, 'max_depth': 10}),
            'metrics': json.dumps({'accuracy': 0.85, 'f1': 0.83}),
            'user_id': user_id
        }
        try:
            model_response = supabase.table('models').insert(model_data).execute()
            logger.info("Created sample model")
        except APIError as e:
            if "duplicate key" in str(e).lower():
                logger.info("Sample model already exists")
            else:
                logger.error(f"Error creating sample model: {e}")
        
        # Create sample location data
        for location_id in ['location_001', 'location_002']:
            # Raw parking data
            raw_data = {
                'location_id': location_id,
                'timestamp': '2024-03-01T12:00:00Z',
                'occupancy': 75.5,
                'latitude': 37.7749,
                'longitude': -122.4194,
                'area_type': 'downtown',
                'user_id': user_id
            }
            try:
                supabase.table('raw_parking_data').insert(raw_data).execute()
                logger.info(f"Created sample raw parking data for {location_id}")
            except APIError as e:
                if "duplicate key" in str(e).lower():
                    logger.info(f"Sample raw parking data already exists for {location_id}")
                else:
                    logger.error(f"Error creating raw parking data: {e}")
            
            # Cleaned parking data
            cleaned_data = {
                'location_id': location_id,
                'timestamp': '2024-03-01T12:00:00Z',
                'occupancy': 75.5,
                'temperature': 22.5,
                'humidity': 60.0,
                'precipitation': 0.0,
                'wind_speed': 5.2,
                'user_id': user_id
            }
            try:
                supabase.table('cleaned_parking_data').insert(cleaned_data).execute()
                logger.info(f"Created sample cleaned parking data for {location_id}")
            except APIError as e:
                if "duplicate key" in str(e).lower():
                    logger.info(f"Sample cleaned parking data already exists for {location_id}")
                else:
                    logger.error(f"Error creating cleaned parking data: {e}")
            
            # Feature engineered data
            feature_data = {
                'location_id': location_id,
                'timestamp': '2024-03-01T12:00:00Z',
                'occupancy': 75.5,
                'temperature': 22.5,
                'humidity': 60.0,
                'precipitation': 0.0,
                'wind_speed': 5.2,
                'day_of_week': 4,
                'hour_of_day': 12,
                'is_weekend': False,
                'is_holiday': False,
                'time_of_day_sin': 0.0,
                'time_of_day_cos': 1.0,
                'day_of_week_sin': 0.0,
                'day_of_week_cos': 1.0,
                'user_id': user_id
            }
            try:
                supabase.table('feature_engineered_data').insert(feature_data).execute()
                logger.info(f"Created sample feature engineered data for {location_id}")
            except APIError as e:
                if "duplicate key" in str(e).lower():
                    logger.info(f"Sample feature engineered data already exists for {location_id}")
                else:
                    logger.error(f"Error creating feature engineered data: {e}")
            
            # Predictions
            prediction_data = {
                'model_id': 'model_2024_test',
                'location_id': location_id,
                'timestamp': '2024-03-01T13:00:00Z',
                'predicted_occupancy': 80.2,
                'actual_occupancy': 82.5,
                'prediction_error': -2.3,
                'features_used': json.dumps({
                    'temperature': 22.5,
                    'hour_of_day': 13,
                    'is_weekend': False
                }),
                'user_id': user_id
            }
            try:
                supabase.table('predictions').insert(prediction_data).execute()
                logger.info(f"Created sample prediction for {location_id}")
            except APIError as e:
                if "duplicate key" in str(e).lower():
                    logger.info(f"Sample prediction already exists for {location_id}")
                else:
                    logger.error(f"Error creating prediction: {e}")
    
    except APIError as e:
        error_msg = str(e)
        logger.error(f"Error accessing users table: {error_msg}")
        
        if "relation" in error_msg.lower() and "does not exist" in error_msg.lower():
            logger.error(
                "The users table doesn't exist. Make sure to run the schema initialization script first.\n"
                "Check if the SQL file was executed successfully."
            )
        elif "permission denied" in error_msg.lower():
            logger.error(
                "Permission denied. Make sure your Supabase API key has sufficient permissions.\n"
                "You may need to use a service role key instead of an anonymous key."
            )
        else:
            logger.error("Unknown error. Check Supabase logs for more details.")
        
        return False
    
    return True


def check_schema_completeness(supabase: Client):
    """Check if the database schema is complete."""
    logger.info("Checking if database schema is complete...")
    
    try:
        response = supabase.rpc('verify_schema_completeness').execute()
        
        if 'error' in response.data:
            logger.error(f"Error checking schema completeness: {response.data['error']}")
            return False
        
        if response.data.get('complete', False):
            logger.info(f"Database schema is complete. Found {response.data.get('tables_count', 0)} tables.")
            return True
        else:
            missing_tables = response.data.get('missing_tables', [])
            logger.warning(f"Database schema is incomplete. Missing tables: {', '.join(missing_tables)}")
            return False
            
    except APIError as e:
        logger.error(f"Error checking schema completeness: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Initialize database schema and insert sample data')
    parser.add_argument('--sql-file', default='scripts/initialize_database_schema.sql',
                        help='Path to SQL initialization file')
    parser.add_argument('--skip-init', action='store_true',
                        help='Skip SQL initialization and only insert sample data')
    parser.add_argument('--skip-sample-data', action='store_true',
                        help='Skip sample data insertion')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    args = parser.parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        # Set all handlers to debug level
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    
    try:
        # Connect to Supabase
        supabase = connect_to_supabase()
        
        # Check for RPC functions
        if not args.skip_init:
            functions_exist = create_execute_sql_function(supabase)
            if not functions_exist:
                logger.warning(
                    "Required RPC functions are not set up. "
                    "Please follow the instructions above to create them."
                )
                logger.info("Exiting. Run this script again after creating the RPC functions.")
                return 1
            
            # Execute schema initialization
            execute_sql_file(supabase, args.sql_file)
        
        # Check if schema is complete
        schema_complete = check_schema_completeness(supabase)
        if not schema_complete and not args.skip_init:
            logger.warning(
                "The database schema appears to be incomplete. "
                "There may have been errors during initialization."
            )
            user_input = input("Continue with sample data insertion anyway? (y/n): ")
            if user_input.lower() != 'y':
                logger.info("Exiting. Please check the errors and try again.")
                return 1
        
        # Insert sample data
        if not args.skip_sample_data:
            success = insert_sample_data(supabase)
            if not success:
                logger.error("Failed to insert sample data.")
                return 1
        
        logger.info("Database initialization complete")
        logger.info("You can now run the test_database_schema.py script to verify the schema.")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 