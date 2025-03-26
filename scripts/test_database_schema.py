#!/usr/bin/env python
"""
Database Schema Testing Script

This script performs comprehensive testing of the database schema to ensure
it works as expected, including table structure validation, relationship
testing, and CRUD operations for each table.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import uuid
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("schema_test_results.log")
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import Supabase client
try:
    from dotenv import load_dotenv
    from supabase import create_client, Client
    
    # Load environment variables
    load_dotenv()
    
    # Initialize Supabase client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY", os.environ.get("SUPABASE_KEY"))
    
    if not url or not key:
        raise ValueError("Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_KEY environment variables.")
    
    supabase: Client = create_client(url, key)
    
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Install dependencies with: pip install supabase python-dotenv")
    sys.exit(1)
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {e}")
    sys.exit(1)

class SchemaTest:
    """Class for testing database schema."""
    
    def __init__(self):
        """Initialize the schema test with empty results."""
        self.test_results = {
            "connection": {"status": "pending", "details": []},
            "tables": {},
            "relationships": [],
            "summary": {"passed": 0, "failed": 0, "warnings": 0}
        }
        self.cleanup_records = {}
        self.rpc_available = False
    
    def test_connection(self) -> bool:
        """Test connection to Supabase."""
        logger.info("Testing connection to Supabase...")
        
        try:
            # Check if our custom RPC functions are available
            try:
                supabase.rpc("get_tables", {}).execute()
                self.rpc_available = True
                logger.info("Custom RPC functions are available")
            except Exception as e:
                logger.warning(f"Custom RPC functions not available: {e}")
                logger.warning("Consider running the create_rpc_functions.sql script in Supabase SQL Editor")
                self.rpc_available = False
            
            # Simple query to check connection
            response = supabase.from_("models").select("*").limit(1).execute()
            
            self.test_results["connection"] = {
                "status": "passed",
                "details": ["Successfully connected to Supabase"]
            }
            logger.info("✅ Connection test passed")
            return True
            
        except Exception as e:
            self.test_results["connection"] = {
                "status": "failed",
                "details": [f"Connection failed: {str(e)}"]
            }
            logger.error(f"❌ Connection test failed: {e}")
            return False
    
    def discover_tables(self) -> List[str]:
        """
        Discover available tables in the database.
        
        Returns:
            List of table names
        """
        logger.info("Discovering tables...")
        
        try:
            if self.rpc_available:
                # Query to get all tables from the information schema using RPC
                response = supabase.rpc("get_tables", {}).execute()
                
                if hasattr(response, "data"):
                    tables = [table["table_name"] for table in response.data]
                    logger.info(f"Discovered {len(tables)} tables: {', '.join(tables)}")
                    return tables
            
            # Fallback approach - try to query known tables
            known_tables = [
                "models", "predictions", "raw_parking_data", "cleaned_parking_data",
                "feature_engineered_data", "drift_analysis", "retraining_events",
                "business_metrics", "location_metrics", "system_health",
                "ab_tests", "test_variants", "test_daily_metrics", "users", "user_roles"
            ]
            
            # Check which tables exist
            existing_tables = []
            for table in known_tables:
                try:
                    supabase.from_(table).select("*").limit(1).execute()
                    existing_tables.append(table)
                except:
                    pass
            
            logger.info(f"Discovered {len(existing_tables)} tables: {', '.join(existing_tables)}")
            return existing_tables
            
        except Exception as e:
            logger.error(f"Failed to discover tables: {e}")
            return []
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """
        Get schema information for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with column information
        """
        try:
            if self.rpc_available:
                # Query to get column information from information schema using RPC
                response = supabase.rpc(
                    "get_table_columns", 
                    {"p_table_name": table_name}
                ).execute()
                
                if hasattr(response, "data"):
                    return {
                        "columns": response.data,
                        "status": "available"
                    }
            
            # Fallback approach - try to infer schema from a sample record
            response = supabase.from_(table_name).select("*").limit(1).execute()
            
            if hasattr(response, "data") and len(response.data) > 0:
                sample_record = response.data[0]
                columns = []
                
                for col_name, value in sample_record.items():
                    data_type = type(value).__name__
                    columns.append({
                        "column_name": col_name,
                        "data_type": data_type,
                        "inferred": True
                    })
                
                return {
                    "columns": columns,
                    "status": "inferred",
                    "warning": "Schema inferred from sample data, may not be complete"
                }
            
            return {
                "columns": [],
                "status": "unknown",
                "warning": "Could not determine schema"
            }
            
        except Exception as e:
            logger.error(f"Failed to get schema for table {table_name}: {e}")
            return {
                "columns": [],
                "status": "error",
                "error": str(e)
            }
    
    def get_foreign_keys(self) -> List[Dict[str, Any]]:
        """
        Get foreign key relationships from the database.
        
        Returns:
            List of foreign key relationships
        """
        try:
            if self.rpc_available:
                # Query to get foreign keys using RPC
                response = supabase.rpc("get_foreign_keys", {}).execute()
                
                if hasattr(response, "data"):
                    return response.data
            
            # If RPC not available, return an empty list
            logger.warning("Could not retrieve foreign key relationships - RPC function not available")
            return []
            
        except Exception as e:
            logger.error(f"Failed to get foreign keys: {e}")
            return []
    
    def test_table_structure(self, table_name: str) -> Dict[str, Any]:
        """
        Test the structure of a table by validating its schema and performing basic operations.
        
        Args:
            table_name: Name of the table to test
            
        Returns:
            Test results for the table
        """
        logger.info(f"Testing table structure: {table_name}")
        
        # Get schema information
        schema_info = self.get_table_schema(table_name)
        
        # Initialize test results
        table_results = {
            "schema": schema_info,
            "operations": {
                "select": {"status": "pending", "details": []},
                "insert": {"status": "pending", "details": []},
                "update": {"status": "pending", "details": []},
                "delete": {"status": "pending", "details": []}
            },
            "issues": []
        }
        
        # Test SELECT operation
        try:
            response = supabase.from_(table_name).select("*").limit(5).execute()
            sample_count = len(response.data) if hasattr(response, "data") else 0
            
            table_results["operations"]["select"] = {
                "status": "passed",
                "details": [f"Successfully selected {sample_count} records"]
            }
            
            # Store sample data for generating test records
            if hasattr(response, "data") and response.data:
                table_results["sample_data"] = response.data[0]
        except Exception as e:
            table_results["operations"]["select"] = {
                "status": "failed",
                "details": [f"SELECT failed: {str(e)}"]
            }
            table_results["issues"].append(f"SELECT operation failed: {str(e)}")
        
        # Skip further testing if SELECT failed
        if table_results["operations"]["select"]["status"] == "failed":
            logger.warning(f"Skipping further tests for {table_name} due to SELECT failure")
            return table_results
        
        # Get column names for test record creation
        columns = [col["column_name"] for col in schema_info["columns"]] if "columns" in schema_info else []
        
        # Generate a test record
        test_record = self._generate_test_record(table_name, schema_info, table_results.get("sample_data"))
        
        # Test INSERT operation if we have a valid test record
        if test_record:
            try:
                response = supabase.from_(table_name).insert(test_record).execute()
                
                if hasattr(response, "data") and response.data:
                    inserted_id = self._get_record_id(response.data[0])
                    table_results["operations"]["insert"] = {
                        "status": "passed",
                        "details": [f"Successfully inserted test record with ID: {inserted_id}"]
                    }
                    
                    # Save ID for cleanup and further testing
                    self.cleanup_records[table_name] = inserted_id
                    
                    # Test UPDATE operation
                    try:
                        update_field = self._get_updateable_field(schema_info)
                        if update_field:
                            update_data = {update_field: f"Updated at {datetime.now().isoformat()}"}
                            response = supabase.from_(table_name).update(update_data).eq("id", inserted_id).execute()
                            
                            table_results["operations"]["update"] = {
                                "status": "passed",
                                "details": [f"Successfully updated test record"]
                            }
                        else:
                            table_results["operations"]["update"] = {
                                "status": "skipped",
                                "details": ["No suitable field found for update test"]
                            }
                    except Exception as e:
                        table_results["operations"]["update"] = {
                            "status": "failed",
                            "details": [f"UPDATE failed: {str(e)}"]
                        }
                        table_results["issues"].append(f"UPDATE operation failed: {str(e)}")
                    
                    # Test DELETE operation
                    try:
                        response = supabase.from_(table_name).delete().eq("id", inserted_id).execute()
                        
                        table_results["operations"]["delete"] = {
                            "status": "passed",
                            "details": [f"Successfully deleted test record"]
                        }
                        
                        # Remove from cleanup since we deleted it
                        self.cleanup_records.pop(table_name, None)
                    except Exception as e:
                        table_results["operations"]["delete"] = {
                            "status": "failed",
                            "details": [f"DELETE failed: {str(e)}"]
                        }
                        table_results["issues"].append(f"DELETE operation failed: {str(e)}")
                else:
                    table_results["operations"]["insert"] = {
                        "status": "failed",
                        "details": ["Insert did not return data"]
                    }
                    table_results["issues"].append("INSERT operation did not return data")
            except Exception as e:
                table_results["operations"]["insert"] = {
                    "status": "failed",
                    "details": [f"INSERT failed: {str(e)}"]
                }
                table_results["issues"].append(f"INSERT operation failed: {str(e)}")
        else:
            table_results["operations"]["insert"] = {
                "status": "skipped",
                "details": ["Could not generate valid test record"]
            }
            table_results["issues"].append("Could not generate valid test record")
        
        logger.info(f"Completed structure tests for {table_name}")
        return table_results
    
    def _generate_test_record(self, table_name: str, schema_info: Dict[str, Any], sample_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a test record for the table based on its schema.
        
        Args:
            table_name: Table name
            schema_info: Schema information
            sample_data: Sample data for reference
            
        Returns:
            Test record dictionary or None if not possible
        """
        if "columns" not in schema_info:
            return None
        
        test_record = {}
        now = datetime.now()
        
        # Table-specific handling for known tables
        if table_name == "models":
            return {
                "model_id": f"test_model_{uuid.uuid4().hex[:8]}",
                "model_type": "test_model",
                "training_date": now.isoformat(),
                "parameters": json.dumps({"test_param": "value"}),
                "metrics": json.dumps({"accuracy": 0.95})
            }
        
        elif table_name == "predictions":
            return {
                "model_id": "test_model",
                "location_id": f"test_loc_{uuid.uuid4().hex[:8]}",
                "timestamp": now.isoformat(),
                "predicted_occupancy": 0.5,
                "actual_occupancy": 0.55
            }
        
        elif table_name == "raw_parking_data":
            return {
                "location_id": f"test_loc_{uuid.uuid4().hex[:8]}",
                "timestamp": now.isoformat(),
                "occupancy": 0.7,
                "latitude": 37.7749,
                "longitude": -122.4194,
                "area_type": "test_area"
            }
        
        elif table_name == "cleaned_parking_data":
            return {
                "location_id": f"test_loc_{uuid.uuid4().hex[:8]}",
                "timestamp": now.isoformat(),
                "occupancy": 0.65,
                "temperature": 72.5,
                "humidity": 65.0,
                "precipitation": 0.0,
                "wind_speed": 5.2
            }
        
        elif table_name == "drift_analysis":
            return {
                "model_id": "test_model",
                "feature_name": "test_feature",
                "drift_score": 0.15,
                "p_value": 0.03,
                "timestamp": now.isoformat(),
                "baseline_timestamp": (now - timedelta(days=7)).isoformat()
            }
        
        # Generic approach for other tables
        for column in schema_info["columns"]:
            col_name = column["column_name"]
            data_type = column.get("data_type", "").lower()
            
            # Skip id columns that are auto-generated
            if col_name == "id" and ("serial" in data_type or "identity" in data_type):
                continue
                
            # Skip created_at/updated_at timestamps that might be auto-generated
            if col_name in ["created_at", "updated_at"] and "timestamp" in data_type:
                continue
            
            # Use sample data as reference if available
            if sample_data and col_name in sample_data:
                sample_value = sample_data[col_name]
                if sample_value is not None:
                    test_record[col_name] = sample_value
                    continue
            
            # Generate values based on data type
            if "int" in data_type:
                test_record[col_name] = random.randint(1, 100)
            elif "float" in data_type or "double" in data_type or "numeric" in data_type:
                test_record[col_name] = round(random.random() * 100, 2)
            elif "bool" in data_type:
                test_record[col_name] = random.choice([True, False])
            elif "timestamp" in data_type or "date" in data_type:
                test_record[col_name] = now.isoformat()
            elif "json" in data_type:
                test_record[col_name] = json.dumps({"test": "value"})
            elif "uuid" in data_type:
                test_record[col_name] = str(uuid.uuid4())
            else:
                # Default to string for other types
                test_record[col_name] = f"test_{col_name}_{uuid.uuid4().hex[:8]}"
        
        return test_record
    
    def _get_record_id(self, record: Dict[str, Any]) -> Any:
        """
        Get the ID of a record, handling different ID field naming conventions.
        
        Args:
            record: Dictionary containing record data
            
        Returns:
            ID value or None if not found
        """
        # Common ID field names
        id_fields = ["id", "ID", "Id", "record_id", "uuid"]
        
        for field in id_fields:
            if field in record:
                return record[field]
        
        # If no standard ID field, return the first field's value
        if record:
            return next(iter(record.values()))
        
        return None
    
    def _get_updateable_field(self, schema_info: Dict[str, Any]) -> Optional[str]:
        """
        Find a suitable field for testing UPDATE operations.
        
        Args:
            schema_info: Schema information
            
        Returns:
            Field name or None if not found
        """
        if "columns" not in schema_info:
            return None
        
        # Prefer text/varchar fields that are not keys
        for column in schema_info["columns"]:
            col_name = column["column_name"]
            data_type = column.get("data_type", "").lower()
            
            if col_name not in ["id", "created_at", "updated_at"] and ("varchar" in data_type or "text" in data_type):
                return col_name
        
        # Fall back to any non-key field
        for column in schema_info["columns"]:
            col_name = column["column_name"]
            if col_name not in ["id", "created_at", "updated_at"]:
                return col_name
        
        return None
    
    def test_relationships(self, tables: List[str]) -> List[Dict[str, Any]]:
        """
        Test relationships between tables (foreign keys).
        
        Args:
            tables: List of table names
            
        Returns:
            List of relationship test results
        """
        logger.info("Testing relationships between tables...")
        relationship_results = []
        
        # Get foreign key relationships from database if RPC is available
        db_relationships = []
        if self.rpc_available:
            db_relationships = self.get_foreign_keys()
            if db_relationships:
                logger.info(f"Found {len(db_relationships)} foreign key relationships in database")
                
                # Convert to our expected format
                for rel in db_relationships:
                    if rel["source_table"] in tables and rel["target_table"] in tables:
                        formatted_rel = {
                            "from_table": rel["source_table"],
                            "from_column": rel["source_column"],
                            "to_table": rel["target_table"],
                            "to_column": rel["target_column"]
                        }
                        relationship_results.append(self._test_relationship(formatted_rel, tables))
                
                return relationship_results
        
        # Define known/expected relationships if RPC not available
        expected_relationships = [
            {"from_table": "predictions", "from_column": "model_id", "to_table": "models", "to_column": "model_id"},
            {"from_table": "drift_analysis", "from_column": "model_id", "to_table": "models", "to_column": "model_id"},
            {"from_table": "retraining_events", "from_column": "model_id", "to_table": "models", "to_column": "model_id"},
            {"from_table": "test_variants", "from_column": "test_id", "to_table": "ab_tests", "to_column": "id"},
            {"from_table": "test_daily_metrics", "from_column": "variant_id", "to_table": "test_variants", "to_column": "id"},
            {"from_table": "user_roles", "from_column": "user_id", "to_table": "users", "to_column": "id"}
        ]
        
        for relation in expected_relationships:
            # Check if both tables exist
            if relation["from_table"] not in tables or relation["to_table"] not in tables:
                continue
            
            # Test the relationship
            relationship_results.append(self._test_relationship(relation, tables))
        
        return relationship_results
    
    def _test_relationship(self, relation: Dict[str, str], tables: List[str]) -> Dict[str, Any]:
        """
        Test a single relationship between tables.
        
        Args:
            relation: Relationship definition
            tables: List of available tables
            
        Returns:
            Relationship test result
        """
        result = {
            "from_table": relation["from_table"],
            "from_column": relation["from_column"],
            "to_table": relation["to_table"],
            "to_column": relation["to_column"],
            "status": "pending",
            "details": []
        }
        
        try:
            # Query to check if the relationship works
            response = supabase.from_(relation["to_table"]).select(relation["to_column"]).limit(1).execute()
            
            if hasattr(response, "data") and response.data:
                foreign_key_value = response.data[0][relation["to_column"]]
                
                # Try to find records in the dependent table
                response = supabase.from_(relation["from_table"]).select("*").eq(relation["from_column"], foreign_key_value).limit(1).execute()
                
                # Record success/failure
                if hasattr(response, "data"):
                    count = len(response.data)
                    
                    if count > 0:
                        result["status"] = "verified"
                        result["details"].append(f"Relationship verified with {count} matching records")
                    else:
                        result["status"] = "possible"
                        result["details"].append("Relationship structure exists but no matching records found")
                else:
                    result["status"] = "error"
                    result["details"].append("Error checking relationship data")
            else:
                result["status"] = "untestable"
                result["details"].append("No parent records available for testing relationship")
                
        except Exception as e:
            result["status"] = "error"
            result["details"].append(f"Error testing relationship: {str(e)}")
        
        logger.info(f"Relationship {relation['from_table']}.{relation['from_column']} -> {relation['to_table']}.{relation['to_column']}: {result['status']}")
        return result
    
    def cleanup(self):
        """Clean up any test records that weren't deleted during testing."""
        logger.info("Cleaning up test records...")
        
        for table_name, record_id in self.cleanup_records.items():
            try:
                logger.info(f"Deleting test record from {table_name}")
                supabase.from_(table_name).delete().eq("id", record_id).execute()
            except Exception as e:
                logger.warning(f"Failed to clean up record in {table_name}: {e}")
    
    def calculate_summary(self):
        """Calculate summary statistics from test results."""
        passed = 0
        failed = 0
        warnings = 0
        
        # Count connection test
        if self.test_results["connection"]["status"] == "passed":
            passed += 1
        else:
            failed += 1
        
        # Count table tests
        for table_name, table_results in self.test_results["tables"].items():
            for op_name, op_results in table_results["operations"].items():
                if op_results["status"] == "passed":
                    passed += 1
                elif op_results["status"] == "failed":
                    failed += 1
                elif op_results["status"] == "skipped":
                    warnings += 1
            
            # Check for schema warnings
            if "warning" in table_results["schema"]:
                warnings += 1
        
        # Count relationship tests
        for relation in self.test_results["relationships"]:
            if relation["status"] == "verified":
                passed += 1
            elif relation["status"] == "error":
                failed += 1
            else:
                warnings += 1
        
        self.test_results["summary"] = {
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "total": passed + failed + warnings
        }
    
    def run_all_tests(self):
        """Run all schema tests."""
        try:
            logger.info("Starting database schema tests...")
            
            # Test connection
            if not self.test_connection():
                logger.error("Connection test failed. Aborting further tests.")
                return self.test_results
            
            # Discover tables
            tables = self.discover_tables()
            if not tables:
                logger.error("No tables discovered. Aborting further tests.")
                return self.test_results
            
            # Test table structures
            for table_name in tables:
                self.test_results["tables"][table_name] = self.test_table_structure(table_name)
            
            # Test relationships
            self.test_results["relationships"] = self.test_relationships(tables)
            
            # Calculate summary statistics
            self.calculate_summary()
            
            # Output summary
            logger.info("Schema tests completed.")
            logger.info(f"Summary: {self.test_results['summary']['passed']} passed, "
                       f"{self.test_results['summary']['failed']} failed, "
                       f"{self.test_results['summary']['warnings']} warnings")
            
            return self.test_results
        finally:
            # Clean up test data
            self.cleanup()
    
    def save_results(self, output_file: str = "schema_test_results.json"):
        """
        Save test results to a file.
        
        Args:
            output_file: Path to output file
        """
        try:
            with open(output_file, "w") as f:
                json.dump(self.test_results, f, indent=2)
            logger.info(f"Test results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test database schema")
    parser.add_argument("--output", default="schema_test_results.json", help="Output file for test results")
    parser.add_argument("--tables", nargs="*", help="Specific tables to test (default: all)")
    parser.add_argument("--skip-cleanup", action="store_true", help="Skip cleanup of test records")
    
    args = parser.parse_args()
    
    try:
        # Initialize and run tests
        tester = SchemaTest()
        test_results = tester.run_all_tests()
        
        # Save results
        tester.save_results(args.output)
        
        # Determine exit code based on results
        if test_results["summary"]["failed"] > 0:
            logger.error("Schema tests failed. See log for details.")
            return 1
        
        if test_results["summary"]["warnings"] > 0:
            logger.warning("Schema tests passed with warnings. See log for details.")
            return 0
        
        logger.info("All schema tests passed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error running schema tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 