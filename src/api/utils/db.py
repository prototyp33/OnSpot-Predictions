from typing import Any, Dict, List, Optional
from api.config.supabase import supabase

class DatabaseClient:
    @staticmethod
    async def fetch_all(table: str, query: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fetch all records from a table with optional query parameters."""
        try:
            result = supabase.client.table(table).select("*")
            if query:
                for key, value in query.items():
                    result = result.eq(key, value)
            return result.execute().data
        except Exception as e:
            raise Exception(f"Error fetching data from {table}: {str(e)}")

    @staticmethod
    async def insert(table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a record into a table."""
        try:
            result = supabase.client.table(table).insert(data).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            raise Exception(f"Error inserting data into {table}: {str(e)}")

    @staticmethod
    async def update(table: str, id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a record in a table."""
        try:
            result = supabase.client.table(table).update(data).eq("id", id).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            raise Exception(f"Error updating data in {table}: {str(e)}")

    @staticmethod
    async def delete(table: str, id: str) -> bool:
        """Delete a record from a table."""
        try:
            result = supabase.client.table(table).delete().eq("id", id).execute()
            return bool(result.data)
        except Exception as e:
            raise Exception(f"Error deleting data from {table}: {str(e)}")

    @staticmethod
    async def fetch_by_id(table: str, id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single record by ID."""
        try:
            result = supabase.client.table(table).select("*").eq("id", id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            raise Exception(f"Error fetching data from {table}: {str(e)}")

# Create a singleton instance
db = DatabaseClient() 