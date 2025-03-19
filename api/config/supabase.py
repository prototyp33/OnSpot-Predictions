from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

url: str = os.getenv("SUPABASE_URL")
# Use service role key for data uploads
key: str = os.getenv("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_KEY"))
supabase: Client = create_client(url, key)

# Export the client instance
__all__ = ['supabase'] 