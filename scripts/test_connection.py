"""Script to test Supabase connection."""
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_connection(url: str, key: str, key_type: str) -> None:
    logger.info(f"\nTesting connection with {key_type} key")
    logger.info(f"Using key starting with: {key[:10]}...")
    
    try:
        # Initialize Supabase client
        supabase: Client = create_client(url, key)
        
        # Try to fetch a single row from raw_parking_data
        response = supabase.table("raw_parking_data").select("*").limit(1).execute()
        logger.info(f"Successfully connected to Supabase with {key_type} key!")
        logger.info(f"Response: {response}")
        
    except Exception as e:
        logger.error(f"Error connecting with {key_type} key: {str(e)}")

def main():
    # Load environment variables
    load_dotenv()
    
    # Get and validate environment variables
    url = os.getenv("SUPABASE_URL", "").strip()
    anon_key = os.getenv("SUPABASE_KEY", "").strip()
    service_key = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
    
    if not url:
        logger.error("Missing SUPABASE_URL")
        return
    
    logger.info(f"Testing connections to Supabase at {url}")
    
    # Test with anon key
    if anon_key:
        test_connection(url, anon_key, "anon")
    else:
        logger.error("Missing SUPABASE_KEY (anon key)")
    
    # Test with service role key
    if service_key:
        test_connection(url, service_key, "service role")
    else:
        logger.error("Missing SUPABASE_SERVICE_KEY")

if __name__ == "__main__":
    main() 