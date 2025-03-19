import os
import logging
from dotenv import load_dotenv
from supabase import create_client, Client

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_connection():
    """Test Supabase connection and table access."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize Supabase client
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_KEY")
        
        if not url or not key:
            logger.error("Missing required environment variables")
            return False
            
        logger.info(f"Testing connection to {url}")
        supabase: Client = create_client(url, key)
        
        # Test raw_parking_data table
        logger.info("Testing raw_parking_data table access...")
        response = supabase.table('raw_parking_data').select("*").limit(1).execute()
        logger.info(f"raw_parking_data response: {response}")
        
        # Test cleaned_parking_data table
        logger.info("Testing cleaned_parking_data table access...")
        response = supabase.table('cleaned_parking_data').select("*").limit(1).execute()
        logger.info(f"cleaned_parking_data response: {response}")
        
        # Test insert with a single record
        test_record = {
            'location_id': 'test_id',
            'timestamp': '2024-03-17T12:00:00',
            'occupancy': 0.5,
            'temperature': 20.0,
            'humidity': 50.0,
            'precipitation': 0.0,
            'wind_speed': 5.0
        }
        
        logger.info("Testing insert into cleaned_parking_data...")
        response = supabase.table('cleaned_parking_data').insert(test_record).execute()
        logger.info(f"Insert response: {response}")
        
        logger.info("All tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_connection() 