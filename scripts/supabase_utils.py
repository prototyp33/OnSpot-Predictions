"""Utility functions for interacting with Supabase."""
import os
import logging
import random
import time
import backoff
from supabase import create_client, Client
from requests.exceptions import RequestException
import ssl
import httpx
from dotenv import load_dotenv
from typing import List, Dict, Any
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Supabase client
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(url, key)

def get_wait_time(tries: int) -> float:
    """Calculate wait time with exponential backoff and jitter."""
    exp_backoff = min(300, (2 ** tries))  # Cap at 300 seconds
    jitter = random.uniform(0, 0.1 * exp_backoff)  # 10% jitter
    return exp_backoff + jitter

@backoff.on_exception(
    backoff.expo,
    (RequestException, ssl.SSLError, httpx.HTTPError),
    max_tries=5,
    max_time=300,
    jitter=backoff.full_jitter
)
def upload_batch(table_name: str, data: List[Dict[str, Any]], batch_size: int = 1) -> bool:
    """Upload a batch of data with retries and error handling."""
    try:
        response = supabase.table(table_name).insert(data).execute()
        return True
    except Exception as e:
        logger.error(f"Error uploading batch to {table_name}: {str(e)}")
        if batch_size > 1:
            logger.info(f"Reducing batch size from {batch_size} to {batch_size // 2}")
            return False
        raise

def upload_data_with_dynamic_batching(data: List[Dict[str, Any]], table_name: str, initial_batch_size: int = 25) -> None:
    """Upload data with dynamic batch sizing."""
    batch_size = initial_batch_size
    total = len(data)
    uploaded = 0
    retries = 0
    max_retries = 3
    start_time = time.time()
    
    logger.info(f"Starting upload of {total:,} records to {table_name}")
    
    with tqdm(total=total, desc=f"Uploading to {table_name}", unit="records") as pbar:
        while uploaded < total and retries < max_retries:
            end_idx = min(uploaded + batch_size, total)
            batch = data[uploaded:end_idx]
            batch_count = len(batch)
            
            try:
                success = upload_batch(table_name, batch, batch_size)
                if success:
                    uploaded += batch_count
                    pbar.update(batch_count)
                    retries = 0  # Reset retries on success
                    
                    # Calculate and display progress statistics
                    elapsed = time.time() - start_time
                    upload_rate = uploaded / elapsed if elapsed > 0 else 0
                    eta_seconds = (total - uploaded) / upload_rate if upload_rate > 0 else 0
                    
                    # Update progress bar description with stats
                    pbar.set_postfix({
                        "batch": batch_size,
                        "speed": f"{upload_rate:.1f} rec/s",
                        "eta": f"{eta_seconds/60:.1f}m" if eta_seconds > 60 else f"{eta_seconds:.0f}s"
                    })
                    
                    # Log detailed progress every 5% or for large datasets, every 1000 records
                    if uploaded % max(int(total * 0.05), 1000) < batch_size:
                        percentage = (uploaded / total) * 100
                        logger.info(f"Progress: {uploaded:,}/{total:,} records ({percentage:.1f}%)")
                        logger.info(f"Upload rate: {upload_rate:.1f} records/second")
                        logger.info(f"Estimated time remaining: {eta_seconds/60:.1f} minutes")
                        logger.info(f"Current batch size: {batch_size}")
                    
                    # If successful, gradually increase batch size
                    if batch_size < initial_batch_size:
                        batch_size = min(batch_size * 2, initial_batch_size)
                    time.sleep(0.2)  # Small delay between successful uploads
                else:
                    # Reduce batch size and retry
                    batch_size = max(1, batch_size // 2)
                    retries += 1
                    logger.warning(f"Upload failed, reducing batch size to {batch_size} (retry {retries}/{max_retries})")
                    time.sleep(2)  # Longer delay before retrying
            except Exception as e:
                logger.error(f"Failed to upload batch after retries: {str(e)}")
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    logger.info(f"Reduced batch size to {batch_size}")
                    retries += 1
                    time.sleep(2)  # Longer delay before retrying
                else:
                    raise
    
    if retries >= max_retries:
        raise Exception(f"Failed to upload data to {table_name} after {max_retries} retries")
    
    # Calculate final statistics
    total_time = time.time() - start_time
    final_rate = uploaded / total_time if total_time > 0 else 0
    
    logger.info(f"✅ Successfully uploaded all {uploaded:,} records to {table_name}")
    logger.info(f"Total time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
    logger.info(f"Average upload rate: {final_rate:.1f} records/second") 