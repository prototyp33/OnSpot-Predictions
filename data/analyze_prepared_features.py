import pandas as pd
import numpy as np
import sys
import logging
from collections import Counter
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Get the directory where the script is located
SCRIPT_DIR = Path(__file__).resolve().parent
FILE_PATH = SCRIPT_DIR / 'prepared_sample_data_features.csv' # Construct path relative to script
# FILE_PATH = 'prepared_sample_data_features.csv' # Original path
CHUNK_SIZE = 50000  # Process 50,000 rows at a time
COLUMNS_TO_ANALYZE = ['location_id', 'capacity', 'occupancy']
# --- End Configuration ---

def analyze_chunk(chunk_df, analysis_summary):
    """Analyzes a single chunk of data and updates the summary."""
    
    # Missing values
    for col in COLUMNS_TO_ANALYZE:
        missing_count = chunk_df[col].isna().sum()
        analysis_summary['missing_counts'][col] += missing_count

    # Capacity analysis
    if 'capacity' in chunk_df.columns and not chunk_df['capacity'].isna().all():
        analysis_summary['capacity_min'] = min(analysis_summary['capacity_min'], chunk_df['capacity'].min())
        analysis_summary['capacity_max'] = max(analysis_summary['capacity_max'], chunk_df['capacity'].max())
        # Check for zero or negative capacity
        analysis_summary['zero_neg_capacity_count'] += (chunk_df['capacity'] <= 0).sum()

    # Occupancy analysis
    if 'occupancy' in chunk_df.columns and not chunk_df['occupancy'].isna().all():
        # Assuming 'occupancy' is a rate/percentage
        analysis_summary['occupancy_rate_min'] = min(analysis_summary['occupancy_rate_min'], chunk_df['occupancy'].min())
        analysis_summary['occupancy_rate_max'] = max(analysis_summary['occupancy_rate_max'], chunk_df['occupancy'].max())
        analysis_summary['neg_occupancy_rate_count'] += (chunk_df['occupancy'] < 0).sum()
        # Check for occupancy rate > 100 (with a small tolerance)
        analysis_summary['occ_rate_over_100_count'] += (chunk_df['occupancy'] > 100.1).sum()

    # Location ID analysis
    if 'location_id' in chunk_df.columns:
        analysis_summary['location_ids'].update(chunk_df['location_id'].unique())

def main():
    logger.info(f"--- Analyzing {FILE_PATH} in chunks of {CHUNK_SIZE} ---")
    
    # Initialize summary dictionary
    analysis_summary = {
        'total_rows': 0,
        'missing_counts': Counter(),
        'capacity_min': np.inf,
        'capacity_max': -np.inf,
        'occupancy_rate_min': np.inf,      # Renamed
        'occupancy_rate_max': -np.inf,      # Renamed
        'location_ids': set(),
        'neg_occupancy_rate_count': 0,     # Renamed
        'zero_neg_capacity_count': 0,
        'occ_rate_over_100_count': 0      # Added this check
    }

    try:
        iterator = pd.read_csv(
            FILE_PATH, 
            chunksize=CHUNK_SIZE, 
            iterator=True, 
            low_memory=False, 
            usecols=COLUMNS_TO_ANALYZE # Only load necessary columns
        )

        chunk_num = 0
        for chunk in iterator:
            chunk_num += 1
            logger.info(f"Processing chunk {chunk_num}...")
            analysis_summary['total_rows'] += len(chunk)
            analyze_chunk(chunk, analysis_summary)

        logger.info("--- Analysis Complete ---")

        # Print Summary
        logger.info(f"Total Rows Processed: {analysis_summary['total_rows']:,}")
        
        logger.info("\nMissing Value Counts:")
        for col, count in analysis_summary['missing_counts'].items():
            if count > 0:
                 logger.info(f"  {col}: {count:,} ({count / analysis_summary['total_rows']:.2%})")
            else:
                 logger.info(f"  {col}: 0")

        logger.info("\nCapacity Analysis:")
        if analysis_summary['capacity_min'] == np.inf:
            logger.info("  Capacity data not found or all missing.")
        else:
            logger.info(f"  Min Capacity: {analysis_summary['capacity_min']}")
            logger.info(f"  Max Capacity: {analysis_summary['capacity_max']}")
            logger.info(f"  Rows with Capacity <= 0: {analysis_summary['zero_neg_capacity_count']:,}")

        logger.info("\nOccupancy Rate Analysis:") # Renamed section
        if analysis_summary['occupancy_rate_min'] == np.inf:
            logger.info("  Occupancy data not found or all missing.")
        else:
            logger.info(f"  Min Occupancy Rate: {analysis_summary['occupancy_rate_min']:.2f}%")
            logger.info(f"  Max Occupancy Rate: {analysis_summary['occupancy_rate_max']:.2f}%")
            logger.info(f"  Rows with Occupancy Rate < 0%: {analysis_summary['neg_occupancy_rate_count']:,}")
            logger.info(f"  Rows with Occupancy Rate > 100.1%: {analysis_summary['occ_rate_over_100_count']:,}")

        logger.info("\nLocation ID Analysis:")
        logger.info(f"  Number of Unique Location IDs: {len(analysis_summary['location_ids'])}")
        # Optionally log some sample IDs if needed
        # sample_ids = list(analysis_summary['location_ids'])[:10]
        # logger.info(f"  Sample Location IDs: {sample_ids}")

    except FileNotFoundError:
        logger.error(f"Error: File not found at {FILE_PATH}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")
        raise

if __name__ == "__main__":
    main() 