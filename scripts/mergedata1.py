import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import ast
import logging
from pathlib import Path
from tqdm import tqdm
import argparse
import requests
from io import StringIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PROXIMITY_THRESHOLD_METERS = 100
# --- Allow either 'lat'/'lon' or 'latitude'/'longitude' --- 
# REQUIRED_SIU_COLUMNS = ['lat', 'lon'] # Original
POSSIBLE_LAT_COLS = ['lat', 'latitude']
POSSIBLE_LON_COLS = ['lon', 'longitude']
# --- End change ---
EARTH_RADIUS_METERS = 6371000
DEFAULT_LEAF_SIZE = 30

# Barcelona Open Data API endpoints
BARCELONA_PARKING_URL = "https://opendata-ajuntament.barcelona.cat/data/api/action/datastore_search"
PARKING_RESOURCE_ID = "1d6c814c-70ef-4147-aa16-a49ddb952f72"  # BSM parking facilities

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Merge parking data from Barcelona Open Data and SIU sources',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Shows default values in help
    )
    parser.add_argument(
        '--siu-path', 
        type=str,
        default='/Users/adrianiraeguialvear/OnSpot_Predictive_Model/data/feature_engineered_data.csv',  
        help='Path to SIU data CSV file'
    )
    parser.add_argument(
        '--output-path', 
        type=str, 
        default='merged_parking_data.csv',
        help='Path for output merged CSV file'
    )
    parser.add_argument(
        '--proximity-threshold', 
        type=float, 
        default=PROXIMITY_THRESHOLD_METERS,
        help='Maximum distance in meters for matching points'
    )
    return parser.parse_args()

def fetch_barcelona_parking_data():
    """
    Fetch parking facility data from Barcelona Open Data portal.
    """
    logger.info("Fetching Barcelona parking data...")
    
    try:
        params = {
            'resource_id': PARKING_RESOURCE_ID,
            'limit': 1000
        }
        
        response = requests.get(BARCELONA_PARKING_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('success'):
            raise ValueError("API request was not successful")
            
        records = data['result']['records']
        logger.info(f"Retrieved {len(records)} records from API")
        
        # Create DataFrame with parsed coordinates
        df = pd.DataFrame(records)
        
        # Parse coordinates from 'Coordenades' field
        def parse_coordinates(coord_str):
            if pd.isna(coord_str):
                return pd.Series({'lon': None, 'lat': None})  # Changed names to match expected format
            try:
                coords = coord_str.split(',')
                if len(coords) >= 2:
                    return pd.Series({
                        'lon': float(coords[0]),  # First coordinate is longitude
                        'lat': float(coords[1])   # Second coordinate is latitude
                    })
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing coordinates: {coord_str} - {e}")
            return pd.Series({'lon': None, 'lat': None})
            
        # Log sample of raw data for debugging
        logger.info("Sample of raw coordinates:")
        logger.info(df['Coordenades'].head())
            
        coord_df = df['Coordenades'].apply(parse_coordinates)
        df = pd.concat([df, coord_df], axis=1)
        
        # Drop rows with missing coordinates
        initial_len = len(df)
        df = df.dropna(subset=['lat', 'lon'])  # Using new column names
        dropped_count = initial_len - len(df)
        if dropped_count > 0:
            logger.warning(f"Dropped {dropped_count} rows with invalid coordinates")
        
        if len(df) == 0:
            raise ValueError("No valid coordinate data found")
            
        logger.info(f"Successfully processed {len(df)} parking facilities")
        
        # Add additional parking information
        df['parking_type'] = df.get('tipus_estacionament', 'Unknown')
        
        # --- Log raw 'places' data for debugging capacity issues ---
        if 'places' in df.columns:
            logger.info("Raw 'places' column value counts before conversion:")
            logger.info(df['places'].value_counts(dropna=False).head(10)) # Log top 10 values + NaNs
            logger.info(f"Data type of 'places' column: {df['places'].dtype}")
        else:
            logger.warning("'places' column not found in fetched Barcelona data!")
        # --- End logging ---

        df['capacity'] = pd.to_numeric(df['places'] if 'places' in df.columns else pd.Series([0] * len(df)), errors='coerce').fillna(0)
        df['name'] = df.get('nom', 'Unknown')
        
        # Log coordinate ranges for validation
        logger.info("\nCoordinate ranges:")
        logger.info(f"Latitude:  {df['lat'].min():.4f} to {df['lat'].max():.4f}")
        logger.info(f"Longitude: {df['lon'].min():.4f} to {df['lon'].max():.4f}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching Barcelona parking data: {e}")
        if 'response' in locals():
            logger.error("API Response:")
            try:
                logger.error(response.text)
            except:
                logger.error("Could not read response text")
        raise

def merge_datasets(bsm_data: pd.DataFrame, siu_data: pd.DataFrame, proximity_threshold: float) -> tuple[pd.DataFrame, dict]:
    """
    Merge datasets based on geographical proximity using BallTree.
    """
    logger.info("Starting matching process...")
    
    # Ensure SIU data uses 'lat', 'lon' for consistency
    if 'latitude' in siu_data.columns and 'lat' not in siu_data.columns:
        siu_data = siu_data.rename(columns={'latitude': 'lat'})
    if 'longitude' in siu_data.columns and 'lon' not in siu_data.columns:
        siu_data = siu_data.rename(columns={'longitude': 'lon'})
        
    # Convert coordinates to radians
    bsm_coords = np.radians(bsm_data[['lat', 'lon']].values)
    siu_coords = np.radians(siu_data[['lat', 'lon']].values)

    # Build BallTree
    tree = BallTree(bsm_coords, metric='haversine', leaf_size=DEFAULT_LEAF_SIZE)

    # Query for nearest neighbors
    logger.info("Finding nearest neighbors...")
    distances, indices = tree.query(siu_coords, k=3)

    # Convert distances to meters
    distances_meters = distances * EARTH_RADIUS_METERS

    # Calculate distance statistics
    closest_distances = distances_meters[:, 0]
    distance_percentiles = np.percentile(closest_distances, [25, 50, 75, 90, 95])

    # Filter matches within threshold
    matched_indices = closest_distances < proximity_threshold
    matched_bsm_indices = indices[matched_indices, 0]
    
    if not matched_indices.any():
        logger.warning("No matches found!")
        return pd.DataFrame(), {}

    # Create matched dataframes
    matched_siu = siu_data[matched_indices].reset_index(drop=True)
    matched_bsm = bsm_data.iloc[matched_bsm_indices].reset_index(drop=True)
    
    # Add distance information
    matched_siu['distance_to_parking'] = closest_distances[matched_indices]
    
    # Add match quality
    matched_siu['match_quality'] = pd.cut(
        matched_siu['distance_to_parking'],
        bins=[0, 25, 50, 100, float('inf')],
        labels=['Excellent', 'Good', 'Fair', 'Poor']
    )
    
    # Rename BSM columns
    matched_bsm = matched_bsm.rename(columns={
        'lat': 'parking_lat',
        'lon': 'parking_lon'
    })
    
    # Merge datasets
    merged_data = pd.concat([matched_siu, matched_bsm], axis=1)
    
    # Calculate statistics
    stats = {
        'total_parking_facilities': len(bsm_data),
        'total_siu_points': len(siu_data),
        'matches_found': matched_indices.sum(),
        'unique_facilities_matched': len(np.unique(matched_bsm_indices)),
        'matching_rate': (matched_indices.sum() / len(siu_data)) * 100,
        'distance_stats': {
            'min': closest_distances[matched_indices].min(),
            'max': closest_distances[matched_indices].max(),
            'mean': closest_distances[matched_indices].mean(),
            'median': np.median(closest_distances[matched_indices]),
            'percentiles': {
                '25th': distance_percentiles[0],
                '75th': distance_percentiles[2],
                '90th': distance_percentiles[3],
                '95th': distance_percentiles[4]
            }
        }
    }
    
    return merged_data, stats

def main():
    """Main execution function."""
    args = parse_arguments()

    try:
        # Load SIU data
        logger.info("Loading SIU data...")
        siu_data = pd.read_csv(args.siu_path)
        
        # --- Validate SIU data (flexible column names) ---
        lat_col = next((col for col in POSSIBLE_LAT_COLS if col in siu_data.columns), None)
        lon_col = next((col for col in POSSIBLE_LON_COLS if col in siu_data.columns), None)
        
        missing_cols = []
        if not lat_col:
            missing_cols.append(f"one of {POSSIBLE_LAT_COLS}")
        if not lon_col:
            missing_cols.append(f"one of {POSSIBLE_LON_COLS}")
        
        # Original check:
        # missing_cols = [col for col in REQUIRED_SIU_COLUMNS if col not in siu_data.columns]
        # --- End validation change ---
        if missing_cols:
            raise ValueError(f"Missing required columns in SIU data: {missing_cols}")
        
        # --- Convert SIU coordinates to numeric using identified columns ---
        siu_data[lat_col] = pd.to_numeric(siu_data[lat_col], errors='coerce')
        siu_data[lon_col] = pd.to_numeric(siu_data[lon_col], errors='coerce')
        # Original conversion:
        # for col in ['lat', 'lon']:
        #     siu_data[col] = pd.to_numeric(siu_data[col], errors='coerce')
        # --- End conversion change ---
        
        # Drop invalid coordinates
        siu_data = siu_data.dropna(subset=[lat_col, lon_col])
        
        # --- Rename columns to lat/lon for consistency downstream --- 
        if lat_col != 'lat':
            siu_data = siu_data.rename(columns={lat_col: 'lat'})
        if lon_col != 'lon':
            siu_data = siu_data.rename(columns={lon_col: 'lon'})
        # --- End rename ---
            
        # Fetch Barcelona parking data
        bsm_data = fetch_barcelona_parking_data()
        
        # Merge datasets
        merged_data, stats = merge_datasets(bsm_data, siu_data, args.proximity_threshold)
        
        if len(merged_data) == 0:
            logger.error("No matches found. Try adjusting the proximity threshold.")
            return
            
        # Log statistics
        logger.info("\n=== Matching Statistics ===")
        logger.info(f"Total parking facilities: {stats['total_parking_facilities']}")
        logger.info(f"Total SIU points: {stats['total_siu_points']}")
        logger.info(f"Matches found: {stats['matches_found']}")
        logger.info(f"Matching rate: {stats['matching_rate']:.2f}%")
        logger.info(f"Unique facilities matched: {stats['unique_facilities_matched']}")
        
        logger.info("\n=== Distance Statistics (meters) ===")
        logger.info(f"Minimum: {stats['distance_stats']['min']:.2f}m")
        logger.info(f"Maximum: {stats['distance_stats']['max']:.2f}m")
        logger.info(f"Mean: {stats['distance_stats']['mean']:.2f}m")
        logger.info(f"Median: {stats['distance_stats']['median']:.2f}m")
        
        # Save results
        merged_data.to_csv(args.output_path, index=False)
        logger.info(f"\nMerged data saved to {args.output_path}")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
    