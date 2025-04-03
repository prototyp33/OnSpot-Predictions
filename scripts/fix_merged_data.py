#!/usr/bin/env python
"""
fix_merged_data.py

Fixes common issues in the merged_parking_data.csv file:
1. Fixes zero or missing capacity values by imputing with reasonable estimates
2. Standardizes location IDs for consistency

Usage:
    python fix_merged_data.py --input [input_path] --output [output_path]
"""

import os
import pandas as pd
import numpy as np
import logging
import argparse
from sklearn.cluster import KMeans
from typing import Dict, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_capacity_issues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix zero or missing capacity values in the dataset.
    
    Approaches:
    1. Group by parking_type and impute with median/mean capacity
    2. If parking_type is missing or unreliable, use geographical clustering
    """
    logger.info("Fixing capacity issues...")
    
    # Create a copy to avoid modifying the original
    fixed_df = df.copy()
    
    # Check capacity column
    total_rows = len(fixed_df)
    missing_capacity = fixed_df['capacity'].isna().sum()
    zero_capacity = (fixed_df['capacity'] == 0).sum()
    negative_capacity = (fixed_df['capacity'] < 0).sum()
    
    problem_capacity = missing_capacity + zero_capacity + negative_capacity
    problem_pct = (problem_capacity / total_rows) * 100
    
    logger.info(f"Found {problem_capacity} rows ({problem_pct:.2f}%) with problematic capacity values:")
    logger.info(f"  - Missing: {missing_capacity}")
    logger.info(f"  - Zero: {zero_capacity}")
    logger.info(f"  - Negative: {negative_capacity}")
    
    # Strategy 1: Impute based on parking_type if available
    if 'parking_type' in fixed_df.columns:
        # Calculate median capacity by parking type (excluding zeros and negatives)
        valid_mask = fixed_df['capacity'] > 0
        type_medians = fixed_df[valid_mask].groupby('parking_type')['capacity'].median()
        logger.info(f"Median capacities by parking type:")
        for ptype, median in type_medians.items():
            logger.info(f"  - {ptype}: {median:.1f}")
        
        # Fill in missing/zero/negative values with type median
        problem_mask = (fixed_df['capacity'].isna()) | (fixed_df['capacity'] <= 0)
        
        for idx in fixed_df[problem_mask].index:
            ptype = fixed_df.loc[idx, 'parking_type']
            if ptype in type_medians:
                fixed_df.loc[idx, 'capacity'] = type_medians[ptype]
            else:
                # Use overall median if type not found
                fixed_df.loc[idx, 'capacity'] = type_medians.median()
    
    # Strategy 2: If parking_type unavailable or still have problems, use geographical clustering
    problem_mask = (fixed_df['capacity'].isna()) | (fixed_df['capacity'] <= 0)
    remaining_problems = problem_mask.sum()
    
    if remaining_problems > 0 and 'lat' in fixed_df.columns and 'lon' in fixed_df.columns:
        logger.info(f"Using geographical clustering for {remaining_problems} remaining capacity issues")
        
        # Get locations with good capacity values
        good_cap_df = fixed_df[~problem_mask]
        
        if len(good_cap_df) > 0:
            # Create geographical clusters
            n_clusters = min(20, len(good_cap_df) // 5)  # Reasonable number of clusters
            coords = good_cap_df[['lat', 'lon']].values
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
            good_cap_df['geo_cluster'] = kmeans.labels_
            
            # Calculate median capacity by cluster
            cluster_medians = good_cap_df.groupby('geo_cluster')['capacity'].median()
            
            # Assign clusters to problematic rows and impute
            problem_coords = fixed_df.loc[problem_mask, ['lat', 'lon']].values
            problem_clusters = kmeans.predict(problem_coords)
            
            # Convert index to list for proper iteration
            problem_indices = fixed_df[problem_mask].index.tolist()
            
            for i, idx in enumerate(problem_indices):
                cluster = problem_clusters[i]
                fixed_df.loc[idx, 'capacity'] = cluster_medians[cluster]
        else:
            # No good capacity values, use a reasonable default
            logger.warning("No valid capacity values found. Using default value of 100.")
            fixed_df.loc[problem_mask, 'capacity'] = 100.0
    
    # Final check for any remaining capacity issues and use default as last resort
    problem_mask = (fixed_df['capacity'].isna()) | (fixed_df['capacity'] <= 0)
    remaining_problems = problem_mask.sum()
    
    if remaining_problems > 0:
        logger.warning(f"Using default capacity value (100) for {remaining_problems} remaining problematic rows")
        fixed_df.loc[problem_mask, 'capacity'] = 100.0
    
    # Ensure capacity is an appropriate data type
    fixed_df['capacity'] = fixed_df['capacity'].astype(float)
    
    logger.info("Capacity issues fixed successfully")
    return fixed_df

def standardize_location_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize location IDs for consistency.
    
    Approaches:
    1. Convert to consistent format (e.g., string with standard prefix)
    2. Perform fuzzy matching to detect/merge similar location IDs
    """
    logger.info("Standardizing location IDs...")
    
    # Create a copy to avoid modifying the original
    fixed_df = df.copy()
    
    # Determine which column contains the primary location identifier
    # (This depends on the dataset structure - adjust as needed)
    location_column = None
    possible_location_columns = ['location_id', 'id_tramo', 'zone_id_categorical', 'zone_id']
    
    for col in possible_location_columns:
        if col in fixed_df.columns:
            location_column = col
            break
    
    if location_column is None:
        logger.warning("No location ID column found. Skipping location ID standardization.")
        return fixed_df
    
    logger.info(f"Using '{location_column}' as the location identifier column")
    
    # Check for null values
    null_locations = fixed_df[location_column].isna().sum()
    if null_locations > 0:
        logger.warning(f"Found {null_locations} rows with missing location IDs")
        
        # If lat/lon available, we could assign location based on coordinates
        if 'lat' in fixed_df.columns and 'lon' in fixed_df.columns:
            logger.info("Assigning location IDs based on coordinates for rows with missing IDs")
            
            # Group by rounded coordinates to assign similar locations
            coord_precision = 4  # Adjust based on desired proximity
            fixed_df['temp_lat'] = fixed_df['lat'].round(coord_precision)
            fixed_df['temp_lon'] = fixed_df['lon'].round(coord_precision)
            
            # For each null location, check if we have matching coordinates
            null_idx = fixed_df[fixed_df[location_column].isna()].index
            for idx in null_idx:
                lat_rounded = fixed_df.loc[idx, 'temp_lat']
                lon_rounded = fixed_df.loc[idx, 'temp_lon']
                
                # Find matching coordinates with non-null location IDs
                matches = fixed_df[
                    (fixed_df['temp_lat'] == lat_rounded) & 
                    (fixed_df['temp_lon'] == lon_rounded) & 
                    (~fixed_df[location_column].isna())
                ]
                
                if len(matches) > 0:
                    # Use most common location ID from matches
                    most_common_loc = matches[location_column].mode()[0]
                    fixed_df.loc[idx, location_column] = most_common_loc
                else:
                    # Create a new synthetic location ID
                    new_loc_id = f"LOC_SYNTH_{lat_rounded}_{lon_rounded}"
                    fixed_df.loc[idx, location_column] = new_loc_id
            
            # Drop temporary columns
            fixed_df = fixed_df.drop(['temp_lat', 'temp_lon'], axis=1)
        else:
            # Assign synthetic IDs if no coordinates available
            logger.info("Assigning synthetic location IDs for rows with missing IDs")
            missing_mask = fixed_df[location_column].isna()
            fixed_df.loc[missing_mask, location_column] = [
                f"LOC_SYNTH_{i}" for i in range(missing_mask.sum())
            ]
    
    # Standardize format if needed
    if fixed_df[location_column].dtype != 'object':
        logger.info(f"Converting {location_column} to string format")
        fixed_df[location_column] = fixed_df[location_column].astype(str)
    
    # Ensure 'LOC_' prefix if appropriate
    if not fixed_df[location_column].str.startswith('LOC_').any():
        logger.info("Adding 'LOC_' prefix to standardize location IDs")
        fixed_df[location_column] = 'LOC_' + fixed_df[location_column]
    
    # Create a new consistent location_id column if it doesn't exist
    if location_column != 'location_id':
        if 'location_id' not in fixed_df.columns:
            logger.info(f"Creating 'location_id' column from {location_column}")
            fixed_df['location_id'] = fixed_df[location_column]
        else:
            # location_id exists but might not be primary
            null_count = fixed_df['location_id'].isna().sum()
            if null_count > 0:
                logger.info(f"Filling {null_count} missing values in location_id from {location_column}")
                fixed_df.loc[fixed_df['location_id'].isna(), 'location_id'] = \
                    fixed_df.loc[fixed_df['location_id'].isna(), location_column]
    
    # Count unique locations
    n_locations = fixed_df['location_id'].nunique()
    logger.info(f"Final dataset contains {n_locations} unique locations")
    
    return fixed_df

def perform_final_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Perform final data quality checks and cleanups."""
    logger.info("Performing final data quality checks...")
    
    # Create a copy to avoid modifying the original
    fixed_df = df.copy()
    
    # Check for any remaining missing values in key columns
    key_columns = ['location_id', 'capacity', 'lat', 'lon']
    key_columns = [col for col in key_columns if col in fixed_df.columns]
    
    for col in key_columns:
        missing = fixed_df[col].isna().sum()
        if missing > 0:
            logger.warning(f"Column '{col}' still has {missing} missing values")
            
            # Fill with sensible defaults if necessary
            if col == 'capacity':
                fixed_df[col] = fixed_df[col].fillna(100.0)
            elif col in ['lat', 'lon']:
                # Can't impute coordinates easily - might drop these rows
                logger.warning(f"Rows with missing {col} values may need to be dropped")
    
    # Check for duplicates based on time and location
    if 'timestamp' in fixed_df.columns and 'location_id' in fixed_df.columns:
        before_count = len(fixed_df)
        fixed_df = fixed_df.drop_duplicates(subset=['timestamp', 'location_id'])
        after_count = len(fixed_df)
        
        if before_count > after_count:
            logger.info(f"Removed {before_count - after_count} duplicate rows based on timestamp and location_id")
    
    # Ensure proper datatypes
    if 'capacity' in fixed_df.columns:
        fixed_df['capacity'] = pd.to_numeric(fixed_df['capacity'], errors='coerce').fillna(100.0)
    
    if 'timestamp' in fixed_df.columns and fixed_df['timestamp'].dtype != 'datetime64[ns]':
        try:
            fixed_df['timestamp'] = pd.to_datetime(fixed_df['timestamp'])
            logger.info("Converted timestamp to datetime format")
        except:
            logger.warning("Could not convert timestamp to datetime format")
    
    logger.info("Final checks complete")
    return fixed_df

def main(input_path: str, output_path: str):
    """Main function to load, fix, and save the dataset."""
    try:
        # 1. Load the data
        logger.info(f"Loading data from {input_path}...")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # 2. Fix capacity issues
        fixed_df = fix_capacity_issues(df)
        
        # 3. Standardize location IDs
        fixed_df = standardize_location_ids(fixed_df)
        
        # 4. Perform final checks
        fixed_df = perform_final_checks(fixed_df)
        
        # 5. Save the fixed data
        logger.info(f"Saving fixed data to {output_path}...")
        fixed_df.to_csv(output_path, index=False)
        
        # 6. Summary statistics
        logger.info("=== Data Fixing Summary ===")
        logger.info(f"Original rows: {len(df)}")
        logger.info(f"Fixed rows: {len(fixed_df)}")
        
        if 'capacity' in fixed_df.columns:
            logger.info(f"Capacity stats:")
            logger.info(f"  - Min: {fixed_df['capacity'].min()}")
            logger.info(f"  - Max: {fixed_df['capacity'].max()}")
            logger.info(f"  - Mean: {fixed_df['capacity'].mean():.2f}")
            logger.info(f"  - Median: {fixed_df['capacity'].median():.2f}")
        
        if 'location_id' in fixed_df.columns:
            logger.info(f"Location ID stats:")
            logger.info(f"  - Unique locations: {fixed_df['location_id'].nunique()}")
            top_locations = fixed_df['location_id'].value_counts().head(5)
            logger.info(f"  - Top 5 locations: {dict(top_locations)}")
        
        logger.info("Data fixing completed successfully")
        
    except Exception as e:
        logger.error(f"Error fixing data: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix issues in merged_parking_data.csv")
    parser.add_argument(
        "--input",
        type=str,
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the fixed CSV file"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw/",
        help="Directory containing input files (if --input not specified)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/",
        help="Directory to save output files (if --output not specified)"
    )
    parser.add_argument(
        "--default-input",
        type=str,
        default="merged_parking_data.csv",
        help="Default input filename if only directories are provided"
    )
    
    args = parser.parse_args()
    
    # Handle input/output paths
    input_path = args.input
    if not input_path:
        input_path = os.path.join(args.input_dir, args.default_input)
        logger.info(f"No input file specified, using default: {input_path}")
    
    output_path = args.output
    if not output_path:
        # Create a filename based on the input filename but with a prefix
        input_basename = os.path.basename(input_path)
        output_basename = f"fixed_{input_basename}"
        output_path = os.path.join(args.output_dir, output_basename)
        logger.info(f"No output file specified, using: {output_path}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    main(input_path, output_path) 