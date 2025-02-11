#!/usr/bin/env python
"""
define_features.py

This script reads the cleaned parking dataset (which already has some basic features from load_data.py),
engineers additional features (temporal, spatial, lag, and contextual), and saves the enhanced dataset.

Usage:
    python define_features.py --input <input_csv> --output <output_csv>
Example:
    python define_features.py \
        --input /Users/adrianiraeguialvear/OnSpot_Predictive_Model/data/cleaned_parking_data_with_features.csv \
        --output /Users/adrianiraeguialvear/OnSpot_Predictive_Model/data/feature_engineered_data.csv
"""

import os
import re
import logging
import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def define_temporal_features(df):
    """
    Enhance temporal features using the existing 'datetime' column.

    Additional features:
      - minute: extracted minute from datetime.
      - slot_index: 5-minute interval index assuming regulated hours from 08:00 to 20:00.
      - sin_time, cos_time: cyclical encodings for slot_index.
    """
    # Ensure 'datetime' is in datetime format.
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"])
    
    # Extract minute (hour, day_of_week, and month should already be present from load_data.py)
    df["minute"] = df["datetime"].dt.minute

    # Calculate slot_index for regulated hours (08:00 to 20:00)
    # Formula: (hour - 8)*12 + (minute // 5) + 1 (each 5-minute slot)
    df["slot_index"] = ((df["hour"] - 8) * 12 + (df["minute"] // 5) + 1).astype(int)

    # Create cyclical time features to capture the periodicity of the day.
    total_slots = 144  # 144 slots from 08:00 to 20:00 (5-minute intervals)
    df["sin_time"] = np.sin(2 * np.pi * df["slot_index"] / total_slots)
    df["cos_time"] = np.cos(2 * np.pi * df["slot_index"] / total_slots)

    logger.info("Temporal features defined: slot_index, sin_time, cos_time.")
    return df


def define_spatial_features(df, n_clusters=5):
    """
    Define spatial features using 'lon' and 'lat' columns by clustering the parking zones.
    
    The resulting cluster ID is stored in the 'geo_cluster' column.
    """
    if "lon" in df.columns and "lat" in df.columns:
        # Drop rows with missing coordinate values
        coords = df[["lon", "lat"]].dropna().values
        if len(coords) > 0:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            # Note: We fit clustering on the complete dataframe columns "lon" and "lat"
            df["geo_cluster"] = kmeans.fit_predict(df[["lon", "lat"]])
            logger.info(f"Spatial features defined using KMeans with {n_clusters} clusters.")
        else:
            logger.warning("No coordinate data available for clustering.")
    else:
        logger.warning("Columns 'lon' and 'lat' not found. Skipping spatial features.")
    if 'id_tramo' in df.columns: # Assuming 'id_tramo' is your zone ID column
        df['zone_id_categorical'] = df['id_tramo'].astype('category')
        df['zone_id_encoded'] = df['zone_id_categorical'].cat.codes
        logger.info("Spatial features defined: zone_id_encoded (Label Encoded).")
    else:
        logger.warning("Column 'id_tramo' not found. Skipping zone_id_encoded feature.") 
    return df


def define_lag_features(df, lag_minutes=[5, 10, 15]):
    """
    Create lag features for the 'occupancy_level' column.

    For each specified lag (in minutes), a new column is created (e.g., 'occupancy_lag_5').
    Assumes data is recorded at 5-minute intervals.
    If a 'zone_id' (or 'id_tramo') column is available, lag is computed per zone.
    """
    if "occupancy_level" not in df.columns:
        logger.warning("Column 'occupancy_level' not found. Skipping lag features.")
        return df

    # Sort data by datetime (and by zone if applicable)
    df = df.sort_values("datetime")
    group_col = "id_tramo" if "id_tramo" in df.columns else None

    for lag in lag_minutes:
        lag_steps = lag // 5  # each step represents a 5-minute interval
        lag_col_name = f"occupancy_lag_{lag}"
        if group_col:
            df[lag_col_name] = df.groupby(group_col)["occupancy_level"].shift(lag_steps)
        else:
            df[lag_col_name] = df["occupancy_level"].shift(lag_steps)
        logger.info(f"Lag feature created: {lag_col_name}")
    return df


def convert_tariff(tariff_str):
    """
    Convert a tariff string (e.g., "2,50 E/h <= 2h") into a float.
    This function removes non-numeric characters and converts commas to dots.
    """
    if pd.isna(tariff_str):
        return np.nan
    # Replace comma with dot and extract the numeric part
    tariff_str = tariff_str.replace(',', '.')
    match = re.search(r"([\d\.]+)", tariff_str)
    return float(match.group(1)) if match else np.nan


def define_additional_features(df):
    """
    Define additional contextual features such as:
      - Converting 'tarifa' to a numeric value.
      - Creating a 'regulated_hours' flag indicating if the record falls within 08:00 to 20:00.
    """
    # Convert 'tarifa' to numeric if present.
    if "tarifa" in df.columns:
        df["tariff_numeric"] = df["tarifa"].apply(convert_tariff)
        logger.info("Tarifa converted to numeric (tariff_numeric).")

    # Create a flag for regulated hours: hour between 8 (inclusive) and 20 (exclusive).
    df["regulated_hours"] = ((df["hour"] >= 8) & (df["hour"] < 20)).astype(int)
    logger.info("Regulated hours flag created.")
    return df


def define_features(df):
    """
    Master function to define and add all features.
    """
    logger.info("Starting feature engineering...")
    df = define_temporal_features(df)
    df = define_spatial_features(df)
    df = define_lag_features(df)
    df = define_additional_features(df)
    logger.info("Feature engineering complete.")
    return df


def main(input_path, output_path):
    """
    Main execution:
      1. Load the dataset.
      2. Engineer additional features.
      3. Save the enhanced dataset.
    """
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)

    # Ensure the datetime column is properly parsed.
    if "datetime" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"])

    # Enhance features
    df = define_features(df)

    # Save the resulting dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Feature-engineered data saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Define additional features for the OnSpot Predictive Model parking dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/Users/adrianiraeguialvear/OnSpot_Predictive_Model/data/cleaned_parking_data_with_features.csv",
        help="Path to the input CSV file (cleaned data with basic features).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/Users/adrianiraeguialvear/OnSpot_Predictive_Model/data/feature_engineered_data.csv",
        help="Path to save the feature-engineered CSV file.",
    )
    args = parser.parse_args()

    main(args.input, args.output)
