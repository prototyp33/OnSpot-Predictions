#!/usr/bin/env python
"""
Script for generating sample parking data for testing.
"""

import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime, timedelta
from parking_sim.data_ingestion import DataIngestion

def generate_sample_data(output_path, start_date, end_date, num_locations=3):
    """
    Generate sample parking data for testing.
    
    Args:
        output_path: Path to save generated data
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        num_locations: Number of parking locations to generate
    """
    # Initialize data ingestion
    data_ingestion = DataIngestion()
    
    # Generate synthetic data
    df = data_ingestion.generate_synthetic_data(
        start_date=start_date,
        end_date=end_date,
        freq='1H',
        num_locations=num_locations
    )
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Generated sample data with {len(df)} rows and saved to {output_path}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample parking data")
    parser.add_argument("--output", default="data/sample_data.csv", help="Output file path")
    parser.add_argument("--start", default=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=datetime.now().strftime('%Y-%m-%d'),
                       help="End date (YYYY-MM-DD)")
    parser.add_argument("--locations", type=int, default=3, help="Number of parking locations")
    
    args = parser.parse_args()
    
    generate_sample_data(args.output, args.start, args.end, args.locations) 