"""Data ingestion module for loading and processing raw parking data."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from onspot.utils.config import load_config

class DataIngestion:
    """Handles data ingestion from various sources."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.data_path = Path(self.config["data"]["storage_path"])
    
    def load_raw_parking_data(self) -> pd.DataFrame:
        """Load raw parking data from JSON file."""
        json_path = self.data_path / "Estacionaments_Area_DUM.json"
        with open(json_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    
    def load_cleaned_parking_data(self) -> pd.DataFrame:
        """Load cleaned parking data with features."""
        csv_path = self.data_path / "cleaned_parking_data_with_features.csv"
        return pd.read_csv(csv_path)
    
    def load_baseline_data(self) -> pd.DataFrame:
        """Load baseline data for model training."""
        csv_path = self.data_path / "baseline_data.csv"
        return pd.read_csv(csv_path)
    
    def load_feature_engineered_data(self) -> pd.DataFrame:
        """Load feature engineered data."""
        csv_path = self.data_path / "feature_engineered_data.csv"
        return pd.read_csv(csv_path)
    
    def save_processed_data(self, data: pd.DataFrame, filename: str) -> None:
        """Save processed data to CSV file."""
        output_path = self.data_path / filename
        data.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}") 