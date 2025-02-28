"""
Parking data generator module.

This module provides the main generator class for creating synthetic parking data
with realistic patterns and features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json

from .data_ingestion import DataIngestion
from .feature_engineering import FeatureEngineering
from .model import ParkingModel
from .utils import TimeUtils, WeatherUtils, TrafficUtils, DataUtils
from .performance_monitor import PerformanceMonitor, monitor_performance
from .memory_efficient import MemoryManager, memory_efficient
from .validation import InputValidator, ValidationError

logger = logging.getLogger(__name__)

class ParkingGenerator:
    """
    Generates synthetic parking occupancy data with realistic patterns.
    
    This class orchestrates the data generation process by combining
    time patterns, weather conditions, traffic levels, and location-specific
    factors to create realistic parking occupancy data.
    
    Attributes:
        config (Dict): Configuration parameters for data generation
        data_ingestion (DataIngestion): Component for fetching external data
        feature_engineering (FeatureEngineering): Component for feature generation
        model (ParkingModel): Component for occupancy prediction
        performance_monitor (PerformanceMonitor): Tracks execution performance
    """
    
    def __init__(self, config_path: Path):
        """
        Initialize the parking data generator.
        
        Args:
            config_path (Path): Path to the configuration JSON file
        
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the configuration is invalid
        """
        self.config = self._load_config(config_path)
        self.data_ingestion = DataIngestion(self.config)
        self.feature_engineering = FeatureEngineering(self.config)
        self.model = ParkingModel(self.config)
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize utility classes
        self.time_utils = TimeUtils()
        self.data_utils = DataUtils()
        
        logger.info("ParkingGenerator initialized successfully")
    
    def _load_config(self, config_path: Path) -> Dict:
        """
        Load and validate configuration from file.
        
        Args:
            config_path (Path): Path to configuration file
        
        Returns:
            Dict: Validated configuration dictionary
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the configuration is invalid
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate with schema
        return InputValidator.validate_config(config)
    
    @memory_efficient
    def generate_dataset(self, output_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Generate synthetic parking data.
        
        This method orchestrates the entire data generation process:
        1. Generates timestamps
        2. Creates weather data
        3. Simulates traffic patterns
        4. Calculates location factors
        5. Predicts occupancy rates
        6. Adds outliers and noise
        7. Saves the result to a CSV file (if output_path is provided)
        
        Args:
            output_path (Optional[Path]): Path to save the generated data.
                If None, the data is only returned and not saved.
        
        Returns:
            pd.DataFrame: Generated parking data with all features
            
        Raises:
            IOError: If unable to write to the output file
        """
        # Validate output path if provided
        if output_path:
            output_path = InputValidator.validate_file_path(output_path)
        
        logger.info("Starting dataset generation")
        
        with self.performance_monitor.monitor("data_preparation"):
            # Generate base timestamps
            timestamps = self.time_utils.generate_timestamps(
                self.config['data_parameters']['time_range']['start'],
                self.config['data_parameters']['time_range']['end']
            )
            
            # Get parking locations
            locations = self.data_ingestion.fetch_parking_locations()
            
            logger.info(f"Generated {len(timestamps)} timestamps and {len(locations)} locations")
        
        # Process locations in chunks to save memory
        result_chunks = []
        
        def process_location_chunk(location_chunk):
            chunk_results = []
            
            for _, location in location_chunk.iterrows():
                # Calculate location-specific factors
                location_factors = self.feature_engineering.calculate_location_factors(
                    location['lat'],
                    location['lon'],
                    location['capacity'],
                    location['parking_type']
                )
                
                # Generate weather once per location to save memory
                weather_data = self.feature_engineering.generate_weather_data(timestamps)
                
                # Predict occupancy
                occupancy = self.model.predict_occupancy(
                    timestamps,
                    weather_data,
                    location_factors,
                    location['parking_type'],
                    self._get_default_weights()
                )
                
                # Create location data
                for i, ts in enumerate(timestamps):
                    chunk_results.append({
                        'timestamp': ts,
                        'latitude': location['lat'],
                        'longitude': location['lon'],
                        'parking_type': location['parking_type'],
                        'parking_capacity': location['capacity'],
                        'temperature': weather_data[0][i],
                        'humidity': weather_data[1][i],
                        'wind_speed': weather_data[2][i],
                        'precipitation': weather_data[3][i],
                        'occupancy_rate': occupancy[i]
                    })
                
                # Clear weather data to free memory
                del weather_data
                MemoryManager.force_garbage_collection()
                
            return chunk_results
        
        # Process locations in chunks
        with self.performance_monitor.monitor("feature_generation"):
            location_chunks = MemoryManager.process_in_chunks(
                locations, 
                chunk_size=5,  # Process 5 locations at a time
                process_func=process_location_chunk
            )
            
            # Flatten results
            result_data = [item for chunk in location_chunks for item in chunk]
        
        # Create dataframe
        df = pd.DataFrame(result_data)
        
        # Save to file if path provided
        if output_path:
            with self.performance_monitor.monitor("saving_data"):
                MemoryManager.save_efficient(
                    df, 
                    output_path,
                    format='parquet' if output_path.suffix == '.parquet' else 'csv'
                )
        
        # Log performance summary
        self.performance_monitor.log_summary()
        
        return df
    
    def _get_default_weights(self) -> Dict[str, float]:
        """
        Get default feature weights for occupancy prediction.
        
        Returns:
            Dict[str, float]: Dictionary of feature weights
        """
        return {
            'traffic_sensitivity': 0.4,
            'zone_influence': 0.3,
            'weather_impact': 0.2,
            'time_pattern': 0.5,
            'capacity_factor': 0.2,
            'special_event': 0.3
        } 