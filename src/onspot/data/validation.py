"""Data validation module for ensuring data quality."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, any]

class DataValidator:
    """Validates parking data quality and structure."""
    
    def __init__(self):
        self.required_columns = {
            'location_id': str,
            'timestamp': str,
            'occupancy_rate': float,
            'total_spots': int,
            'available_spots': int
        }
    
    def validate_parking_data(self, data: pd.DataFrame) -> ValidationResult:
        """Validate parking data structure and values."""
        errors = []
        warnings = []
        stats = {}
        
        # Check required columns
        missing_cols = set(self.required_columns.keys()) - set(data.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        if not errors:  # Only continue if we have all required columns
            # Validate data types
            for col, dtype in self.required_columns.items():
                try:
                    data[col] = data[col].astype(dtype)
                except Exception as e:
                    errors.append(f"Invalid data type for {col}: {str(e)}")
            
            # Validate value ranges
            if 'occupancy_rate' in data.columns:
                invalid_rates = data[
                    (data['occupancy_rate'] < 0) | (data['occupancy_rate'] > 1)
                ]
                if not invalid_rates.empty:
                    errors.append(
                        f"Found {len(invalid_rates)} invalid occupancy rates"
                    )
            
            # Validate spot counts
            if 'total_spots' in data.columns and 'available_spots' in data.columns:
                invalid_spots = data[
                    (data['available_spots'] > data['total_spots']) |
                    (data['available_spots'] < 0) |
                    (data['total_spots'] <= 0)
                ]
                if not invalid_spots.empty:
                    errors.append(
                        f"Found {len(invalid_spots)} invalid spot counts"
                    )
            
            # Calculate statistics
            stats = {
                'total_rows': len(data),
                'null_counts': data.isnull().sum().to_dict(),
                'occupancy_stats': data['occupancy_rate'].describe().to_dict()
                if 'occupancy_rate' in data.columns else {}
            }
            
            # Check for potential data quality issues
            if stats['null_counts'].get('occupancy_rate', 0) > 0:
                warnings.append(
                    f"Found {stats['null_counts']['occupancy_rate']} null occupancy rates"
                )
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats
        )
    
    def validate_feature_engineered_data(
        self, data: pd.DataFrame, feature_list: Optional[List[str]] = None
    ) -> ValidationResult:
        """Validate feature engineered data."""
        errors = []
        warnings = []
        stats = {}
        
        # Validate basic structure
        base_validation = self.validate_parking_data(data)
        errors.extend(base_validation.errors)
        warnings.extend(base_validation.warnings)
        
        if feature_list:
            # Check for required features
            missing_features = set(feature_list) - set(data.columns)
            if missing_features:
                errors.append(f"Missing required features: {missing_features}")
        
        # Check for high correlation between features
        if len(data.columns) > 1:
            corr_matrix = data.corr()
            high_corr = np.where(np.abs(corr_matrix) > 0.95)
            high_corr_pairs = [
                (data.columns[i], data.columns[j])
                for i, j in zip(*high_corr)
                if i < j  # Avoid duplicate pairs
            ]
            if high_corr_pairs:
                warnings.append(
                    f"Found {len(high_corr_pairs)} highly correlated feature pairs"
                )
        
        # Update statistics
        stats.update({
            'feature_stats': {
                col: data[col].describe().to_dict()
                for col in data.columns
                if data[col].dtype in ['int64', 'float64']
            }
        })
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats
        ) 