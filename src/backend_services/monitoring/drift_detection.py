"""
Data drift detection module for the OnSpot Predictive Model.
"""
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from alibi_detect.cd import TabularDrift
from alibi_detect.utils.saving import save_detector, load_detector
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DRIFT_DETECTOR_DIR = Path("monitoring/drift_detectors")
DRIFT_RESULTS_DIR = Path("monitoring/drift/results")
REFERENCE_DATA_PATH = Path("data/reference_data.csv")

# Ensure directories exist
DRIFT_DETECTOR_DIR.mkdir(parents=True, exist_ok=True)
DRIFT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Pydantic models
class DriftResult(BaseModel):
    """Data drift detection result."""
    feature: str
    p_value: float
    statistic: float
    drift_detected: bool
    threshold: float = 0.05
    timestamp: datetime = Field(default_factory=datetime.now)
    meta_data: Optional[Dict[str, Any]] = None

class DriftSummary(BaseModel):
    """Summary of drift detection results."""
    timestamp: datetime = Field(default_factory=datetime.now)
    drift_detected: bool
    n_features_with_drift: int
    features_with_drift: List[str]
    results: List[DriftResult]

def load_reference_data() -> pd.DataFrame:
    """
    Load reference data for drift detection.
    
    Returns:
        pd.DataFrame: Reference data
    """
    if not REFERENCE_DATA_PATH.exists():
        logger.error(f"Reference data not found at {REFERENCE_DATA_PATH}")
        raise FileNotFoundError(f"Reference data not found at {REFERENCE_DATA_PATH}")
    
    return pd.read_csv(REFERENCE_DATA_PATH)

def get_or_create_drift_detector(
    reference_data: pd.DataFrame,
    feature_names: List[str],
    categorical_features: Optional[List[str]] = None,
    p_val: float = 0.05,
    detector_name: str = "default"
) -> TabularDrift:
    """
    Get an existing drift detector or create a new one.
    
    Args:
        reference_data: Reference data for drift detection
        feature_names: List of feature names
        categorical_features: List of categorical feature names
        p_val: p-value threshold for drift detection
        detector_name: Name of the detector
        
    Returns:
        TabularDrift: Drift detector
    """
    detector_path = DRIFT_DETECTOR_DIR / f"{detector_name}.pkl"
    
    if detector_path.exists():
        logger.info(f"Loading existing drift detector from {detector_path}")
        return load_detector(detector_path)
    
    logger.info(f"Creating new drift detector with {len(feature_names)} features")
    
    # Prepare reference data
    X_ref = reference_data[feature_names].values
    
    # Create drift detector
    detector = TabularDrift(
        X_ref=X_ref,
        p_val=p_val,
        categories_per_feature={i: None for i in range(len(feature_names))} if categorical_features is None else None,
        feature_names=feature_names
    )
    
    # Save detector
    save_detector(detector, detector_path)
    logger.info(f"Saved drift detector to {detector_path}")
    
    return detector

def detect_drift(
    current_data: pd.DataFrame,
    reference_data: Optional[pd.DataFrame] = None,
    feature_names: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    p_val: float = 0.05,
    detector_name: str = "default"
) -> DriftSummary:
    """
    Detect drift between reference and current data.
    
    Args:
        current_data: Current data to check for drift
        reference_data: Reference data for drift detection (if None, will be loaded)
        feature_names: List of feature names (if None, will use all common columns)
        categorical_features: List of categorical feature names
        p_val: p-value threshold for drift detection
        detector_name: Name of the detector
        
    Returns:
        DriftSummary: Summary of drift detection results
    """
    # Load reference data if not provided
    if reference_data is None:
        reference_data = load_reference_data()
    
    # Determine feature names if not provided
    if feature_names is None:
        feature_names = list(set(reference_data.columns) & set(current_data.columns))
        # Remove any non-numeric columns that aren't in categorical_features
        if categorical_features is None:
            feature_names = [f for f in feature_names if pd.api.types.is_numeric_dtype(reference_data[f])]
    
    # Get or create drift detector
    detector = get_or_create_drift_detector(
        reference_data=reference_data,
        feature_names=feature_names,
        categorical_features=categorical_features,
        p_val=p_val,
        detector_name=detector_name
    )
    
    # Prepare current data
    X_current = current_data[feature_names].values
    
    # Detect drift
    logger.info(f"Detecting drift for {len(feature_names)} features")
    drift_result = detector.predict(X_current)
    
    # Extract results
    drift_detected = drift_result["data"]["is_drift"] == 1
    p_values = drift_result["data"]["p_val"]
    statistics = drift_result["data"]["distance"]
    
    # Create result objects
    results = []
    features_with_drift = []
    
    for i, feature in enumerate(feature_names):
        feature_drift = p_values[i] < p_val
        if feature_drift:
            features_with_drift.append(feature)
        
        results.append(
            DriftResult(
                feature=feature,
                p_value=float(p_values[i]),
                statistic=float(statistics[i]),
                drift_detected=feature_drift,
                threshold=p_val
            )
        )
    
    # Create summary
    summary = DriftSummary(
        drift_detected=drift_detected,
        n_features_with_drift=len(features_with_drift),
        features_with_drift=features_with_drift,
        results=results
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = DRIFT_RESULTS_DIR / f"drift_result_{timestamp}.json"
    with open(result_path, "w") as f:
        f.write(summary.json(indent=2))
    
    logger.info(f"Saved drift results to {result_path}")
    logger.info(f"Drift detected: {drift_detected}, Features with drift: {len(features_with_drift)}")
    
    return summary

def get_drift_history(
    days: int = 7,
    feature: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get drift detection history.
    
    Args:
        days: Number of days of history to retrieve
        feature: Filter results by feature name
        
    Returns:
        List[Dict[str, Any]]: Drift history
    """
    # Calculate cutoff date
    cutoff_date = datetime.now() - timedelta(days=days)
    
    # Get all result files
    result_files = list(DRIFT_RESULTS_DIR.glob("drift_result_*.json"))
    
    # Sort by timestamp (newest first)
    result_files.sort(reverse=True)
    
    # Load results
    history = []
    
    for result_file in result_files:
        try:
            # Parse timestamp from filename
            timestamp_str = result_file.stem.split("_", 2)[2]
            file_timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            
            # Skip if older than cutoff
            if file_timestamp < cutoff_date:
                continue
            
            # Load result
            with open(result_file, "r") as f:
                result_data = f.read()
            
            summary = DriftSummary.parse_raw(result_data)
            
            # Filter by feature if specified
            if feature is not None:
                filtered_results = [r for r in summary.results if r.feature == feature]
                if not filtered_results:
                    continue
                
                summary.results = filtered_results
            
            # Add to history
            history.append(summary.dict())
            
        except Exception as e:
            logger.error(f"Error loading drift result from {result_file}: {e}")
    
    return history

if __name__ == "__main__":
    # Example usage
    try:
        # Load reference and current data
        reference_data = load_reference_data()
        current_data = pd.read_csv("data/current_data.csv")
        
        # Detect drift
        drift_summary = detect_drift(
            current_data=current_data,
            reference_data=reference_data
        )
        
        print(f"Drift detected: {drift_summary.drift_detected}")
        print(f"Features with drift: {drift_summary.features_with_drift}")
        
    except Exception as e:
        logger.error(f"Error in drift detection: {e}") 