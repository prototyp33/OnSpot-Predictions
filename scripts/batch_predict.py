#!/usr/bin/env python
"""
batch_predict.py

Performs batch predictions using the trained Random Forest model
on new parking data.

Usage:
    python batch_predict.py --model [model_path] --input [input_path] --output [output_path]
"""

import os
import pandas as pd
import numpy as np
import pickle
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model(model_path: str) -> Any:
    """
    Load the trained Random Forest model from disk.
    
    Args:
        model_path: Path to the saved model pickle file
        
    Returns:
        The loaded model object
    """
    logger.info(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
    return model

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the input data by transforming and selecting the appropriate features.
    
    Args:
        df: Input DataFrame containing the raw data
        
    Returns:
        DataFrame with the prepared features ready for prediction
    """
    logger.info("Preparing features...")
    
    # Make a copy to avoid modifying the original data
    features_df = df.copy()
    
    # Ensure required columns exist
    required_columns = ['timestamp', 'location_id', 'occupancy', 'capacity']
    missing_columns = [col for col in required_columns if col not in features_df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Convert timestamp to datetime if it's not already
    if features_df['timestamp'].dtype != 'datetime64[ns]':
        features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
    
    # Extract datetime features
    features_df['hour'] = features_df['timestamp'].dt.hour
    features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
    features_df['month'] = features_df['timestamp'].dt.month
    features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
    
    # Calculate occupancy rate if not already available
    if 'occupancy_rate' not in features_df.columns:
        # Ensure capacity is greater than zero to avoid division by zero
        features_df['capacity'] = features_df['capacity'].replace(0, 1)
        features_df['occupancy_rate'] = features_df['occupancy'] / features_df['capacity']
        # Cap occupancy rate at 1.0 (100%)
        features_df['occupancy_rate'] = features_df['occupancy_rate'].clip(0, 1)
    
    # Encode location_id as categorical (if model expects this)
    # This would typically be done using the same encoding as during training
    # For simplicity, we're using category codes, but in production you'd want to
    # use the same mapping as during training
    features_df['location_id_categorical'] = features_df['location_id'].astype('category').cat.codes
    
    # Select only the features that were used during model training
    # This list should match the features used during training
    model_features = [
        'hour', 'day_of_week', 'month', 'is_weekend',
        'location_id_categorical', 'capacity'
    ]
    
    # Check if all required features are available
    missing_model_features = [feat for feat in model_features if feat not in features_df.columns]
    if missing_model_features:
        raise ValueError(f"Missing model features: {missing_model_features}")
    
    # Return only the relevant features in the correct order
    logger.info(f"Features prepared. Using {len(model_features)} features for prediction")
    return features_df[model_features]

def make_predictions(model: Any, features_df: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using the model.
    
    Args:
        model: The loaded model object
        features_df: DataFrame containing the prepared features
        
    Returns:
        Array of predicted values
    """
    logger.info(f"Making predictions on {len(features_df)} rows...")
    
    try:
        predictions = model.predict(features_df)
        logger.info("Predictions completed successfully")
        return predictions
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

def save_predictions(
    input_df: pd.DataFrame, 
    predictions: np.ndarray, 
    output_path: str
) -> None:
    """
    Save the predictions along with relevant input data.
    
    Args:
        input_df: Original input DataFrame
        predictions: Array of predictions
        output_path: Path to save the predictions
    """
    logger.info(f"Saving predictions to {output_path}...")
    
    # Create a copy of the input data
    result_df = input_df.copy()
    
    # Add predictions
    result_df['predicted_occupancy'] = predictions
    
    # Calculate predicted occupancy rate
    result_df['predicted_occupancy_rate'] = result_df['predicted_occupancy'] / result_df['capacity']
    result_df['predicted_occupancy_rate'] = result_df['predicted_occupancy_rate'].clip(0, 1)
    
    # Add timestamp of when the prediction was made
    result_df['prediction_timestamp'] = datetime.now()
    
    # Save to CSV
    result_df.to_csv(output_path, index=False)
    
    logger.info(f"Saved predictions for {len(result_df)} rows")
    
    # Log some basic stats
    logger.info("Prediction summary:")
    logger.info(f"  - Min predicted occupancy: {result_df['predicted_occupancy'].min():.2f}")
    logger.info(f"  - Max predicted occupancy: {result_df['predicted_occupancy'].max():.2f}")
    logger.info(f"  - Avg predicted occupancy: {result_df['predicted_occupancy'].mean():.2f}")
    logger.info(f"  - Avg predicted occupancy rate: {result_df['predicted_occupancy_rate'].mean():.2f}")

def main(model_path: str, input_path: str, output_path: str):
    """
    Main function to run the batch prediction pipeline.
    
    Args:
        model_path: Path to the trained model file
        input_path: Path to the input data file
        output_path: Path to save the predictions
    """
    try:
        # 1. Load the data
        logger.info(f"Loading input data from {input_path}...")
        input_df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(input_df)} rows and {len(input_df.columns)} columns")
        
        # 2. Load the model
        model = load_model(model_path)
        
        # 3. Prepare features
        features_df = prepare_features(input_df)
        
        # 4. Make predictions
        predictions = make_predictions(model, features_df)
        
        # 5. Save predictions
        save_predictions(input_df, predictions, output_path)
        
        logger.info("Batch prediction completed successfully")
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction with Random Forest model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model pickle file"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the predictions CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model_predictions/",
        help="Directory to save predictions (used if --output not provided)"
    )
    
    args = parser.parse_args()
    
    # If output is not a complete path, create it in the output directory
    output_path = args.output
    if not os.path.dirname(output_path):
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output_dir, f"predictions_{timestamp}.csv")
    
    main(args.model, args.input, output_path) 