"""
Main training script for the parking occupancy prediction model.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import logging
import os
from datetime import datetime
import joblib
import json
from typing import Dict, List, Optional
import sys

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Debug information about environment
logger.debug(f"Python path: {sys.path}")
logger.debug(f"Current working directory: {os.getcwd()}")
logger.debug(f"Directory contents: {os.listdir('.')}")

try:
    from .advanced_features import engineer_advanced_features
    from .sliding_window import SlidingWindowTrainer
    from .validation import validate_data
    logger.debug("Successfully imported local modules")
except ImportError as e:
    logger.error(f"Failed to import local modules: {e}")
    raise

def load_and_preprocess_data(data_path: str, config: Dict) -> pd.DataFrame:
    """
    Load and preprocess data.
    
    Parameters:
    -----------
    data_path : str
        Path to data file
    config : Dict
        Configuration dictionary
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed dataframe
    """
    logger.debug(f"Attempting to load data from: {data_path}")
    logger.debug(f"Config contents: {config}")
    
    # Check if data file exists
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.debug(f"Loaded dataframe shape: {df.shape}")
    logger.debug(f"Dataframe columns: {df.columns.tolist()}")
    
    # Convert integer columns that should be categorical to categorical type
    categorical_columns = config['validation']['categorical_columns']
    for col in categorical_columns:
        if col in df.columns and pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype('category')
            logger.debug(f"Converted {col} to categorical type")
    
    # Add detailed logging for tipo values
    logger.debug(f"Unique values in 'tipo' column: {df['tipo'].unique().tolist()}")
    logger.debug(f"Value counts for 'tipo' column:\n{df['tipo'].value_counts()}")
    
    # Log validation rules
    logger.debug(f"Validation rules: {config['validation']}")
    
    # Convert datetime to pandas datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Validate data
    try:
        validate_data(df, config['validation'])
        logger.debug("Data validation successful")
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise
    
    # Engineer features
    try:
        df = engineer_advanced_features(df)
        logger.debug(f"Feature engineering complete. Final shape: {df.shape}")
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise
    
    return df

def prepare_features(
    df: pd.DataFrame,
    config: Dict
) -> tuple:
    """
    Prepare features for training.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    config : Dict
        Configuration dictionary
        
    Returns:
    --------
    tuple
        (feature_columns, target_column, scaler, processed_df)
    """
    # Get feature columns from config
    numeric_features = config['features']['numeric_features']
    binary_features = config['features']['binary_features']
    categorical_features = config['features']['categorical_features']
    
    # Create a copy of the dataframe for preprocessing
    df_processed = df.copy()
    
    # One-hot encode categorical features
    for cat_feature in categorical_features:
        if cat_feature in df_processed.columns:
            cat_dummies = pd.get_dummies(
                df_processed[cat_feature], 
                prefix=cat_feature,
                drop_first=False  # Keep all categories for interpretability
            )
            df_processed = pd.concat([df_processed, cat_dummies], axis=1)
    
    # Scale numeric features
    scaler = StandardScaler()
    if numeric_features:
        df_processed[numeric_features] = scaler.fit_transform(df_processed[numeric_features])
    
    # Combine all feature columns
    encoded_categorical_features = [col for col in df_processed.columns 
                                  if any(col.startswith(f"{cat}_") for cat in categorical_features)]
    
    final_feature_columns = numeric_features + binary_features + encoded_categorical_features
    
    # Create final dataframe with features and required columns
    required_columns = ['datetime']  # Add any other required columns here
    final_columns = required_columns + final_feature_columns
    
    # Log feature information
    logger.info(f"Numeric features: {numeric_features}")
    logger.info(f"Binary features: {binary_features}")
    logger.info(f"Categorical features encoded: {encoded_categorical_features}")
    logger.info(f"Total features: {len(final_feature_columns)}")
    logger.info(f"Additional required columns: {required_columns}")
    
    return final_feature_columns, config['target_column'], scaler, df_processed[final_columns]

def train_model(
    df: pd.DataFrame,
    config: Dict,
    output_dir: str
) -> None:
    """
    Train the parking occupancy prediction model.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed dataframe
    config : Dict
        Configuration dictionary
    output_dir : str
        Directory to save model artifacts
    """
    # Prepare features
    feature_columns, target_column, scaler, df_processed = prepare_features(df, config)
    
    # Initialize base model
    base_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=42
    )
    
    # Initialize sliding window trainer
    trainer = SlidingWindowTrainer(
        base_model=base_model,
        window_size=config['model_params']['window_size'],
        step_size=config['model_params']['step_size'],
        min_samples=config['model_params']['min_samples'],
        feature_columns=feature_columns,
        target_column=target_column
    )
    
    # Train models
    logger.info("Training models...")
    trainer.fit(df_processed)
    
    # Save models and artifacts
    save_artifacts(
        trainer=trainer,
        scaler=scaler,
        config=config,
        feature_columns=feature_columns,
        output_dir=output_dir
    )

def save_artifacts(
    trainer: SlidingWindowTrainer,
    scaler: StandardScaler,
    config: Dict,
    feature_columns: List[str],
    output_dir: str
) -> None:
    """
    Save model artifacts.
    
    Parameters:
    -----------
    trainer : SlidingWindowTrainer
        Trained sliding window model
    scaler : StandardScaler
        Fitted feature scaler
    config : Dict
        Configuration dictionary
    feature_columns : List[str]
        List of feature column names
    output_dir : str
        Output directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save models
    models_dir = os.path.join(output_dir, 'models')
    trainer.save_models(models_dir)
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    # Save feature information
    feature_info = {
        'numeric_features': [col for col in feature_columns 
                           if not (col.startswith(('tipo_', 'tarifa_', 'location_cluster_')) or 
                                 col in ['morning_peak', 'midday', 'evening_peak', 'night',
                                       'business_hours', 'workday', 'monday', 'friday',
                                       'saturday', 'sunday', 'is_holiday', 'high_season',
                                       'shoulder_season', 'low_season', 'school_year',
                                       'school_holiday'])],
        'categorical_features': {
            'tipo': [col for col in feature_columns if col.startswith('tipo_')],
            'tarifa': [col for col in feature_columns if col.startswith('tarifa_')],
            'location_cluster': [col for col in feature_columns if col.startswith('location_cluster_')]
        },
        'binary_features': [col for col in feature_columns 
                          if col in ['morning_peak', 'midday', 'evening_peak', 'night',
                                   'business_hours', 'workday', 'monday', 'friday',
                                   'saturday', 'sunday', 'is_holiday', 'high_season',
                                   'shoulder_season', 'low_season', 'school_year',
                                   'school_holiday']]
    }
    
    features_path = os.path.join(output_dir, 'feature_info.json')
    with open(features_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    # Save all feature columns for reference
    all_features_path = os.path.join(output_dir, 'feature_columns.txt')
    with open(all_features_path, 'w') as f:
        f.write('\n'.join(feature_columns))
    
    # Save training metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'config': config,
        'feature_columns': feature_columns,
        'preprocessing': {
            'scaler': 'StandardScaler',
            'categorical_encoding': 'one-hot',
            'binary_features': 'no_transformation'
        }
    }
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"Saved model artifacts to {output_dir}")
    logger.info("Feature types saved:")
    logger.info(f"- Numeric features: {len(feature_info['numeric_features'])}")
    logger.info(f"- Binary features: {len(feature_info['binary_features'])}")
    for cat, features in feature_info['categorical_features'].items():
        logger.info(f"- {cat} categories: {len(features)}")

def main():
    """Main training function."""
    # Load configuration
    config_path = os.getenv('CONFIG_PATH', 'config/training_config.json')
    with open(config_path) as f:
        config = json.load(f)
    
    # Set up output directory
    output_dir = os.path.join(
        config['data']['output_path'],
        f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Load and preprocess data
    df = load_and_preprocess_data(config['data']['input_path'], config)
    
    # Train model
    train_model(df, config, output_dir)
    
if __name__ == '__main__':
    main() 