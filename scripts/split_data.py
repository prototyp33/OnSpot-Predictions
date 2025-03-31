import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import logging
from typing import Tuple, Dict
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def identify_feature_types(df: pd.DataFrame) -> Tuple[list, list]:
    """
    Identify numerical and categorical columns in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Tuple[list, list]: Lists of numerical and categorical column names
    """
    # Exclude specific columns from features
    exclude_cols = ['timestamp', 'date', 'occupancy']
    
    # Identify numerical and categorical columns
    numerical_cols = []
    categorical_cols = []
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        
        if np.issubdtype(df[col].dtype, np.number):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
            
    return numerical_cols, categorical_cols

def create_preprocessor(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for numerical and categorical features.
    
    Args:
        numerical_cols (list): List of numerical column names
        categorical_cols (list): List of categorical column names
        
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor

def time_based_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train, validation, and test sets based on timestamp.
    
    Args:
        df (pd.DataFrame): Input dataframe
        train_ratio (float): Proportion of data for training
        val_ratio (float): Proportion of data for validation
        test_ratio (float): Proportion of data for testing
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test sets
    """
    # Ensure the dataframe is sorted by timestamp
    df = df.sort_values('timestamp')
    
    # Calculate split points
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Split the data
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    # Log split information
    logger.info(f"Train set: {len(train_df)} samples ({train_ratio*100:.1f}%)")
    logger.info(f"Validation set: {len(val_df)} samples ({val_ratio*100:.1f}%)")
    logger.info(f"Test set: {len(test_df)} samples ({test_ratio*100:.1f}%)")
    
    # Log date ranges
    logger.info(f"Train period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    logger.info(f"Validation period: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
    logger.info(f"Test period: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    
    return train_df, val_df, test_df

def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
    preprocessor: ColumnTransformer = None
) -> None:
    """
    Save the train, validation, and test splits to disk.
    
    Args:
        train_df (pd.DataFrame): Training set
        val_df (pd.DataFrame): Validation set
        test_df (pd.DataFrame): Test set
        output_dir (str): Directory to save the splits
        preprocessor (ColumnTransformer, optional): Fitted preprocessor to save
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the splits
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'validation.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    # Save preprocessor if provided
    if preprocessor is not None:
        joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.pkl'))
    
    logger.info(f"Splits saved to {output_dir}")

def main(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> None:
    """
    Main function to split the dataset and save the results.
    
    Args:
        input_path (str): Path to the preprocessed dataset
        output_dir (str): Directory to save the splits
        train_ratio (float): Proportion of data for training
        val_ratio (float): Proportion of data for validation
        test_ratio (float): Proportion of data for testing
    """
    logger.info("Starting data splitting process...")
    
    # Load the data
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Identify feature types
    numerical_cols, categorical_cols = identify_feature_types(df)
    logger.info(f"Found {len(numerical_cols)} numerical features and {len(categorical_cols)} categorical features")
    
    # Create preprocessor
    preprocessor = create_preprocessor(numerical_cols, categorical_cols)
    
    # Split the data
    train_df, val_df, test_df = time_based_split(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    # Save the splits and preprocessor
    save_splits(train_df, val_df, test_df, output_dir, preprocessor)
    
    logger.info("Data splitting completed successfully")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Split preprocessed parking data into train, validation, and test sets')
    parser.add_argument('--input-path', type=str, required=True, help='Path to preprocessed dataset')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the splits')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Proportion of data for training')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Proportion of data for validation')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Proportion of data for testing')
    
    args = parser.parse_args()
    
    main(
        input_path=args.input_path,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    ) 