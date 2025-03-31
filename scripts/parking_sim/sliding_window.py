"""
Sliding window training implementation for parking occupancy prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.base import BaseEstimator, clone
from typing import Optional, Tuple, List, Union, Dict
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

class SlidingWindowTrainer:
    """
    Implements sliding window training for time series prediction.
    """
    
    def __init__(
        self,
        base_model: BaseEstimator,
        window_size: int = 24,  # 24 hours default window
        step_size: int = 1,    # 1 hour step size
        min_samples: int = 1000,
        feature_columns: List[str] = None,
        target_column: str = 'occupancy_rate',
        validation_split: float = 0.2  # Add validation split parameter
    ):
        """
        Initialize the sliding window trainer.
        
        Parameters:
        -----------
        base_model : BaseEstimator
            The base model to use for training (will be cloned for each window)
        window_size : int
            Size of the training window in hours
        step_size : int
            Number of hours to slide the window forward
        min_samples : int
            Minimum number of samples required for training
        feature_columns : List[str]
            List of feature column names
        target_column : str
            Name of the target column
        validation_split : float
            Fraction of data to use for validation (default: 0.2)
        """
        self.base_model = base_model
        self.window_size = window_size
        self.step_size = step_size
        self.min_samples = min_samples
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.validation_split = validation_split  # Store validation split
        self.models = {}  # Store models for each window
        self.window_weights = {}  # Store weights for each window
        
    def create_windows(self, df: pd.DataFrame, date_column: str = 'datetime') -> List[pd.DataFrame]:
        """
        Create sliding windows from the dataframe.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        date_column : str, optional
            Name of the datetime column, by default 'datetime'
            
        Returns:
        --------
        List[pd.DataFrame]
            List of dataframes, each representing a window
        """
        # Ensure datetime column is properly formatted
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        
        # Log initial data info
        logger.debug(f"Creating windows from {len(df)} records")
        logger.debug(f"Date range: {df[date_column].min()} to {df[date_column].max()}")
        
        # Sort dataframe by date
        df = df.sort_values(by=date_column).reset_index(drop=True)
        
        # Get date range
        start_date = df[date_column].min()
        end_date = df[date_column].max()
        
        # Calculate window parameters in hours
        window_hours = self.window_size  # Window size is already in hours
        step_hours = self.step_size  # Step size is already in hours
        
        logger.debug(f"Total time range: {(end_date - start_date).total_seconds() / 3600:.1f} hours")
        logger.debug(f"Window size: {window_hours} hours")
        logger.debug(f"Step size: {step_hours} hours")
        
        # Create windows
        windows = []
        current_start = start_date
        
        while current_start + pd.Timedelta(hours=window_hours) <= end_date:
            current_end = current_start + pd.Timedelta(hours=window_hours)
            
            # Get data for current window
            window_mask = (df[date_column] >= current_start) & (df[date_column] < current_end)
            window_df = df[window_mask].copy()
            
            logger.debug(f"Window {len(windows)+1}:")
            logger.debug(f"  Start: {current_start}")
            logger.debug(f"  End: {current_end}")
            logger.debug(f"  Records: {len(window_df)}")
            
            # Only keep windows with sufficient samples
            if len(window_df) >= self.min_samples:
                windows.append(window_df)
                logger.debug(f"  Status: Added (sufficient samples)")
            else:
                logger.debug(f"  Status: Skipped (insufficient samples, needed {self.min_samples})")
            
            # Move to next window
            current_start += pd.Timedelta(hours=step_hours)
        
        logger.info(f"Created {len(windows)} windows")
        if windows:
            logger.debug("Window statistics:")
            logger.debug(f"  Average window size: {np.mean([len(w) for w in windows]):.0f} records")
            logger.debug(f"  Min window size: {min(len(w) for w in windows)} records")
            logger.debug(f"  Max window size: {max(len(w) for w in windows)} records")
        
        return windows
    
    def train_window(
        self,
        window_df: pd.DataFrame,
        date_column: str = 'datetime'
    ) -> Tuple[BaseEstimator, Dict[str, float]]:
        """
        Train model on a single window.
        
        Parameters:
        -----------
        window_df : pandas.DataFrame
            Window dataframe
        date_column : str, optional
            Name of the datetime column, by default 'datetime'
            
        Returns:
        --------
        Tuple[BaseEstimator, Dict[str, float]]
            Trained model and metrics
        """
        # Sort by date within window
        window_df = window_df.sort_values(by=date_column)
        
        # Split features and target
        X = window_df[self.feature_columns]
        y = window_df[self.target_column]
        
        # Get validation indices
        n_samples = len(window_df)
        n_val = int(n_samples * self.validation_split)
        train_idx = np.arange(n_samples - n_val)
        val_idx = np.arange(n_samples - n_val, n_samples)
        
        # Split data
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        # Train model
        model = clone(self.base_model)
        model.fit(X_train, y_train)
        
        # Calculate metrics
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'train_r2': r2_score(y_train, train_pred),
            'val_r2': r2_score(y_val, val_pred),
            'window_start': window_df[date_column].min(),
            'window_end': window_df[date_column].max(),
            'n_samples': n_samples
        }
        
        return model, metrics
    
    def fit(
        self,
        df: pd.DataFrame,
        date_column: str = 'datetime'
    ) -> 'SlidingWindowTrainer':
        """
        Train models on all windows.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        date_column : str
            Name of the datetime column
            
        Returns:
        --------
        self
        """
        logger.info("Creating sliding windows...")
        windows = self.create_windows(df, date_column)
        logger.info(f"Created {len(windows)} windows")
        
        if not windows:
            logger.warning("No windows were created. Check window_size, step_size, and min_samples parameters.")
            return self
        
        logger.info("Training models on each window...")
        for i, window_df in enumerate(windows, 1):
            logger.info(f"Training model for window {i}/{len(windows)}")
            logger.info(f"Window period: {window_df[date_column].min()} to {window_df[date_column].max()}")
            logger.info(f"Window size: {len(window_df)} samples")
            
            model, metrics = self.train_window(window_df, date_column)
            window_key = tuple(window_df[date_column].values[[0, -1]])
            self.models[window_key] = model
            self.window_weights[window_key] = metrics
            
            logger.info(f"Window {i} metrics:")
            logger.info(f"  Train RMSE: {metrics['train_rmse']:.4f}")
            logger.info(f"  Val RMSE: {metrics['val_rmse']:.4f}")
            logger.info(f"  Train R2: {metrics['train_r2']:.4f}")
            logger.info(f"  Val R2: {metrics['val_r2']:.4f}")
                
        # Normalize weights
        if self.window_weights:
            logger.info("Normalizing window weights...")
            total_weight = sum(self.window_weights.values())
            if total_weight > 0:
                self.window_weights = {
                    k: v/total_weight for k, v in self.window_weights.items()
                }
            logger.info("Window weights normalized")
        else:
            logger.warning("No window weights to normalize")
            
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        date_column: str = 'datetime'
    ) -> np.ndarray:
        """
        Make predictions using weighted ensemble of window models.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input features
        date_column : str
            Name of the datetime column
            
        Returns:
        --------
        numpy.ndarray
            Weighted average predictions
        """
        if not self.models:
            raise ValueError("No models trained. Call fit() first.")
            
        predictions = np.zeros((len(X), len(self.models)))
        
        for i, ((start_date, end_date), model) in enumerate(self.models.items()):
            pred = model.predict(X[self.feature_columns])
            predictions[:, i] = pred * self.window_weights[(start_date, end_date)]
            
        return np.sum(predictions, axis=1)
    
    def save_models(self, directory: str):
        """Save all window models to disk."""
        os.makedirs(directory, exist_ok=True)
        
        for (start_date, end_date), model in self.models.items():
            filename = f"model_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl"
            joblib.dump(model, os.path.join(directory, filename))
            
        # Save window weights
        weights_file = os.path.join(directory, "window_weights.pkl")
        joblib.dump(self.window_weights, weights_file)
        
    def load_models(self, directory: str):
        """Load all window models from disk."""
        self.models = {}
        
        # Load window weights
        weights_file = os.path.join(directory, "window_weights.pkl")
        if os.path.exists(weights_file):
            self.window_weights = joblib.load(weights_file)
        
        for filename in os.listdir(directory):
            if filename.startswith("model_") and filename.endswith(".pkl"):
                # Parse dates from filename
                dates = filename[6:-4].split("_")
                start_date = datetime.strptime(dates[0], "%Y%m%d")
                end_date = datetime.strptime(dates[1], "%Y%m%d")
                
                model_path = os.path.join(directory, filename)
                self.models[(start_date, end_date)] = joblib.load(model_path) 