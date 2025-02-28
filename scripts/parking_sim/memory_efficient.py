"""Module for memory-efficient data processing."""

import functools
import logging
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Union

# Try to import dask, but provide fallbacks if not available
try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages memory-efficient operations."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize memory manager.
        
        Args:
            config: Configuration for memory management
        """
        self.config = config or {}
        self.use_dask = DASK_AVAILABLE and self.config.get('use_dask', True)
        
    def process_large_dataframe(self, df: pd.DataFrame, func: Callable) -> pd.DataFrame:
        """
        Process a large dataframe efficiently.
        
        Args:
            df: Input dataframe
            func: Function to apply to dataframe
            
        Returns:
            Processed dataframe
        """
        if self.use_dask and len(df) > 100000:
            # Use dask for large dataframes if available
            try:
                ddf = dd.from_pandas(df, npartitions=10)
                result = ddf.map_partitions(func).compute()
                return result
            except Exception as e:
                logger.warning(f"Dask processing failed: {e}. Falling back to pandas.")
                return func(df)
        else:
            # Use pandas for smaller dataframes or if dask is not available
            return func(df)
        
    def chunk_process(self, data: np.ndarray, func: Callable, chunk_size: int = 10000) -> np.ndarray:
        """
        Process data in chunks to reduce memory usage.
        
        Args:
            data: Input data array
            func: Function to apply to each chunk
            chunk_size: Size of each chunk
            
        Returns:
            Processed data
        """
        results = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            processed = func(chunk)
            results.append(processed)
            
        return np.concatenate(results)

def memory_efficient(func: Callable) -> Callable:
    """
    Decorator for memory-efficient processing.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get the first argument (self) if it exists
        self_obj = args[0] if args else None
        
        # Check if the object has a memory_manager attribute
        if hasattr(self_obj, 'memory_manager'):
            memory_manager = self_obj.memory_manager
        else:
            # Create a default memory manager
            memory_manager = MemoryManager()
            
        # Execute the function with memory management
        try:
            result = func(*args, **kwargs)
            return result
        except MemoryError:
            logger.warning("Memory error encountered. Attempting to free memory and retry.")
            import gc
            gc.collect()
            
            # Try again with more aggressive memory management
            kwargs['use_chunks'] = True
            return func(*args, **kwargs)
            
    return wrapper 