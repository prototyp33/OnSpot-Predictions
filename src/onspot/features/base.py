"""Base feature engineering module."""

from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any, Optional

class BaseFeatureTransformer(ABC):
    """Abstract base class for feature transformers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.feature_names: List[str] = []
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BaseFeatureTransformer':
        """Fit the transformer on training data."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using fitted parameters."""
        pass
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data."""
        return self.fit(data).transform(data)
    
    def get_feature_names(self) -> List[str]:
        """Get the names of features this transformer produces."""
        return self.feature_names

class FeaturePipeline:
    """Pipeline for chaining multiple feature transformers."""
    
    def __init__(self, transformers: List[BaseFeatureTransformer]):
        self.transformers = transformers
    
    def fit(self, data: pd.DataFrame) -> 'FeaturePipeline':
        """Fit all transformers in sequence."""
        for transformer in self.transformers:
            data = transformer.fit_transform(data)
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using all fitted transformers."""
        for transformer in self.transformers:
            data = transformer.transform(data)
        return data
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data using all transformers."""
        return self.fit(data).transform(data)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features produced by the pipeline."""
        feature_names = []
        for transformer in self.transformers:
            feature_names.extend(transformer.get_feature_names())
        return feature_names 