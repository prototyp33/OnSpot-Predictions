"""
Model Performance Metrics Calculator
Handles calculation of various model performance metrics with error handling
"""

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dataclasses import dataclass
from datetime import datetime
import logging
import time
from abc import ABC, abstractmethod
from joblib import Parallel, delayed
import scipy.stats

logger = logging.getLogger(__name__)

class MetricCalculator(ABC):
    """Abstract base class for metric calculators"""
    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> float:
        pass

class RMSECalculator(MetricCalculator):
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> float:
        return np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weights))

class MAECalculator(MetricCalculator):
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> float:
        return mean_absolute_error(y_true, y_pred, sample_weight=sample_weights)

class R2Calculator(MetricCalculator):
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> float:
        return r2_score(y_true, y_pred, sample_weight=sample_weights)

class MAPECalculator(MetricCalculator):
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> float:
        mask = y_true != 0
        if not np.any(mask):
            return float('inf')
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

class MetricsFactory:
    """Factory for creating metric calculators"""
    def __init__(self):
        self._calculators = {}
        self._register_default_calculators()
    
    def _register_default_calculators(self):
        self.register_calculator('rmse', RMSECalculator())
        self.register_calculator('mae', MAECalculator())
        self.register_calculator('r2', R2Calculator())
        self.register_calculator('mape', MAPECalculator())
    
    def register_calculator(self, name: str, calculator: MetricCalculator):
        self._calculators[name] = calculator
    
    def get_calculator(self, name: str) -> MetricCalculator:
        if name not in self._calculators:
            raise ValueError(f"No calculator registered for metric '{name}'")
        return self._calculators[name]
    
    def items(self):
        return self._calculators.items()

@dataclass
class MetricResult:
    """Container for metric calculation results with metadata"""
    value: float
    timestamp: datetime
    sample_size: int
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Optional[Dict] = None

class MetricsCalculationError(Exception):
    """Custom exception for metrics calculation errors"""
    pass

class ModelMetricsCalculator:
    """Handles calculation of various model performance metrics"""
    
    def __init__(self, model_id: str, metrics_factory: Optional[MetricsFactory] = None):
        self.model_id = model_id
        self.metrics_factory = metrics_factory or MetricsFactory()
        self.valid_range = (-float('inf'), float('inf'))  # Can be set by user
    
    def calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> Dict[str, MetricResult]:
        """
        Calculate comprehensive regression metrics with enhanced logging
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            sample_weights: Optional weights for weighted metrics
            
        Returns:
            Dictionary of metric names to MetricResult objects
            
        Raises:
            MetricsCalculationError: If inputs are invalid or calculation fails
        """
        logger.info(f"Starting regression metrics calculation for model {self.model_id}")
        start_time = time.time()
        
        try:
            # Sanitize and validate inputs
            y_true = self._sanitize_input(y_true)
            y_pred = self._sanitize_input(y_pred)
            self._validate_inputs(y_true, y_pred)
            
            current_time = datetime.now()
            sample_size = len(y_true)
            metrics = {}
            
            # Calculate basic metrics using factory
            for metric_name, calculator in self.metrics_factory.items():
                metric_start = time.time()
                
                value = calculator.calculate(y_true, y_pred, sample_weights)
                confidence_interval = self._bootstrap_confidence_interval(
                    y_true, y_pred, calculator.calculate, n_iterations=1000
                ) if metric_name in ['rmse', 'mae'] else None
                
                metrics[metric_name] = MetricResult(
                    value=value,
                    timestamp=current_time,
                    sample_size=sample_size,
                    confidence_interval=confidence_interval
                )
                
                metric_duration = time.time() - metric_start
                logger.debug(f"Calculated {metric_name} in {metric_duration:.2f}s")
            
            # Calculate advanced metrics
            advanced_metrics = self.calculate_advanced_metrics(y_true, y_pred)
            metrics.update(advanced_metrics)
            
            total_duration = time.time() - start_time
            logger.info(f"Completed metrics calculation in {total_duration:.2f}s")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}", exc_info=True)
            raise MetricsCalculationError(f"Failed to calculate metrics: {str(e)}")
    
    def _sanitize_input(self, data: np.ndarray) -> np.ndarray:
        """Sanitize input data for security"""
        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        # Convert to float64 to prevent overflow
        data = data.astype(np.float64)
        
        # Replace invalid values
        data = np.nan_to_num(data, nan=0.0, posinf=None, neginf=None)
        
        return data
    
    def _validate_inputs(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Enhanced input validation with comprehensive checks"""
        # Basic checks
        if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
            raise TypeError("Inputs must be numpy arrays")
        
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}")
        
        if len(y_true) == 0:
            raise ValueError("Empty arrays provided")
        
        # Data quality checks
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            raise ValueError("Arrays contain NaN values")
        
        if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
            raise ValueError("Arrays contain infinite values")
        
        # Statistical validity checks
        if np.std(y_true) == 0:
            raise ValueError("Zero variance in true values")
        
        # Range checks
        if np.any(y_pred < self.valid_range[0]) or np.any(y_pred > self.valid_range[1]):
            raise ValueError(f"Predictions outside valid range {self.valid_range}")
    
    def _bootstrap_confidence_interval(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_func: callable,
        n_iterations: int = 1000,
        confidence_level: float = 0.95,
        n_jobs: int = -1
    ) -> Tuple[float, float]:
        """Calculate confidence intervals using parallel bootstrapping"""
        def single_bootstrap():
            indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
            return metric_func(y_true[indices], y_pred[indices])
        
        bootstrap_estimates = Parallel(n_jobs=n_jobs)(
            delayed(single_bootstrap)() for _ in range(n_iterations)
        )
        
        percentiles = [(1 - confidence_level) / 2, 1 - (1 - confidence_level) / 2]
        return tuple(np.percentile(bootstrap_estimates, [p * 100 for p in percentiles]))
    
    def calculate_advanced_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, MetricResult]:
        """Calculate advanced statistical metrics"""
        metrics = {}
        current_time = datetime.now()
        sample_size = len(y_true)
        
        # Residual analysis
        residuals = y_pred - y_true
        metrics['residual_normality'] = MetricResult(
            value=scipy.stats.normaltest(residuals)[1],
            timestamp=current_time,
            sample_size=sample_size,
            metadata={'test': 'normaltest'}
        )
        
        # Heteroscedasticity test (Breusch-Pagan)
        def _breusch_pagan_test(y_true, residuals):
            # Simple version - correlation between squared residuals and predicted values
            squared_resid = residuals ** 2
            correlation = np.corrcoef(squared_resid, y_true)[0, 1]
            return abs(correlation)
        
        metrics['heteroscedasticity'] = MetricResult(
            value=_breusch_pagan_test(y_true, residuals),
            timestamp=current_time,
            sample_size=sample_size,
            metadata={'test': 'breusch_pagan'}
        )
        
        # Durbin-Watson test for autocorrelation
        def _durbin_watson_test(residuals):
            diff = np.diff(residuals)
            return np.sum(diff ** 2) / np.sum(residuals ** 2)
        
        metrics['autocorrelation'] = MetricResult(
            value=_durbin_watson_test(residuals),
            timestamp=current_time,
            sample_size=sample_size,
            metadata={'test': 'durbin_watson'}
        )
        
        return metrics
    
    def calculate_threshold_violations(
        self,
        metric_value: float,
        threshold: float,
        comparison_operator: str = '>',
        context: Optional[Dict] = None
    ) -> Dict[str, Union[bool, float, str, Dict]]:
        """Enhanced threshold violation detection with context and severity levels"""
        operators = {
            '>': np.greater,
            '<': np.less,
            '>=': np.greater_equal,
            '<=': np.less_equal,
            '==': np.equal,
            '!=': np.not_equal
        }
        
        if comparison_operator not in operators:
            raise ValueError(f"Invalid operator. Must be one of {list(operators.keys())}")
        
        is_violated = operators[comparison_operator](metric_value, threshold)
        violation_margin = abs(metric_value - threshold) if is_violated else 0
        
        # Calculate severity level
        severity = self._calculate_severity(
            metric_value,
            threshold,
            violation_margin,
            comparison_operator
        )
        
        return {
            'is_violated': bool(is_violated),
            'current_value': metric_value,
            'threshold': threshold,
            'comparison': comparison_operator,
            'violation_margin': violation_margin,
            'severity': severity,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_severity(
        self,
        value: float,
        threshold: float,
        margin: float,
        operator: str
    ) -> str:
        """Calculate violation severity level"""
        relative_margin = margin / threshold if threshold != 0 else float('inf')
        
        if relative_margin < 0.1:
            return 'LOW'
        elif relative_margin < 0.25:
            return 'MEDIUM'
        elif relative_margin < 0.5:
            return 'HIGH'
        return 'CRITICAL' 