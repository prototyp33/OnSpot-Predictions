"""Weather impact analyzer for assessing weather effects on parking patterns."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import stats

from ..conditions.model import WeatherCondition, WeatherForecast, WeatherSeverity
from ..conditions.repository import WeatherRepository

@dataclass
class ParkingOccupancy:
    """Value object representing parking occupancy data."""
    timestamp: datetime
    occupied_spots: int
    total_spots: int
    
    @property
    def occupancy_rate(self) -> float:
        """Calculate occupancy rate as percentage."""
        return (self.occupied_spots / self.total_spots) * 100 if self.total_spots > 0 else 0

@dataclass
class WeatherImpactMetrics:
    """Value object representing weather impact analysis metrics."""
    correlation_coefficient: float
    p_value: float
    impact_score: float
    confidence_level: float
    sample_size: int

class WeatherImpactAnalyzer:
    """Analyzes the impact of weather conditions on parking patterns."""
    
    def __init__(self, weather_repository: WeatherRepository):
        self.weather_repository = weather_repository
        self._correlation_threshold = 0.3
        self._confidence_threshold = 0.95
        self._min_sample_size = 30
    
    def analyze_location_impact(self, location_id: str,
                              occupancy_data: List[ParkingOccupancy],
                              time_window: timedelta = timedelta(days=30)) -> Optional[WeatherImpactMetrics]:
        """Analyze weather impact on parking patterns for a specific location."""
        if not occupancy_data or len(occupancy_data) < self._min_sample_size:
            return None
        
        # Get weather forecast for the location
        forecast = self.weather_repository.get_forecast(location_id)
        if not forecast:
            return None
        
        # Prepare data for analysis
        weather_severity = []
        occupancy_rates = []
        
        for occupancy in occupancy_data:
            # Find closest weather condition in time
            closest_condition = self._find_closest_condition(
                occupancy.timestamp,
                forecast.conditions
            )
            
            if closest_condition:
                weather_severity.append(
                    closest_condition.calculate_severity().value
                )
                occupancy_rates.append(occupancy.occupancy_rate)
        
        if len(weather_severity) < self._min_sample_size:
            return None
        
        # Calculate correlation and significance
        correlation_coef, p_value = stats.pearsonr(
            weather_severity,
            occupancy_rates
        )
        
        # Calculate impact score (normalized between 0 and 1)
        impact_score = abs(correlation_coef)
        
        # Calculate confidence level
        confidence_level = 1 - p_value
        
        return WeatherImpactMetrics(
            correlation_coefficient=correlation_coef,
            p_value=p_value,
            impact_score=impact_score,
            confidence_level=confidence_level,
            sample_size=len(weather_severity)
        )
    
    def get_impact_recommendations(self, metrics: WeatherImpactMetrics) -> List[str]:
        """Generate recommendations based on weather impact analysis."""
        recommendations = []
        
        # Check correlation strength and direction
        if abs(metrics.correlation_coefficient) >= self._correlation_threshold:
            if metrics.correlation_coefficient > 0:
                recommendations.append(
                    "Strong positive correlation between weather severity and parking occupancy. "
                    "Consider increasing capacity during severe weather."
                )
            else:
                recommendations.append(
                    "Strong negative correlation between weather severity and parking occupancy. "
                    "Consider reducing capacity during severe weather."
                )
        
        # Check confidence level
        if metrics.confidence_level >= self._confidence_threshold:
            recommendations.append(
                "High confidence in weather impact analysis. "
                "Implement weather-based capacity adjustments."
            )
        else:
            recommendations.append(
                "Low confidence in weather impact analysis. "
                "Collect more data before making significant changes."
            )
        
        # Check sample size
        if metrics.sample_size < self._min_sample_size * 2:
            recommendations.append(
                f"Limited sample size ({metrics.sample_size} samples). "
                "Consider collecting more data for more reliable analysis."
            )
        
        return recommendations
    
    def predict_occupancy_impact(self, location_id: str,
                               current_occupancy: ParkingOccupancy,
                               forecast: WeatherForecast) -> Optional[float]:
        """Predict the impact of forecasted weather on parking occupancy."""
        # Get historical impact metrics
        historical_occupancy = self._get_historical_occupancy(
            location_id,
            current_occupancy.timestamp - timedelta(days=30),
            current_occupancy.timestamp
        )
        
        if not historical_occupancy:
            return None
        
        impact_metrics = self.analyze_location_impact(
            location_id,
            historical_occupancy
        )
        
        if not impact_metrics or impact_metrics.confidence_level < self._confidence_threshold:
            return None
        
        # Calculate predicted impact based on weather severity
        next_condition = forecast.conditions[0] if forecast.conditions else None
        if not next_condition:
            return None
        
        severity = next_condition.calculate_severity().value
        baseline_occupancy = current_occupancy.occupancy_rate
        
        # Apply correlation coefficient to predict change
        predicted_change = (
            severity * impact_metrics.correlation_coefficient *
            (baseline_occupancy / 100)
        )
        
        return max(0, min(100, baseline_occupancy + predicted_change))
    
    def _find_closest_condition(self, target_time: datetime,
                              conditions: List[WeatherCondition]) -> Optional[WeatherCondition]:
        """Find the weather condition closest to the target time."""
        if not conditions:
            return None
        
        closest_condition = min(
            conditions,
            key=lambda c: abs((c.recorded_at - target_time).total_seconds())
        )
        
        # Only return if within 1 hour
        if abs((closest_condition.recorded_at - target_time).total_seconds()) <= 3600:
            return closest_condition
        return None
    
    def _get_historical_occupancy(self, location_id: str,
                                start_time: datetime,
                                end_time: datetime) -> List[ParkingOccupancy]:
        """Mock method to get historical occupancy data.
        In a real implementation, this would fetch data from a database."""
        # This is a placeholder that should be replaced with actual data access
        return []

class WeatherImpactService:
    """Application service for weather impact analysis."""
    
    def __init__(self, analyzer: WeatherImpactAnalyzer):
        self.analyzer = analyzer
    
    def analyze_current_impact(self, location_id: str,
                             current_occupancy: ParkingOccupancy) -> Dict:
        """Analyze current weather impact and provide recommendations."""
        # Get historical occupancy for the past 30 days
        historical_occupancy = self.analyzer._get_historical_occupancy(
            location_id,
            current_occupancy.timestamp - timedelta(days=30),
            current_occupancy.timestamp
        )
        
        # Analyze impact
        impact_metrics = self.analyzer.analyze_location_impact(
            location_id,
            historical_occupancy
        )
        
        if not impact_metrics:
            return {
                'status': 'insufficient_data',
                'message': 'Not enough data for impact analysis'
            }
        
        # Get recommendations
        recommendations = self.analyzer.get_impact_recommendations(impact_metrics)
        
        return {
            'status': 'success',
            'metrics': {
                'correlation': impact_metrics.correlation_coefficient,
                'confidence': impact_metrics.confidence_level,
                'impact_score': impact_metrics.impact_score
            },
            'recommendations': recommendations
        }
    
    def get_weather_adjusted_capacity(self, location_id: str,
                                    current_occupancy: ParkingOccupancy,
                                    forecast: WeatherForecast) -> Dict:
        """Calculate weather-adjusted parking capacity."""
        predicted_occupancy = self.analyzer.predict_occupancy_impact(
            location_id,
            current_occupancy,
            forecast
        )
        
        if predicted_occupancy is None:
            return {
                'status': 'prediction_failed',
                'message': 'Unable to predict weather impact'
            }
        
        # Calculate adjusted capacity
        current_rate = current_occupancy.occupancy_rate
        adjustment_factor = predicted_occupancy / current_rate if current_rate > 0 else 1
        
        return {
            'status': 'success',
            'current_occupancy': current_rate,
            'predicted_occupancy': predicted_occupancy,
            'adjustment_factor': adjustment_factor,
            'recommended_capacity': int(
                current_occupancy.total_spots * adjustment_factor
            )
        } 