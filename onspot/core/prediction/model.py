"""Core domain model for parking prediction."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
from uuid import UUID, uuid4
import numpy as np

@dataclass(frozen=True)
class PredictionInterval:
    """Value object representing a prediction time interval."""
    start_time: datetime
    end_time: datetime
    
    def __post_init__(self):
        if self.start_time >= self.end_time:
            raise ValueError("End time must be after start time")
    
    @property
    def duration_hours(self) -> float:
        return (self.end_time - self.start_time).total_seconds() / 3600

@dataclass(frozen=True)
class OccupancyRate:
    """Value object representing parking occupancy rate."""
    value: float
    
    def __post_init__(self):
        if not 0 <= self.value <= 1:
            raise ValueError("Occupancy rate must be between 0 and 1")

@dataclass(frozen=True)
class Confidence:
    """Value object representing prediction confidence."""
    value: float
    
    def __post_init__(self):
        if not 0 <= self.value <= 1:
            raise ValueError("Confidence must be between 0 and 1")

@dataclass
class PredictionResult:
    """Entity representing a single prediction result."""
    id: UUID
    interval: PredictionInterval
    occupancy: OccupancyRate
    confidence: Confidence
    features: Dict[str, float]
    generated_at: datetime
    
    @classmethod
    def create(cls, interval: PredictionInterval, occupancy: float, 
               confidence: float, features: Dict[str, float]) -> 'PredictionResult':
        return cls(
            id=uuid4(),
            interval=interval,
            occupancy=OccupancyRate(occupancy),
            confidence=Confidence(confidence),
            features=features,
            generated_at=datetime.now()
        )

class ParkingPrediction:
    """Aggregate root for parking predictions."""
    
    def __init__(self, location_id: str, model_version: str):
        self.id = uuid4()
        self.location_id = location_id
        self.model_version = model_version
        self._predictions: List[PredictionResult] = []
        self._events = []
    
    def add_prediction(self, prediction: PredictionResult) -> None:
        """Add a new prediction result."""
        self._predictions.append(prediction)
        self._events.append({
            'type': 'PredictionGenerated',
            'prediction_id': prediction.id,
            'location_id': self.location_id,
            'timestamp': datetime.now()
        })
    
    def get_predictions(self, interval: Optional[PredictionInterval] = None) -> List[PredictionResult]:
        """Get predictions, optionally filtered by time interval."""
        if interval is None:
            return self._predictions.copy()
        
        return [
            pred for pred in self._predictions
            if (pred.interval.start_time >= interval.start_time and 
                pred.interval.end_time <= interval.end_time)
        ]
    
    def get_average_confidence(self) -> float:
        """Calculate average prediction confidence."""
        if not self._predictions:
            return 0.0
        return np.mean([pred.confidence.value for pred in self._predictions])
    
    def check_accuracy_threshold(self, threshold: float = 0.8) -> bool:
        """Check if predictions meet accuracy threshold."""
        avg_confidence = self.get_average_confidence()
        if avg_confidence < threshold:
            self._events.append({
                'type': 'AccuracyThresholdBreached',
                'location_id': self.location_id,
                'average_confidence': avg_confidence,
                'threshold': threshold,
                'timestamp': datetime.now()
            })
            return False
        return True
    
    @property
    def events(self) -> List[Dict]:
        """Get accumulated domain events."""
        return self._events.copy()
    
    def clear_events(self) -> None:
        """Clear accumulated events after they've been processed."""
        self._events.clear()

class OccupancyModel:
    """Domain service for occupancy predictions."""
    
    def __init__(self, model_version: str):
        self.model_version = model_version
    
    def predict(self, interval: PredictionInterval, features: Dict[str, float]) -> PredictionResult:
        """Generate a prediction for the given interval and features."""
        # This is a placeholder for the actual prediction logic
        # In a real implementation, this would use the underlying ML model
        occupancy = 0.5  # Example placeholder
        confidence = 0.8  # Example placeholder
        
        return PredictionResult.create(
            interval=interval,
            occupancy=occupancy,
            confidence=confidence,
            features=features
        )

class PredictionService:
    """Application service for managing predictions."""
    
    def __init__(self, model: OccupancyModel):
        self.model = model
        self._predictions: Dict[str, ParkingPrediction] = {}
    
    def create_prediction(self, location_id: str, interval: PredictionInterval,
                         features: Dict[str, float]) -> PredictionResult:
        """Create a new prediction for a location."""
        # Get or create prediction aggregate
        if location_id not in self._predictions:
            self._predictions[location_id] = ParkingPrediction(
                location_id=location_id,
                model_version=self.model.model_version
            )
        
        # Generate prediction
        prediction = self.model.predict(interval, features)
        
        # Add to aggregate
        self._predictions[location_id].add_prediction(prediction)
        
        return prediction
    
    def get_location_predictions(self, location_id: str,
                               interval: Optional[PredictionInterval] = None) -> List[PredictionResult]:
        """Get predictions for a specific location."""
        if location_id not in self._predictions:
            return []
        return self._predictions[location_id].get_predictions(interval)
    
    def check_accuracy_thresholds(self) -> List[str]:
        """Check accuracy thresholds for all locations."""
        breached_locations = []
        for location_id, prediction in self._predictions.items():
            if not prediction.check_accuracy_threshold():
                breached_locations.append(location_id)
        return breached_locations 