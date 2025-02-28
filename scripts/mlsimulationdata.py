import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any, Callable
from jsonschema import validate
import holidays
from scipy import stats
import requests
import time
import argparse
from performance_monitor import PerformanceMonitor

# Constants
BUSINESS_HOURS = (8, 20)
PEAK_HOURS = [(8, 10), (17, 19)]
WEEKEND_OCCUPANCY_FACTOR = 0.7
RAIN_OCCUPANCY_FACTOR = 0.85
HOLIDAY_OCCUPANCY_FACTOR = 0.6
TRAFFIC_PEAK_HOURS = {
    'morning': (7, 10),
    'evening': (16, 19)
}
TRAFFIC_BASE_LEVEL = 0.4
TRAFFIC_PEAK_FACTOR = 0.8
SPECIAL_EVENTS = {
    # Format: 'YYYY-MM-DD': ('Event Name', impact_factor)
    '2024-02-25': ('Barcelona Marathon', 0.9),
    '2024-05-15': ('Music Festival', 0.7),
    '2024-08-15': ('La Mercè Festival', 0.8),
    '2024-12-24': ('Christmas Eve', 0.6)
}
BARCELONA_WEATHER = {
    # month: (avg_temp, temp_std, avg_rain_days, avg_humidity, daylight_hours)
    1: (11.4, 2.5, 5, 72, 9.5),   # January
    2: (12.0, 2.8, 4, 70, 10.5),  # February
    3: (13.9, 3.0, 5, 70, 12),    # March
    4: (15.5, 3.2, 5, 69, 13.5),  # April
    5: (18.8, 3.5, 5, 68, 14.5),  # May
    6: (22.5, 3.8, 4, 67, 15),    # June
    7: (25.3, 3.5, 2, 66, 14.8),  # July
    8: (25.5, 3.3, 4, 68, 13.8),  # August
    9: (22.7, 3.0, 6, 71, 12.5),  # September
    10: (18.8, 2.8, 7, 74, 11),   # October
    11: (14.8, 2.5, 6, 74, 10),   # November
    12: (12.0, 2.3, 5, 72, 9)     # December
}
OUTLIER_TYPES = {
    'equipment_failure': {'probability': 0.003, 'effect': 'zero'},
    'special_event': {'probability': 0.005, 'effect': 'overflow'},
    'sensor_malfunction': {'probability': 0.002, 'effect': 'random'},
    'maintenance': {'probability': 0.004, 'effect': 'zero'},
    'emergency': {'probability': 0.001, 'effect': 'overflow'}
}
BARCELONA_PARKING_URL = "https://opendata-ajuntament.barcelona.cat/data/api/action/datastore_search"
PARKING_RESOURCE_ID = "1d6c814c-70ef-4147-aa16-a49ddb952f72"  # BSM parking facilities
BARCELONA_ZONES = {
    # zone_name: (center_lat, center_lon, radius, traffic_sensitivity)
    'business_district': (41.3874, 2.1686, 0.01, 1.5),    # Plaça Catalunya area
    'tourist_area': (41.3851, 2.1734, 0.008, 1.3),        # Las Ramblas
    'residential': (41.4034, 2.1744, 0.015, 0.7),         # Gràcia
    'shopping_district': (41.3937, 2.1647, 0.012, 1.2),   # Passeig de Gràcia
    'beach_area': (41.3751, 2.1925, 0.01, 0.9),          # Barceloneta
}
TEMP_COMFORT = {
    'optimal_temp': 22.0,    # Most comfortable temperature
    'comfort_range': 5.0,    # Standard deviation for comfort curve
    'winter_offset': -3.0,   # People tolerate cooler temps in winter
    'summer_offset': 2.0,    # People tolerate warmer temps in summer
    'impact_weight': 0.4     # Maximum impact on occupancy
}
PARKING_PATTERNS = {
    'Public': {
        'base_level': 50,
        'daily_amplitude': 30,
        'peak_hours': [(8, 10, 20), (17, 19, 15)],  # (start, end, amplitude)
        'weekend_factor': 0.7
    },
    'Resident': {
        'base_level': 70,
        'daily_amplitude': 20,
        'peak_hours': [(7, 9, -25), (18, 20, 25)],  # Negative means dip (cars leaving)
        'weekend_factor': 1.2
    },
    'Mixed': {
        'base_level': 60,
        'daily_amplitude': 25,
        'peak_hours': [(8, 10, 15), (17, 19, 20)],
        'weekend_factor': 0.9
    }
}
PARKING_DURATION = {
    'Public': {
        'short_term': {'mean_hours': 2, 'weight': 0.7},  # Shopping, errands
        'medium_term': {'mean_hours': 5, 'weight': 0.2}, # Work visits
        'long_term': {'mean_hours': 10, 'weight': 0.1}   # All-day parking
    },
    'Resident': {
        'short_term': {'mean_hours': 1, 'weight': 0.1},  # Quick home visits
        'medium_term': {'mean_hours': 8, 'weight': 0.3}, # Work day absence
        'long_term': {'mean_hours': 14, 'weight': 0.6}   # Overnight parking
    },
    'Mixed': {
        'short_term': {'mean_hours': 2, 'weight': 0.5},
        'medium_term': {'mean_hours': 6, 'weight': 0.3},
        'long_term': {'mean_hours': 12, 'weight': 0.2}
    }
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# JSON Schema for config validation
CONFIG_SCHEMA = {
    "type": "object",
    "required": [
        "data_parameters",
        "feature_ranges",
        "time_patterns",
        "occupancy_factors",
        "traffic",
        "special_events",
        "weather",
        "temperature_comfort",
        "zones",
        "parking_patterns",
        "parking_duration",
        "outlier_types",
        "api_config"
    ],
    "properties": {
        "data_parameters": {
            "type": "object",
            "required": ["num_samples", "time_range", "location"],
            "properties": {
                "num_samples": {"type": "integer", "minimum": 1},
                "time_range": {
                    "type": "object",
                    "required": ["start", "end"],
                    "properties": {
                        "start": {"type": "string", "format": "date"},
                        "end": {"type": "string", "format": "date"}
                    }
                },
                "location": {
                    "type": "object",
                    "required": ["lat_range", "lon_range"],
                    "properties": {
                        "lat_range": {"type": "array", "minItems": 2, "maxItems": 2},
                        "lon_range": {"type": "array", "minItems": 2, "maxItems": 2}
                    }
                }
            }
        },
        "feature_ranges": {
            "type": "object",
            "required": ["temperature", "humidity", "wind_speed", "precipitation", "occupancy_rate"],
            "properties": {
                "temperature": {"type": "array", "minItems": 2, "maxItems": 2},
                "humidity": {"type": "array", "minItems": 2, "maxItems": 2},
                "wind_speed": {"type": "array", "minItems": 2, "maxItems": 2},
                "precipitation": {"type": "array", "minItems": 2, "maxItems": 2},
                "occupancy_rate": {"type": "array", "minItems": 2, "maxItems": 2}
            }
        },
        "time_patterns": {
            "type": "object",
            "required": ["business_hours", "peak_hours"],
            "properties": {
                "business_hours": {
                    "type": "object",
                    "required": ["start", "end"],
                    "properties": {
                        "start": {"type": "integer", "minimum": 0, "maximum": 23},
                        "end": {"type": "integer", "minimum": 0, "maximum": 23}
                    }
                },
                "peak_hours": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["start", "end"],
                        "properties": {
                            "start": {"type": "integer", "minimum": 0, "maximum": 23},
                            "end": {"type": "integer", "minimum": 0, "maximum": 23}
                        }
                    }
                }
            }
        },
        "occupancy_factors": {
            "type": "object",
            "required": ["weekend", "rain", "holiday"],
            "properties": {
                "weekend": {"type": "number", "minimum": 0, "maximum": 2},
                "rain": {"type": "number", "minimum": 0, "maximum": 2},
                "holiday": {"type": "number", "minimum": 0, "maximum": 2}
            }
        },
        "traffic": {
            "type": "object",
            "required": ["peak_hours", "base_level", "peak_factor", "smoothing_window"],
            "properties": {
                "peak_hours": {
                    "type": "object",
                    "required": ["morning", "evening"],
                    "properties": {
                        "morning": {"type": "array", "minItems": 2, "maxItems": 2},
                        "evening": {"type": "array", "minItems": 2, "maxItems": 2}
                    }
                },
                "base_level": {"type": "number", "minimum": 0, "maximum": 1},
                "peak_factor": {"type": "number", "minimum": 0, "maximum": 2},
                "smoothing_window": {"type": "integer", "minimum": 1, "maximum": 10}
            }
        },
        "special_events": {
            "type": "object",
            "patternProperties": {
                "^[0-9]{4}-[0-9]{2}-[0-9]{2}$": {
                    "type": "object",
                    "required": ["name", "impact"],
                    "properties": {
                        "name": {"type": "string"},
                        "impact": {"type": "number", "minimum": 0, "maximum": 2}
                    }
                }
            }
        },
        "weather": {
            "type": "object",
            "required": ["monthly_patterns", "correlation"],
            "properties": {
                "monthly_patterns": {
                    "type": "object",
                    "patternProperties": {
                        "^([1-9]|1[0-2])$": {  # Matches months 1-12
                            "type": "object",
                            "required": ["avg_temp", "temp_std", "rain_days", "humidity", "daylight"],
                            "properties": {
                                "avg_temp": {"type": "number"},
                                "temp_std": {"type": "number", "minimum": 0},
                                "rain_days": {"type": "integer", "minimum": 0, "maximum": 31},
                                "humidity": {"type": "number", "minimum": 0, "maximum": 100},
                                "daylight": {"type": "number", "minimum": 0, "maximum": 24}
                            }
                        }
                    },
                    "minProperties": 12,
                    "maxProperties": 12
                },
                "correlation": {
                    "type": "object",
                    "required": ["temp_humidity", "temp_wind", "temp_rain", "humidity_rain"],
                    "properties": {
                        "temp_humidity": {"type": "number", "minimum": -1, "maximum": 1},
                        "temp_wind": {"type": "number", "minimum": -1, "maximum": 1},
                        "temp_rain": {"type": "number", "minimum": -1, "maximum": 1},
                        "humidity_rain": {"type": "number", "minimum": -1, "maximum": 1}
                    }
                }
            }
        },
        "temperature_comfort": {
            "type": "object",
            "required": ["optimal_temp", "comfort_range", "winter_offset", "summer_offset", "impact_weight"],
            "properties": {
                "optimal_temp": {"type": "number"},
                "comfort_range": {"type": "number", "minimum": 0},
                "winter_offset": {"type": "number"},
                "summer_offset": {"type": "number"},
                "impact_weight": {"type": "number", "minimum": 0, "maximum": 1}
            }
        },
        "zones": {
            "type": "object",
            "required": ["business_district", "tourist_area", "residential", 
                        "shopping_district", "beach_area"],
            "patternProperties": {
                "^[a-z_]+$": {
                    "type": "object",
                    "required": ["center", "radius", "traffic_sensitivity", "description"],
                    "properties": {
                        "center": {
                            "type": "object",
                            "required": ["lat", "lon"],
                            "properties": {
                                "lat": {"type": "number", "minimum": 41, "maximum": 42},
                                "lon": {"type": "number", "minimum": 2, "maximum": 3}
                            }
                        },
                        "radius": {"type": "number", "minimum": 0.001, "maximum": 0.1},
                        "traffic_sensitivity": {"type": "number", "minimum": 0, "maximum": 2},
                        "description": {"type": "string", "minLength": 1}
                    }
                }
            }
        },
        "parking_patterns": {
            "type": "object",
            "required": ["Public", "Resident", "Mixed"],
            "patternProperties": {
                "^[A-Za-z]+$": {
                    "type": "object",
                    "required": ["base_level", "daily_amplitude", "peak_hours", "weekend_factor"],
                    "properties": {
                        "base_level": {"type": "number", "minimum": 0, "maximum": 100},
                        "daily_amplitude": {"type": "number", "minimum": 0, "maximum": 50},
                        "peak_hours": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["start", "end", "amplitude"],
                                "properties": {
                                    "start": {"type": "integer", "minimum": 0, "maximum": 23},
                                    "end": {"type": "integer", "minimum": 0, "maximum": 23},
                                    "amplitude": {"type": "number", "minimum": -50, "maximum": 50}
                                }
                            },
                            "minItems": 1
                        },
                        "weekend_factor": {"type": "number", "minimum": 0, "maximum": 2}
                    }
                }
            }
        },
        "parking_duration": {
            "type": "object",
            "required": ["Public", "Resident", "Mixed"],
            "patternProperties": {
                "^[A-Za-z]+$": {
                    "type": "object",
                    "required": ["short_term", "medium_term", "long_term"],
                    "properties": {
                        "short_term": {
                            "type": "object",
                            "required": ["mean_hours", "weight"],
                            "properties": {
                                "mean_hours": {"type": "number", "minimum": 0, "maximum": 24},
                                "weight": {"type": "number", "minimum": 0, "maximum": 1}
                            }
                        },
                        "medium_term": {
                            "$ref": "#/properties/parking_duration/patternProperties/^[A-Za-z]+$/properties/short_term"
                        },
                        "long_term": {
                            "$ref": "#/properties/parking_duration/patternProperties/^[A-Za-z]+$/properties/short_term"
                        }
                    },
                    "additionalProperties": False
                }
            }
        },
        "outlier_types": {
            "type": "object",
            "patternProperties": {
                "^[a-zA-Z_]+$": {
                    "type": "object",
                    "required": ["probability", "effect"],
                    "properties": {
                        "probability": {"type": "number", "minimum": 0, "maximum": 1},
                        "effect": {"type": "string", "enum": ["zero", "overflow", "random"]}
                    }
                }
            }
        },
        "api_config": {
            "type": "object",
            "required": ["barcelona_open_data", "google_maps", "openweathermap", "eventbrite"],
            "properties": {
                "barcelona_open_data": {
                    "type": "object",
                    "required": ["parking_url", "parking_resource_id", "rate_limit"],
                    "properties": {
                        "parking_url": {"type": "string", "format": "uri"},
                        "parking_resource_id": {"type": "string", "pattern": "^[a-zA-Z0-9-]+$"},
                        "rate_limit": {"type": "integer", "minimum": 1}
                    }
                },
                "google_maps": {
                    "type": "object",
                    "required": ["api_key", "rate_limit"],
                    "properties": {
                        "api_key": {"type": "string"},
                        "rate_limit": {"type": "integer", "minimum": 1}
                    }
                },
                "openweathermap": {
                    "$ref": "#/properties/api_config/properties/google_maps"
                },
                "eventbrite": {
                    "$ref": "#/properties/api_config/properties/google_maps"
                }
            }
        }
    },
    "additionalProperties": False
}

def benchmark_function(func: Callable, *args, **kwargs) -> tuple[Any, float]:
    """Benchmark a function's execution time."""
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    return result, execution_time

class ParkingDataGenerator:
    def __init__(self, config_path: Path):
        """Initialize the generator with validated configuration."""
        self.perf_monitor = PerformanceMonitor()
        self.config = self._load_and_validate_config(config_path)
        self.data_params = self.config['data_parameters']
        self.feature_ranges = self.config['feature_ranges']
        self.es_holidays = holidays.ES(prov='CT', years=[2024])
        
        # Run tests in debug mode
        if logger.getEffectiveLevel() == logging.DEBUG:
            self.test_vectorized_operations()

    def _load_and_validate_config(self, config_path: Path) -> Dict:
        """Load and validate configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            validate(instance=config, schema=CONFIG_SCHEMA)
            return config
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            raise
        except Exception as e:
            logger.error(f"Configuration error: {e}")
            raise

    def _generate_base_timestamps(self) -> List[datetime]:
        """Generate evenly distributed timestamps."""
        start = datetime.strptime(self.data_params['time_range']['start'], '%Y-%m-%d')
        end = datetime.strptime(self.data_params['time_range']['end'], '%Y-%m-%d')
        
        # Generate timestamps with higher density during business hours
        timestamps = []
        current = start
        while current <= end:
            hour = np.random.choice(24, p=self._get_hour_weights())
            current = current.replace(hour=hour, minute=np.random.randint(0, 60))
            timestamps.append(current)
            current += timedelta(minutes=np.random.randint(15, 120))
        
        return sorted(timestamps[:self.data_params['num_samples']])

    def _get_hour_weights(self) -> np.ndarray:
        """Generate probability weights for each hour of the day."""
        weights = np.ones(24) * 0.5  # Base probability
        weights[BUSINESS_HOURS[0]:BUSINESS_HOURS[1]] = 1.0  # Business hours
        for start, end in PEAK_HOURS:
            weights[start:end] = 2.0  # Peak hours
        return weights / weights.sum()

    def _generate_correlated_weather(self, timestamps: List[datetime]) -> Tuple[np.ndarray, ...]:
        """Generate weather data with realistic seasonal patterns and daily cycles."""
        num_samples = len(timestamps)
        
        # Initialize arrays
        temp = np.zeros(num_samples)
        humidity = np.zeros(num_samples)
        wind_speed = np.zeros(num_samples)
        precipitation = np.zeros(num_samples)
        
        # Generate base patterns
        for i, ts in enumerate(timestamps):
            month = ts.month
            hour = ts.hour + ts.minute/60
            
            # Get monthly averages
            base_temp, temp_std, rain_days, avg_humidity, daylight = BARCELONA_WEATHER[month]
            
            # Temperature variation throughout the day
            hour_factor = np.sin(np.pi * (hour - 4) / 24)  # Peak at 4pm
            daily_temp = base_temp + 5 * hour_factor  # Daily temperature cycle
            
            # Add seasonal and random variation
            temp[i] = daily_temp + np.random.normal(0, temp_std)
            
            # Humidity is inversely related to temperature
            humidity[i] = avg_humidity - 10 * hour_factor + np.random.normal(0, 5)
            
            # Wind tends to be stronger during daytime
            day_wind_factor = 1 + 0.5 * np.sin(np.pi * (hour - 6) / 12)
            wind_speed[i] = np.random.weibull(2) * 5 * day_wind_factor
            
            # Precipitation is more likely in certain months and early morning
            rain_probability = rain_days / 30  # Base probability from monthly average
            if hour < 6:  # Early morning has higher chance
                rain_probability *= 1.5
            
            if np.random.random() < rain_probability:
                # Use gamma distribution for precipitation amounts
                precipitation[i] = np.random.gamma(shape=1.5, scale=2.0)
        
        # Add temporal correlation (weather tends to persist)
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        temp = np.convolve(temp, kernel, mode='same')
        humidity = np.convolve(humidity, kernel, mode='same')
        wind_speed = np.convolve(wind_speed, kernel, mode='same')
        
        # Ensure precipitation events are clustered
        precipitation = self._cluster_precipitation(precipitation)
        
        # Clip values to realistic ranges
        temp = np.clip(temp, self.feature_ranges['temperature'][0],
                      self.feature_ranges['temperature'][1])
        humidity = np.clip(humidity, self.feature_ranges['humidity'][0],
                          self.feature_ranges['humidity'][1])
        wind_speed = np.clip(wind_speed, self.feature_ranges['wind_speed'][0],
                            self.feature_ranges['wind_speed'][1])
        precipitation = np.clip(precipitation, self.feature_ranges['precipitation'][0],
                              self.feature_ranges['precipitation'][1])
        
        return temp, humidity, wind_speed, precipitation

    def _cluster_precipitation(self, precipitation: np.ndarray) -> np.ndarray:
        """Create realistic clusters of precipitation events."""
        num_samples = len(precipitation)
        clustered = np.zeros_like(precipitation)
        
        # Find precipitation events
        rain_events = precipitation > 0
        
        # Create clusters
        cluster_length = np.random.randint(3, 8)  # Typical rain duration
        i = 0
        while i < num_samples:
            if rain_events[i]:
                # Start of a rain event
                end = min(i + cluster_length, num_samples)
                # Gradually increase then decrease intensity
                pattern = np.sin(np.pi * np.arange(end-i) / (end-i))
                clustered[i:end] = precipitation[i] * pattern
                i = end
            else:
                i += 1
        
        return clustered

    def _generate_time_patterns(self, timestamps: List[datetime], parking_type: str) -> np.ndarray:
        """Generate time-based patterns using multiple sine waves based on parking type."""
        hours = np.array([ts.hour + ts.minute/60 for ts in timestamps])
        days = np.array([ts.weekday() for ts in timestamps])
        
        # Get pattern parameters for this parking type
        pattern = PARKING_PATTERNS.get(parking_type, PARKING_PATTERNS['Mixed'])
        
        # Base daily pattern
        base_level = pattern['base_level']
        daily_amplitude = pattern['daily_amplitude']
        
        # Daily cycle with type-specific parameters
        daily = base_level + daily_amplitude * np.sin(2 * np.pi * (hours - 8) / 24)
        
        # Peak hour patterns
        peak_pattern = np.zeros_like(hours)
        for start, end, amplitude in pattern['peak_hours']:
            # Gaussian peaks or dips
            peak_center = (start + end) / 2
            peak_width = (end - start) / 4  # Controls width of peak
            peak_pattern += amplitude * np.exp(-((hours - peak_center) ** 2) / (2 * peak_width ** 2))
        
        # Weekly pattern varies by type
        weekend_mask = np.array([d >= 5 for d in days])
        weekly_factor = np.ones_like(days)
        weekly_factor[weekend_mask] = pattern['weekend_factor']
        
        # Special handling for different types
        if (parking_type == 'Resident'):
            # Residents have reverse commute pattern
            workday_mask = ~weekend_mask
            morning_hours = (hours >= 7) & (hours <= 9)
            evening_hours = (hours >= 17) & (hours <= 19)
            
            # More cars during night, fewer during work hours
            daily[workday_mask & morning_hours] *= 0.7  # Cars leaving
            daily[workday_mask & evening_hours] *= 1.3  # Cars returning
            
        elif (parking_type == 'Public'):
            # Public parking has stronger business hour correlation
            business_hours = (hours >= 8) & (hours <= 20)
            daily[~business_hours] *= 0.6  # Lower occupancy outside business hours
            # Lunch time peak
            lunch_peak = 15 * np.exp(-((hours - 14) ** 2) / 8)  # Added missing parenthesis and width parameter
            peak_pattern += lunch_peak
        
        # Combine all patterns
        base_pattern = daily + peak_pattern
        
        # Apply weekly factor
        base_pattern *= weekly_factor
        
        # Add monthly pattern (e.g., higher usage in summer for tourist areas)
        months = np.array([ts.month for ts in timestamps])
        if (parking_type == 'Public'):
            # Higher occupancy in tourist season (June-September)
            summer_factor = 1 + 0.2 * np.sin(2 * np.pi * (months - 6) / 12)
            base_pattern *= summer_factor
        
        return base_pattern

    def _simulate_traffic_pattern(self, timestamps: List[datetime]) -> np.ndarray:
        """Simulate traffic patterns based on time and known events."""
        num_samples = len(timestamps)
        traffic_levels = np.full(num_samples, TRAFFIC_BASE_LEVEL)
        
        for i, ts in enumerate(timestamps):
            hour = ts.hour + ts.minute/60
            date_str = ts.strftime('%Y-%m-%d')
            
            # Base traffic pattern
            if (ts.weekday() < 5):  # Weekdays
                # Morning peak
                if (TRAFFIC_PEAK_HOURS['morning'][0] <= hour <= TRAFFIC_PEAK_HOURS['morning'][1]):
                    peak_progress = (hour - TRAFFIC_PEAK_HOURS['morning'][0]) / 3
                    traffic_levels[i] += TRAFFIC_PEAK_FACTOR * np.sin(np.pi * peak_progress)
                
                # Evening peak
                elif (TRAFFIC_PEAK_HOURS['evening'][0] <= hour <= TRAFFIC_PEAK_HOURS['evening'][1]):
                    peak_progress = (hour - TRAFFIC_PEAK_HOURS['evening'][0]) / 3
                    traffic_levels[i] += TRAFFIC_PEAK_FACTOR * np.sin(np.pi * peak_progress)
            
            else:  # Weekends
                # Shopping hours
                if (11 <= hour <= 20):
                    traffic_levels[i] += 0.3 * np.sin(np.pi * (hour - 11) / 9)
            
            # Special events impact
            if (date_str in SPECIAL_EVENTS):
                event_name, impact = SPECIAL_EVENTS[date_str]
                # Event impact varies throughout the day
                event_factor = impact * np.sin(np.pi * hour / 24)
                traffic_levels[i] += event_factor
                
                logger.debug(f"Special event '{event_name}' affecting traffic on {date_str}")
        
        # Add some random variation
        noise = np.random.normal(0, 0.1, num_samples)
        traffic_levels += noise
        
        return np.clip(traffic_levels, 0, 1)

    def _calculate_temp_comfort(self, temp: np.ndarray, timestamps: List[datetime]) -> np.ndarray:
        """Calculate temperature comfort factor using seasonal-adjusted Gaussian."""
        months = np.array([ts.month for ts in timestamps])
        
        # Adjust optimal temperature based on season
        season_offset = np.zeros_like(temp)
        # Winter adjustment (Dec-Feb)
        winter_mask = (months == 12) | (months <= 2)
        season_offset[winter_mask] = TEMP_COMFORT['winter_offset']
        # Summer adjustment (Jun-Aug)
        summer_mask = (months >= 6) & (months <= 8)
        season_offset[summer_mask] = TEMP_COMFORT['summer_offset']
        
        # Calculate comfort level using Gaussian function
        optimal_temp = TEMP_COMFORT['optimal_temp'] + season_offset
        comfort_level = np.exp(
            -((temp - optimal_temp) ** 2) / 
            (2 * TEMP_COMFORT['comfort_range'] ** 2)
        )
        
        # Scale the impact
        comfort_factor = 1 - (TEMP_COMFORT['impact_weight'] * (1 - comfort_level))
        
        return comfort_factor

    def _calculate_arrival_rate(self, 
                          timestamp: datetime,
                          current_occupancy: float,
                          capacity: int,
                          weather_factor: float,
                          traffic_factor: float,
                          location_factors: Dict[str, float],
                          parking_type: str) -> float:
        """
        Calculate arrival rate based on all relevant factors.
        Returns hourly arrival rate.
        """
        # Base arrival rate depends on available spaces and current occupancy
        available_spaces = capacity * (1 - current_occupancy)
        if (available_spaces <= 0):
            return 0.0
        
        # Time-based factors
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5
        
        # Get type-specific patterns
        pattern = PARKING_PATTERNS.get(parking_type, PARKING_PATTERNS['Mixed'])
        
        # Base rate varies by parking type and time of day
        if (parking_type == 'Public'):
            if (is_weekend):
                base_rate = 0.3 if 10 <= hour <= 20 else 0.1
            else:
                base_rate = 0.5 if 8 <= hour <= 19 else 0.2
        elif (parking_type == 'Resident'):
            if (is_weekend):
                base_rate = 0.2  # Steady low turnover on weekends
            else:
                # Higher turnover during commute hours
                base_rate = 0.4 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0.2
        else:  # Mixed
            base_rate = 0.3 if 8 <= hour <= 20 else 0.15

        # Adjust base rate by location factors
        zone_multiplier = 1.0
        for zone, influence in location_factors.items():
            if (zone.endswith('_influence') and influence > 0.1):
                if ('business_district' in zone and not is_weekend):
                    zone_multiplier *= (1 + 0.5 * influence)
                elif ('shopping_district' in zone and (is_weekend or 12 <= hour <= 19)):
                    zone_multiplier *= (1 + 0.3 * influence)
                elif ('tourist_area' in zone):
                    zone_multiplier *= (1 + 0.2 * influence)

        # Calculate final arrival rate
        arrival_rate = (
            base_rate 
            * available_spaces 
            * weather_factor 
            * traffic_factor 
            * zone_multiplier 
            * location_factors['type_factor']
        )

        # Add small random variation
        arrival_rate *= np.random.normal(1, 0.1)

        return max(0, arrival_rate)

    def _calculate_time_factors(self, timestamps: List[datetime]) -> Dict[str, np.ndarray]:
        """
        Vectorized calculation of time-based factors.
        Returns dict with various time-related arrays.
        """
        # Convert timestamps to arrays of components
        hours = np.array([ts.hour + ts.minute/60 for ts in timestamps])
        weekdays = np.array([ts.weekday() for ts in timestamps])
        months = np.array([ts.month for ts in timestamps])
        
        # Convert dates to numpy datetime64 array explicitly
        dates = np.array([np.datetime64(ts.date()) for ts in timestamps])
        
        # Weekend mask
        is_weekend = weekdays >= 5
        
        # Holiday masks
        is_holiday = np.array([date in self.es_holidays for date in dates])
        next_dates = dates + np.timedelta64(1, 'D')  # Now works with numpy datetime64
        is_pre_holiday = np.array([date in self.es_holidays for date in next_dates])
        
        # Business hours mask
        business_hours = (hours >= 8) & (hours <= 20)
        
        # Peak hours masks
        morning_peak = (hours >= 7) & (hours <= 9)
        evening_peak = (hours >= 17) & (hours <= 19)
        
        # Seasonal masks
        is_summer = (months >= 6) & (months <= 8)
        is_winter = (months == 12) | (months <= 2)
        
        return {
            'hours': hours,
            'weekdays': weekdays,
            'months': months,
            'dates': dates,
            'is_weekend': is_weekend,
            'is_holiday': is_holiday,
            'is_pre_holiday': is_pre_holiday,
            'business_hours': business_hours,
            'morning_peak': morning_peak,
            'evening_peak': evening_peak,
            'is_summer': is_summer,
            'is_winter': is_winter
        }

    def _calculate_occupancy(self, timestamps: List[datetime], 
                            weather: Tuple[np.ndarray, ...],
                            location_factors: Dict[str, float],
                            parking_type: str) -> np.ndarray:
        """Vectorized calculation of occupancy rates."""
        time_factors = self._calculate_time_factors(timestamps)
        
        # Base occupancy calculation
        hours = time_factors['hours']
        base_pattern = PARKING_PATTERNS[parking_type]['base_level']
        daily_cycle = np.sin(2 * np.pi * (hours - 8) / 24)
        base_occupancy = base_pattern + PARKING_PATTERNS[parking_type]['daily_amplitude'] * daily_cycle
        
        # Calculate all factors at once
        factors = np.ones(len(timestamps))
        
        # Weekend transition (vectorized)
        friday_afternoon = (time_factors['weekdays'] == 4) & (hours >= 15)
        weekend_factor = np.where(
            friday_afternoon,
            np.interp(hours[friday_afternoon], [15, 20], [1.0, WEEKEND_OCCUPANCY_FACTOR]),
            np.where(time_factors['is_weekend'], WEEKEND_OCCUPANCY_FACTOR, 1.0)
        )
        factors *= weekend_factor
        # Holiday effect (vectorized)
        holiday_factor = np.where(
            time_factors['is_holiday'],
            HOLIDAY_OCCUPANCY_FACTOR,
            np.where(
                time_factors['is_pre_holiday'] & (hours >= 15),
                np.interp(hours, [15, 23], [1.0, HOLIDAY_OCCUPANCY_FACTOR]),
                1.0
            )
        )
        factors *= holiday_factor
        
        # Zone effects (vectorized)
        zone_factor = np.ones_like(factors)
        zone_factor *= location_factors['zone_impact']  # Use the new weighted zone impact
        factors *= zone_factor
        
        # Weather impact (already vectorized)
        weather_factor = self._calculate_weather_factor_vectorized(weather, timestamps)
        
        # Combine all factors
        occupancy = base_occupancy * factors * weather_factor
        
        # Add noise (vectorized)
        noise_scale = np.where(time_factors['business_hours'], 5.0, 2.0)
        noise = np.random.normal(0, noise_scale, len(timestamps))
        
        return np.clip(occupancy + noise,
                      self.feature_ranges['occupancy_rate'][0],
                      self.feature_ranges['occupancy_rate'][1])

    def _inject_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inject realistic outliers into the dataset based on various scenarios.
        """
        num_samples = len(df)
        df = df.copy()
        
        for outlier_type, params in OUTLIER_TYPES.items():
            # Determine outlier positions
            mask = np.random.random(num_samples) < params['probability']
            
            if (mask.sum() > 0):
                if (params['effect'] == 'zero'):
                    # Equipment failure or maintenance - zero occupancy
                    df.loc[mask, 'occupancy_rate'] = 0
                    
                elif (params['effect'] == 'overflow'):
                    # Special events or emergencies - very high occupancy
                    df.loc[mask, 'occupancy_rate'] *= 1.5
                    
                elif (params['effect'] == 'random'):
                    # Sensor malfunction - random readings
                    df.loc[mask, 'occupancy_rate'] = np.random.uniform(
                        0, 
                        self.feature_ranges['occupancy_rate'][1] * 1.2,  # Allow 20% over max
                        size=mask.sum()
                    )
                    
                # Log outlier injection
                logger.debug(f"Injected {mask.sum()} {outlier_type} outliers")
        
        # Create temporal clusters of outliers (e.g., extended equipment failure)
        cluster_starts = np.random.choice(
            num_samples - 24,  # Ensure room for cluster
            size=int(num_samples * 0.001),  # 0.1% of timestamps
            replace=False
        )
        
        for start in cluster_starts:
            cluster_length = np.random.randint(4, 24)  # 4-24 hour equipment failure
            df.loc[start:start + cluster_length, 'occupancy_rate'] = 0
            logger.debug(f"Injected equipment failure cluster of length {cluster_length}")
        
        # Add correlation between outliers and conditions
        # More sensor malfunctions during extreme weather
        extreme_weather = (
            (df['temperature'] > np.percentile(df['temperature'], 95)) |
            (df['precipitation'] > np.percentile(df['precipitation'], 95))
        )
        malfunction_mask = (np.random.random(num_samples) < 0.05) & extreme_weather
        df.loc[malfunction_mask, 'occupancy_rate'] = np.random.uniform(
            0,
            self.feature_ranges['occupancy_rate'][1] * 1.3,
            size=malfunction_mask.sum()
        )
        
        # Clip final values to physical limits
        df['occupancy_rate'] = np.clip(
            df['occupancy_rate'],
            0,
            self.feature_ranges['occupancy_rate'][1] * 1.2  # Allow some overflow
        )
        
        # Mark outliers in a new column
        df['is_outlier'] = (
            (df['occupancy_rate'] == 0) |
            (df['occupancy_rate'] > self.feature_ranges['occupancy_rate'][1])
        )
        
        return df

    def _fetch_barcelona_parking_locations(self) -> pd.DataFrame:
        """Fetch real parking facility locations from Barcelona Open Data."""
        try:
            params = {
                'resource_id': PARKING_RESOURCE_ID,
                'limit': 1000  # Adjust if needed
            }
            
            response = requests.get(BARCELONA_PARKING_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if (not data.get('success')):
                raise ValueError("API request unsuccessful")
            
            records = data['result']['records']
            df = pd.DataFrame(records)
            
            # Parse coordinates
            def parse_coordinates(coord_str):
                try:
                    coords = coord_str.split(',')
                    return pd.Series({
                        'lat': float(coords[1]),
                        'lon': float(coords[0])
                    })
                except:
                    return pd.Series({'lat': None, 'lon': None})
            
            coord_df = df['Coordenades'].apply(parse_coordinates)
            df = pd.concat([df, coord_df], axis=1)
            
            # Add parking capacity and type
            df['capacity'] = pd.to_numeric(df['places'], errors='coerce').fillna(0)
            df['parking_type'] = df['tipus_estacionament']
            
            return df[['lat', 'lon', 'capacity', 'parking_type']]
            
        except Exception as e:
            logger.error(f"Error fetching parking locations: {e}")
            # Fall back to predefined locations if API fails
            return self._get_fallback_locations()

    def _get_fallback_locations(self) -> pd.DataFrame:
        """Provide fallback parking locations if API fails."""
        return pd.DataFrame({
            'lat': [41.3851, 41.3937, 41.4034, 41.3954],
            'lon': [2.1734, 2.1647, 2.1744, 2.1915],
            'capacity': [400, 300, 250, 350],
            'parking_type': ['Public', 'Public', 'Public', 'Public']
        })

    @PerformanceMonitor.monitor('simulate_parking_durations')
    def _simulate_parking_durations_vectorized(self, timestamps: List[datetime], 
                                         occupancy: np.ndarray,
                                         weather_data: Tuple[np.ndarray, ...],
                                         traffic_levels: np.ndarray,
                                         location_factors: Dict[str, float],
                                         parking_type: str,
                                         capacity: int) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized version of parking duration simulation."""
        num_samples = len(timestamps)
        time_diffs = np.array([(timestamps[i] - timestamps[i-1]).total_seconds() / 3600 
                          for i in range(1, num_samples)])
    
        # Pre-calculate weather factors
        weather_factor = self._calculate_weather_factor_vectorized(weather_data, timestamps)
    
        # Pre-calculate base arrival rates
        hours = np.array([ts.hour for ts in timestamps])
        is_weekend = np.array([ts.weekday() >= 5 for ts in timestamps])
    
        # Vectorized base rate calculation
        base_rates = np.where(
            is_weekend,
            np.where((hours >= 10) & (hours <= 20), 0.3, 0.1),  # Weekend rates
            np.where((hours >= 8) & (hours <= 19), 0.5, 0.2)    # Weekday rates
        )
    
        # Initialize arrays
        arrivals = np.zeros(num_samples)
        departures = np.zeros(num_samples)
        current_occupancy = np.zeros(num_samples)
    
        # Use sparse matrix for vehicle tracking
        from scipy.sparse import lil_matrix
        vehicle_matrix = lil_matrix((num_samples, num_samples), dtype=np.float32)
    
        # Vectorized simulation
        for i in range(1, num_samples):
            # Calculate available spaces
            available_space = max(0, capacity * (1 - current_occupancy[i-1]))
        
            # Calculate arrival rate
            arrival_rate = (base_rates[i] * available_space * weather_factor[i] * 
                       (1 + 0.4 * traffic_levels[i]) * location_factors['type_factor'])
        
            # Generate arrivals
            new_arrivals = np.random.poisson(arrival_rate * time_diffs[i-1])
            arrivals[i] = new_arrivals
        
            if (new_arrivals > 0):
                # Generate durations for new vehicles
                durations = self._generate_durations_vectorized(
                    new_arrivals, timestamps[i], parking_type
                )
            
                # Add vehicles to sparse matrix
                end_times = np.minimum(
                    i + (durations * 3600 / np.mean(time_diffs)),
                    num_samples - 1
                ).astype(int)
            
                for j, end_time in enumerate(end_times):
                    vehicle_matrix[i, i:end_time] = 1
        
            # Update current occupancy
            current_occupancy[i] = vehicle_matrix[:i+1, i].sum() / capacity
            departures[i] = vehicle_matrix[:, i-1].sum() - vehicle_matrix[:, i].sum()
    
        # Calculate turnover rate
        turnover_rate = (arrivals + departures) / capacity
    
        # Smooth results
        kernel_size = 3
        kernel = np.ones(kernel_size) / kernel_size
        turnover_rate = np.convolve(turnover_rate, kernel, mode='same')
        current_occupancy = np.convolve(current_occupancy, kernel, mode='same')
    
        return turnover_rate, current_occupancy

    def _generate_durations_vectorized(self, num_vehicles: int, 
                                 current_time: datetime,
                                 parking_type: str) -> np.ndarray:
        """Vectorized generation of parking durations."""
        duration_params = PARKING_DURATION[parking_type]
        
        # Pre-calculate weights based on time of day
        if (current_time.hour >= 18):
            weights = np.array([0.2, 0.3, 0.5])  # Evening weights
        elif (current_time.hour <= 9):
            weights = np.array([0.6, 0.3, 0.1])  # Morning weights
        else:
            weights = np.array([params['weight'] for params in duration_params.values()])
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Generate duration types for all vehicles at once
        duration_types = np.random.choice(
            list(duration_params.keys()),
            size=num_vehicles,
            p=weights
        )
        
        # Vectorized duration generation
        mean_durations = np.array([
            duration_params[dtype]['mean_hours']
            for dtype in duration_types
        ])
        
        return np.random.exponential(mean_durations)

    def generate_dataset(self, output_path: Path) -> pd.DataFrame:
        """Generate complete synthetic dataset with parking duration modeling."""
        try:
            # Fetch real parking locations
            parking_locations = self._fetch_barcelona_parking_locations()
            num_locations = len(parking_locations)
            
            # Generate base data
            timestamps = self._generate_base_timestamps()
            weather_data = self._generate_correlated_weather(timestamps)
            traffic_levels = self._simulate_traffic_pattern(timestamps)
            
            # Generate data for each parking location
            all_data = []
            for idx, location in parking_locations.iterrows():
                # Calculate location-specific factors
                location_factors = self._calculate_location_factor(
                    location['lat'], 
                    location['lon'],
                    location['capacity'],
                    location['parking_type']
                )
                
                # Generate occupancy for this location
                occupancy = self._calculate_occupancy(
                    timestamps, 
                    weather_data,
                    location_factors=location_factors,
                    parking_type=location['parking_type']
                )
                
                # Simulate parking durations and turnover
                turnover_rate, adjusted_occupancy = self._simulate_parking_durations_vectorized(
                    timestamps,
                    occupancy,
                    weather_data,
                    traffic_levels,
                    location_factors,
                    location['parking_type'],
                    location['capacity']
                )
                
                # Create location-specific DataFrame
                location_df = pd.DataFrame({
                    'timestamp': timestamps,
                    'latitude': location['lat'],
                    'longitude': location['lon'],
                    'parking_capacity': location['capacity'],
                    'parking_type': location['parking_type'],
                    'temperature': weather_data[0],
                    'humidity': weather_data[1],
                    'wind_speed': weather_data[2],
                    'precipitation': weather_data[3],
                    'traffic_level': traffic_levels,
                    'occupancy_rate': adjusted_occupancy,
                    'turnover_rate': turnover_rate
                })
                
                all_data.append(location_df)
            
            # Combine all location data
            df = pd.concat(all_data, ignore_index=True)
            
            # Add event information and inject outliers
            df['special_event'] = df['timestamp'].dt.strftime('%Y-%m-%d').map(
                {date: event[0] for date, event in SPECIAL_EVENTS.items()}
            )
            df = self._inject_outliers(df)
            
            # Save dataset
            df.to_csv(output_path, index=False)
            
            # Log statistics
            self._log_dataset_statistics(df, num_locations)
            
            return df

        except Exception as e:
            logger.error(f"Error generating dataset: {e}")
            raise

    def _calculate_location_factor(self, lat: float, lon: float, 
                                 capacity: int, parking_type: str) -> Dict[str, float]:
        """Calculate location-specific factors including zone-based traffic sensitivity."""
        # Base capacity factor
        capacity_factor = 1.0 - (0.1 * (capacity / 1000))
        
        # Type-based factor
        type_factors = {
            'Public': 1.0,
            'Resident': 0.9,
            'Mixed': 0.95
        }
        type_factor = type_factors.get(parking_type, 1.0)
        
        # Calculate zone influences with weighted average
        zone_factors = {}
        weighted_traffic_sensitivity = 0.0
        total_weight = 0.0
        
        for zone, (zone_lat, zone_lon, radius, sensitivity) in BARCELONA_ZONES.items():
            # Calculate distance to zone center
            distance = np.sqrt((lat - zone_lat)**2 + (lon - zone_lon)**2)
            
            # Calculate influence based on distance (exponential decay)
            influence = np.exp(-distance / radius)
            
            # Weight based on zone's traffic sensitivity and influence
            weight = sensitivity * influence
            total_weight += weight
            
            # Accumulate weighted traffic sensitivity
            weighted_traffic_sensitivity += sensitivity * weight
            
            # Store individual zone influence
            zone_factors[f"{zone}_influence"] = influence
        
        # Normalize traffic sensitivity if there's any influence
        if total_weight > 0:
            # Normalize weighted traffic sensitivity to [0.5, 1.5] range
            normalized_sensitivity = 0.5 + (weighted_traffic_sensitivity / total_weight) / max(
                zone[3] for zone in BARCELONA_ZONES.values()
            )
            traffic_sensitivity = np.clip(normalized_sensitivity, 0.5, 1.5)
        else:
            traffic_sensitivity = 1.0  # Default sensitivity
        
        # Calculate weighted zone impact
        zone_impact = 0.0
        if total_weight > 0:
            for zone, (_, _, _, sensitivity) in BARCELONA_ZONES.items():
                influence = zone_factors[f"{zone}_influence"]
                weight = sensitivity * influence
                zone_impact += (weight / total_weight) * influence
            
            # Normalize zone impact to [0.5, 1.5] range
            zone_impact = 0.5 + zone_impact
            zone_impact = np.clip(zone_impact, 0.5, 1.5)
        else:
            zone_impact = 1.0
        
        return {
            'capacity_factor': capacity_factor,
            'type_factor': type_factor,
            'traffic_sensitivity': traffic_sensitivity,
            'zone_impact': zone_impact,
            **zone_factors
        }

    def _log_dataset_statistics(self, df: pd.DataFrame, num_locations: int):
        """Log detailed dataset statistics including turnover metrics."""
        logger.info("\nDataset Statistics:")
        logger.info(f"Number of parking locations: {num_locations}")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        logger.info("\nParking Types Distribution:")
        type_counts = df['parking_type'].value_counts()
        for ptype, count in type_counts.items():
            logger.info(f"{ptype}: {count} samples")
        
        logger.info("\nOccupancy Statistics by Parking Type:")
        for ptype in df['parking_type'].unique():
            mask = df['parking_type'] == ptype
            avg_occ = df.loc[mask, 'occupancy_rate'].mean()
            logger.info(f"{ptype} average occupancy: {avg_occ:.2f}%")
        
        outlier_count = df['is_outlier'].sum()
        logger.info(f"\nOutlier Statistics:")
        logger.info(f"Total outliers: {outlier_count} ({outlier_count/len(df)*100:.2f}%)")
        
        logger.info("\nTurnover Statistics by Parking Type:")
        for ptype in df['parking_type'].unique():
            mask = df['parking_type'] == ptype
            avg_turnover = df.loc[mask, 'turnover_rate'].mean()
            peak_turnover = df.loc[mask, 'turnover_rate'].max()
            logger.info(f"{ptype}:")
            logger.info(f"  Average turnover rate: {avg_turnover:.3f} vehicles/hour")
            logger.info(f"  Peak turnover rate: {peak_turnover:.3f} vehicles/hour")

    def test_vectorized_operations(self, num_samples: int = 1000) -> None:
        """
        Test and benchmark vectorized operations against non-vectorized versions.
        """
        logger.info("\nRunning vectorization tests...")
        
        # Generate test data
        timestamps = self._generate_base_timestamps()[:num_samples]
        weather_data = self._generate_correlated_weather(timestamps)
        
        # Test time factors calculation
        logger.info("\nTesting time factors calculation:")
        time_factors, vec_time = benchmark_function(self._calculate_time_factors, timestamps)
        
        # Verify time factors
        assert all(key in time_factors for key in [
            'hours', 'weekdays', 'months', 'is_weekend', 'business_hours'
        ]), "Missing expected time factors"
        
        assert len(time_factors['hours']) == num_samples, "Incorrect array length"
        assert np.all((time_factors['hours'] >= 0) & (time_factors['hours'] < 24)), "Invalid hours"
        assert np.all((time_factors['weekdays'] >= 0) & (time_factors['weekdays'] < 7)), "Invalid weekdays"
        
        logger.info(f"Time factors calculation: {vec_time:.4f} seconds")
        
        # Test occupancy calculation
        logger.info("\nTesting occupancy calculation:")
        
        test_location = {
            'lat': 41.3874,
            'lon': 2.1686,
            'capacity': 300,
            'parking_type': 'Public'
        }
        
        location_factors = self._calculate_location_factor(
            test_location['lat'],
            test_location['lon'],
            test_location['capacity'],
            test_location['parking_type']
        )
        
        occupancy, vec_time = benchmark_function(
            self._calculate_occupancy,
            timestamps,
            weather_data,
            location_factors,
            test_location['parking_type']
        )
        
        # Verify occupancy values
        assert len(occupancy) == num_samples, "Incorrect occupancy array length"
        assert np.all((occupancy >= 0) & (occupancy <= 100)), "Invalid occupancy values"
        
        # Check for expected patterns
        time_factors = self._calculate_time_factors(timestamps)
        business_hours_mask = time_factors['business_hours']
        weekend_mask = time_factors['is_weekend']
        
        # Business hours should generally have higher occupancy
        avg_business = np.mean(occupancy[business_hours_mask])
        avg_non_business = np.mean(occupancy[~business_hours_mask])
        assert avg_business > avg_non_business, "Business hours occupancy not higher than non-business hours"
        
        # Weekends should have different patterns for public parking
        if (test_location['parking_type'] == 'Public'):
            avg_weekend = np.mean(occupancy[weekend_mask])
            avg_weekday = np.mean(occupancy[~weekend_mask])
            assert avg_weekend != avg_weekday, "No difference between weekend and weekday occupancy"
        
        logger.info(f"Occupancy calculation: {vec_time:.4f} seconds")
        
        # Test weather impact
        logger.info("\nTesting weather impact:")
        
        # Create extreme weather conditions
        extreme_temp = np.full_like(weather_data[0], 35)  # Hot temperature
        extreme_weather = (extreme_temp, weather_data[1], weather_data[2], weather_data[3])
        
        extreme_occupancy, _ = benchmark_function(
            self._calculate_occupancy,
            timestamps,
            extreme_weather,
            location_factors,
            test_location['parking_type']
        )
        
        # Verify weather impact
        assert not np.array_equal(occupancy, extreme_occupancy), "Weather has no impact on occupancy"
        
        logger.info("All vectorization tests passed successfully!")
        
        return {
            'time_factors': time_factors,
            'occupancy': occupancy,
            'extreme_occupancy': extreme_occupancy
        }

    def _calibrate_weights(self, real_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calibrate model weights using real parking data.
        Uses gradient descent to minimize prediction error.
        """
        logger.info("Starting weight calibration...")
        
        # Initial weights (can be moved to config)
        weights = {
            'traffic_sensitivity': 0.4,
            'zone_influence': 0.3,
            'weather_impact': 0.2,
            'time_pattern': 0.5,
            'capacity_factor': 0.2,
            'special_event': 0.3
        }
        
        # Learning parameters
        learning_rate = 0.01
        num_epochs = 100
        batch_size = 128
        min_improvement = 0.0001  # Early stopping threshold
        
        # Prepare real data
        real_occupancy = real_data['occupancy_rate'].values
        timestamps = pd.to_datetime(real_data['timestamp'])
        locations = real_data[['latitude', 'longitude', 'parking_type', 'parking_capacity']].drop_duplicates()
        
        best_weights = weights.copy()
        best_error = float('inf')
        
        try:
            for epoch in range(num_epochs):
                total_error = 0
                weight_gradients = {k: 0.0 for k in weights}
                
                # Process in batches
                for i in range(0, len(real_data), batch_size):
                    batch_data = real_data.iloc[i:i+batch_size]
                    batch_occupancy = real_occupancy[i:i+batch_size]
                    
                    # Generate predictions with current weights
                    predicted_occupancy = self._predict_with_weights(
                        batch_data,
                        weights
                    )
                    
                    # Calculate error
                    error = np.mean((predicted_occupancy - batch_occupancy) ** 2)
                    total_error += error
                    
                    # Calculate gradients for each weight
                    for weight_name in weights:
                        # Compute partial derivative
                        weight_delta = weights[weight_name] * 0.01  # Small delta for numerical gradient
                        test_weights = weights.copy()
                        test_weights[weight_name] += weight_delta
                        
                        predicted_with_delta = self._predict_with_weights(
                            batch_data,
                            test_weights
                        )
                        
                        # Numerical gradient
                        gradient = np.mean(
                            (predicted_with_delta - predicted_occupancy) * 
                            (predicted_occupancy - batch_occupancy)
                        ) / weight_delta
                        
                        weight_gradients[weight_name] += gradient
                
                # Update weights using gradients
                for weight_name in weights:
                    weights[weight_name] -= learning_rate * weight_gradients[weight_name]
                    # Ensure weights stay in valid range
                    weights[weight_name] = np.clip(weights[weight_name], 0.0, 1.0)
                
                # Normalize weights to sum to 1
                weight_sum = sum(weights.values())
                weights = {k: v/weight_sum for k, v in weights.items()}
                
                # Check for improvement
                if (total_error < best_error - min_improvement):
                    best_error = total_error
                    best_weights = weights.copy()
                    logger.info(f"Epoch {epoch}: New best error: {best_error:.4f}")
                    logger.info("Current weights:")
                    for k, v in weights.items():
                        logger.info(f"  {k}: {v:.3f}")
                
                # Early stopping
                if (epoch > 10 and total_error >= best_error - min_improvement):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
        except Exception as e:
            logger.error(f"Error during weight calibration: {e}")
            return best_weights
        
        logger.info("Weight calibration completed")
        logger.info("Final weights:")
        for k, v in best_weights.items():
            logger.info(f"  {k}: {v:.3f}")
        
        return best_weights

    def _predict_with_weights(self, data: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
        """
        Generate occupancy predictions using given weights.
        """
        # Get base components
        time_pattern = self._generate_time_patterns(
            pd.to_datetime(data['timestamp']).tolist(),
            data['parking_type'].iloc[0]
        )
        
        traffic_levels = self._simulate_traffic_pattern(
            pd.to_datetime(data['timestamp']).tolist()
        )
        
        # Calculate location factors
        location_factors = self._calculate_location_factor(
            data['latitude'].iloc[0],
            data['longitude'].iloc[0],
            data['parking_capacity'].iloc[0],
            data['parking_type'].iloc[0]
        )
        
        # Weather impact
        weather_data = (
            data['temperature'].values,
            data['humidity'].values,
            data['wind_speed'].values,
            data['precipitation'].values
        )
        weather_factor = (
            self._calculate_temp_comfort(weather_data[0], pd.to_datetime(data['timestamp']).tolist()) *
            np.where(weather_data[3] > 0, 1 - (0.3 * np.tanh(weather_data[3] / 10)), 1.0) *
            np.clip(1 - (weather_data[2] / 50), 0.7, 1.0)
        )
        
        # Special events impact
        special_event_factor = np.ones(len(data))
        event_dates = data['timestamp'].dt.strftime('%Y-%m-%d')
        for date, (_, impact) in SPECIAL_EVENTS.items():
            special_event_factor[event_dates == date] *= (1 + impact)
        
        # Combine all factors with weights
        prediction = (
            weights['time_pattern'] * time_pattern +
            weights['traffic_sensitivity'] * traffic_levels * location_factors['traffic_sensitivity'] +
            weights['zone_influence'] * sum(
                influence for name, influence in location_factors.items()
                if name.endswith('_influence')
            ) +
            weights['weather_impact'] * weather_factor +
            weights['capacity_factor'] * location_factors['capacity_factor'] +
            weights['special_event'] * (special_event_factor - 1)
        )
        
        return np.clip(prediction, 0, 100)

    def calibrate_and_save_weights(self, real_data_path: Path, weights_output_path: Path) -> None:
        """
        Calibrate weights using real data and save to file.
        """
        try:
            # Load real data
            real_data = pd.read_csv(real_data_path)
            required_columns = [
                'timestamp', 'latitude', 'longitude', 'parking_type',
                'parking_capacity', 'occupancy_rate', 'temperature',
                'humidity', 'wind_speed', 'precipitation'
            ]
            
            missing_cols = [col for col in required_columns if col not in real_data.columns]
            if (missing_cols):
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert timestamp to datetime
            real_data['timestamp'] = pd.to_datetime(real_data['timestamp'])
            
            # Calibrate weights
            calibrated_weights = self._calibrate_weights(real_data)
            
            # Save weights
            with open(weights_output_path, 'w') as f:
                json.dump(calibrated_weights, f, indent=4)
            
            logger.info(f"Calibrated weights saved to {weights_output_path}")
            
        except Exception as e:
            logger.error(f"Error in weight calibration: {e}")
            raise

    def _calculate_weather_factor_vectorized(self, weather_data: Tuple[np.ndarray, ...],
                                       timestamps: List[datetime]) -> np.ndarray:
        """Vectorized calculation of weather impact factors."""
        temp, humidity, wind_speed, precipitation = weather_data
        
        # Pre-calculate masks for efficiency
        months = np.array([ts.month for ts in timestamps])
        winter_mask = (months == 12) | (months <= 2)
        summer_mask = (months >= 6) & (months <= 8)
        
        # Vectorized temperature comfort calculation
        season_offset = np.zeros_like(temp)
        season_offset[winter_mask] = TEMP_COMFORT['winter_offset']
        season_offset[summer_mask] = TEMP_COMFORT['summer_offset']
        
        optimal_temp = TEMP_COMFORT['optimal_temp'] + season_offset
        comfort_level = np.exp(
            -((temp - optimal_temp) ** 2) / 
            (2 * TEMP_COMFORT['comfort_range'] ** 2)
        )
        
        # Combine all weather factors
        return (
            (1 - (TEMP_COMFORT['impact_weight'] * (1 - comfort_level))) *
            np.where(precipitation > 0, 1 - (0.3 * np.tanh(precipitation / 10)), 1.0) *
            np.clip(1 - (wind_speed / 50), 0.7, 1.0) *
            (1 - 0.2 * np.clip((humidity - 70) / 30, 0, 1))
        )

def main():
    """Main execution function."""
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--test', action='store_true', help='Run vectorization tests')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--calibrate', action='store_true', help='Calibrate weights with real data')
        parser.add_argument('--real-data', type=str, help='Path to real parking data CSV')
        args = parser.parse_args()
        
        if (args.debug):
            logger.setLevel(logging.DEBUG)
        
        config_path = Path('config/generator_config.json')
        output_path = Path('data/synthetic_parking_data.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        generator = ParkingDataGenerator(config_path)
        
        if (args.calibrate):
            if (not args.real_data):
                raise ValueError("--real-data path required for calibration")
            weights_path = Path('config/calibrated_weights.json')
            generator.calibrate_and_save_weights(
                Path(args.real_data),
                weights_path
            )
        elif (args.test):
            generator.test_vectorized_operations(num_samples=1000)
        else:
            df = generator.generate_dataset(output_path)
            
            logger.info("\nDataset Statistics:")
            logger.info(f"Number of samples: {len(df)}")
            logger.info("\nFeature Ranges:")
            for column in df.select_dtypes(include=[np.number]).columns:
                logger.info(f"{column}: {df[column].min():.2f} to {df[column].max():.2f}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
