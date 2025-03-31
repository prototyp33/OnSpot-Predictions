# Weather Impact Analysis

The OnSpot Predictive Model includes a sophisticated weather impact analysis system that helps understand and predict how weather conditions affect parking patterns. This feature enables parking operators to make data-driven decisions about capacity management based on weather forecasts.

## Overview

The weather impact analysis system consists of several components:

1. **Weather Condition Tracking**: Monitors and stores weather conditions including temperature, precipitation, wind speed, humidity, and cloud cover.
2. **Impact Analysis**: Analyzes correlations between weather conditions and parking occupancy patterns.
3. **Capacity Adjustment**: Provides recommendations for parking capacity adjustments based on weather forecasts.
4. **Historical Analysis**: Uses historical data to improve prediction accuracy over time.

## Key Features

### Weather Severity Calculation

The system calculates a weather severity index (0-100) based on multiple factors:

- Temperature extremes
- Precipitation levels
- Wind speed
- Combined weather effects

Example severity calculation:
```python
severity = (
    0.4 * temperature_impact +
    0.3 * precipitation_impact +
    0.3 * wind_impact
)
```

### Impact Analysis Metrics

The analysis provides several key metrics:

- **Correlation Coefficient**: Measures the strength and direction of weather impact (-1 to 1)
- **Confidence Level**: Statistical confidence in the analysis (0-1)
- **Impact Score**: Normalized measure of weather impact (0-1)
- **Sample Size**: Number of data points used in the analysis

### Capacity Adjustment Recommendations

The system provides:

- Real-time capacity adjustment suggestions
- Confidence-based recommendations
- Automated alerts for severe weather conditions
- Historical trend analysis

## API Reference

### Adding Weather Conditions

```http
POST /weather/conditions/{location_id}
```

Request body:
```json
{
    "temperature_celsius": 25.0,
    "precipitation_mm": 0.0,
    "wind_speed_ms": 5.0,
    "humidity": 60.0,
    "cloud_cover": 30.0
}
```

### Analyzing Weather Impact

```http
POST /weather/impact/{location_id}
```

Request body:
```json
{
    "timestamp": "2024-03-20T10:00:00Z",
    "occupied_spots": 75,
    "total_spots": 100
}
```

Response:
```json
{
    "status": "success",
    "metrics": {
        "correlation": 0.8,
        "confidence": 0.95,
        "impact_score": 0.8
    },
    "recommendations": [
        "Strong positive correlation between weather severity and parking occupancy.",
        "Consider increasing capacity during severe weather."
    ]
}
```

### Getting Weather-Adjusted Capacity

```http
POST /weather/capacity/{location_id}
```

Response:
```json
{
    "status": "success",
    "current_occupancy": 75.0,
    "predicted_occupancy": 85.0,
    "adjustment_factor": 1.13,
    "recommended_capacity": 113
}
```

## Implementation Guide

### Setting Up Weather Monitoring

1. Configure weather data sources:
   ```python
   from onspot.weather.conditions.repository import JsonWeatherRepository
   
   repository = JsonWeatherRepository(storage_path="data/weather")
   ```

2. Initialize the impact analyzer:
   ```python
   from onspot.weather.analysis.impact_analyzer import WeatherImpactAnalyzer
   
   analyzer = WeatherImpactAnalyzer(repository)
   ```

3. Create the service:
   ```python
   from onspot.weather.analysis.impact_analyzer import WeatherImpactService
   
   service = WeatherImpactService(analyzer)
   ```

### Analyzing Weather Impact

```python
# Current occupancy data
occupancy = ParkingOccupancy(
    timestamp=datetime.now(),
    occupied_spots=75,
    total_spots=100
)

# Analyze impact
result = service.analyze_current_impact("location_1", occupancy)

# Get capacity adjustment
adjusted = service.get_weather_adjusted_capacity(
    "location_1",
    occupancy,
    weather_forecast
)
```

## Best Practices

1. **Data Collection**
   - Maintain consistent weather data collection intervals
   - Store historical data for at least 30 days
   - Validate weather data quality before analysis

2. **Analysis Configuration**
   - Set appropriate correlation thresholds (default: 0.3)
   - Configure confidence thresholds (default: 0.95)
   - Adjust minimum sample size requirements (default: 30)

3. **Capacity Management**
   - Implement gradual capacity adjustments
   - Monitor adjustment effectiveness
   - Consider seasonal patterns

4. **Integration**
   - Use the weather impact API in conjunction with other parking metrics
   - Implement automated alerts for severe weather
   - Regular system performance review

## Troubleshooting

Common issues and solutions:

1. **Insufficient Data**
   - Ensure regular weather data collection
   - Check data storage configuration
   - Verify historical data retention

2. **Low Confidence Results**
   - Increase data collection frequency
   - Review correlation thresholds
   - Check for data quality issues

3. **Inconsistent Predictions**
   - Validate weather data sources
   - Check for seasonal anomalies
   - Review impact calculation weights

## Future Enhancements

Planned improvements:

1. **Machine Learning Integration**
   - Advanced pattern recognition
   - Automated threshold adjustment
   - Seasonal pattern learning

2. **Extended Weather Metrics**
   - Air quality integration
   - Weather radar data
   - Long-term forecast integration

3. **Enhanced Reporting**
   - Interactive dashboards
   - Custom report generation
   - Real-time monitoring

## Contributing

To contribute to the weather impact analysis system:

1. Review the existing implementation in `onspot/weather/`
2. Follow the test-driven development approach
3. Add tests for new features
4. Document API changes and new functionality
5. Submit pull requests with comprehensive descriptions

## Additional Resources

- [Weather Data Sources](../integration/weather_sources.md)
- [API Documentation](../api_reference/weather.md)
- [Configuration Guide](../configuration/weather_config.md)
- [Example Implementations](../examples/weather_impact.md) 