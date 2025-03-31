# API System

This directory contains the REST API implementation for the OnSpot Predictive Model project.

## Directory Structure

```
api/
├── routes/            # API route handlers
│   ├── predictions/   # Prediction endpoints
│   ├── models/        # Model management
│   └── health/        # Health checks
│
├── middleware/        # API middleware
│   ├── auth/         # Authentication
│   ├── validation/   # Request validation
│   └── logging/      # Request logging
│
├── schemas/          # API schemas
│   ├── requests/     # Request schemas
│   └── responses/    # Response schemas
│
├── services/         # Business logic
│   ├── predictor/    # Prediction service
│   ├── monitoring/   # Monitoring service
│   └── storage/      # Storage service
│
└── tests/            # API tests
    ├── integration/  # Integration tests
    └── unit/         # Unit tests
```

## API Endpoints

### Predictions

#### Make Prediction
```http
POST /api/v1/predictions
Content-Type: application/json

{
  "location_id": "sf-downtown-01",
  "timestamp": "2024-03-20T14:30:00Z",
  "features": {
    "weather": {
      "temperature": 18.5,
      "precipitation": 0.0
    },
    "events": [
      {
        "type": "sports",
        "distance": 0.5
      }
    ]
  }
}
```

#### Get Prediction History
```http
GET /api/v1/predictions?location_id=sf-downtown-01&start_date=2024-03-01&end_date=2024-03-20
```

### Model Management

#### Get Model Info
```http
GET /api/v1/models/current
```

#### Update Model
```http
POST /api/v1/models/update
Content-Type: application/json

{
  "model_id": "v1.2.3",
  "description": "Updated model with improved accuracy"
}
```

### Health Checks

#### API Health
```http
GET /api/v1/health
```

#### Model Health
```http
GET /api/v1/health/model
```

## Authentication

### API Keys
```http
Authorization: Bearer <api_key>
```

### Rate Limiting
- 100 requests per minute per API key
- 1000 requests per hour per API key
- Custom limits available for enterprise users

## Request Validation

### Input Schema
```python
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class PredictionRequest(BaseModel):
    location_id: str
    timestamp: datetime
    features: dict
    options: Optional[dict] = None
```

### Response Schema
```python
class PredictionResponse(BaseModel):
    prediction_id: str
    timestamp: datetime
    occupancy_rate: float
    confidence: float
    model_version: str
```

## Error Handling

### Error Codes
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 429: Too Many Requests
- 500: Internal Server Error

### Error Response
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input format",
    "details": {
      "field": "timestamp",
      "reason": "must be in ISO format"
    }
  }
}
```

## Usage Examples

### Python Client
```python
from onspot.client import OnSpotClient

client = OnSpotClient(api_key="your-api-key")

# Make prediction
prediction = client.predict(
    location_id="sf-downtown-01",
    timestamp="2024-03-20T14:30:00Z",
    features={
        "weather": {"temperature": 18.5},
        "events": [{"type": "sports", "distance": 0.5}]
    }
)

# Get prediction history
history = client.get_predictions(
    location_id="sf-downtown-01",
    start_date="2024-03-01",
    end_date="2024-03-20"
)
```

### cURL Examples
```bash
# Make prediction
curl -X POST \
  https://api.onspot.ai/v1/predictions \
  -H 'Authorization: Bearer your-api-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "location_id": "sf-downtown-01",
    "timestamp": "2024-03-20T14:30:00Z",
    "features": {
      "weather": {"temperature": 18.5}
    }
  }'

# Get model info
curl -X GET \
  https://api.onspot.ai/v1/models/current \
  -H 'Authorization: Bearer your-api-key'
```

## Development

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
```

### Running Locally
```bash
# Start API server
uvicorn api.main:app --reload --port 8000

# Run tests
pytest api/tests/
```

## Deployment

### Docker
```bash
# Build image
docker build -t onspot-api .

# Run container
docker run -p 8000:8000 onspot-api
```

### Kubernetes
```bash
# Apply deployment
kubectl apply -f k8s/api-deployment.yaml

# Scale deployment
kubectl scale deployment onspot-api --replicas=3
```

## Monitoring

### Metrics
- Request count
- Response time
- Error rate
- Prediction accuracy
- Resource usage

### Logging
- Request/response logging
- Error logging
- Performance logging
- Audit logging

## Best Practices

1. API Design
   - RESTful principles
   - Version endpoints
   - Clear documentation
   - Consistent responses

2. Security
   - API key validation
   - Rate limiting
   - Input validation
   - Error handling

3. Performance
   - Response caching
   - Async processing
   - Connection pooling
   - Resource optimization

4. Testing
   - Unit tests
   - Integration tests
   - Load tests
   - Security tests 