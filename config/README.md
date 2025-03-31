# Configuration Management

This directory contains all configuration files for the OnSpot Predictive Model project.

## Directory Structure

```
config/
├── environments/           # Environment-specific configurations
│   ├── development.yaml   # Development environment
│   ├── staging.yaml       # Staging environment
│   ├── production.yaml    # Production environment
│   └── test.yaml         # Testing environment
│
├── schemas/              # JSON schemas for validation
│   ├── data/            # Data validation schemas
│   ├── model/           # Model configuration schemas
│   └── api/             # API request/response schemas
│
└── default.yaml         # Default configuration (base config)
```

## Configuration System

The configuration system uses a hierarchical approach:
1. Default configuration (`default.yaml`)
2. Environment-specific overrides
3. Local overrides (`.env`)
4. Environment variables

### Default Configuration

```yaml
# default.yaml
app:
  name: "OnSpot Predictive Model"
  version: "1.0.0"
  environment: "development"

data:
  storage_path: "data/"
  validation:
    min_samples: 30
    correlation_threshold: 0.3
    confidence_threshold: 0.95

models:
  storage_path: "models/"
  versioning:
    enabled: true
    storage: "production_models/"
  retraining:
    interval_days: 30
    min_samples: 1000

api:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]
  timeout: 30

monitoring:
  enabled: true
  drift_detection:
    interval_hours: 24
    threshold: 0.1
  metrics:
    export_path: "monitoring/metrics/"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/onspot.log"
```

### Environment-Specific Configuration

Example production configuration:
```yaml
# environments/production.yaml
app:
  environment: "production"

api:
  cors_origins: ["https://api.onspot.com"]
  timeout: 60

monitoring:
  drift_detection:
    interval_hours: 12
    threshold: 0.05

logging:
  level: "WARNING"
```

## Usage

### Loading Configuration

```python
from onspot.utils.config import load_config

# Load configuration for current environment
config = load_config()

# Access configuration values
api_port = config["api"]["port"]
model_path = config["models"]["storage_path"]
```

### Environment Variables

Configuration can be overridden using environment variables:

```bash
# Override configuration values
export ONSPOT_API_PORT=9000
export ONSPOT_MONITORING_ENABLED=false
```

### Local Development

1. Copy `.env.example` to `.env`
2. Modify values for local development
3. Values in `.env` take precedence over YAML configs

## Validation Schemas

### Data Schemas
- Input data validation
- Feature validation
- Output validation

### Model Schemas
- Model configuration validation
- Hyperparameter validation
- Training configuration validation

### API Schemas
- Request validation
- Response validation
- Error response validation

## Best Practices

1. Never commit sensitive values
2. Use environment variables for secrets
3. Keep configurations modular
4. Document all configuration options
5. Validate configurations at startup

## Adding New Configuration

1. Add to `default.yaml`
2. Update relevant environment configs
3. Add validation schema if needed
4. Update documentation
5. Update tests 