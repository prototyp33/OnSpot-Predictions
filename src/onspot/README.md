# OnSpot Source Code

This directory contains the core implementation of the OnSpot Predictive Model.

## Module Organization

```
src/onspot/
├── api/                # API implementation
│   ├── main.py        # FastAPI application
│   ├── routes/        # API endpoints
│   └── schemas.py     # Data models
├── data/              # Data processing
│   ├── ingestion.py   # Data loading
│   └── validation.py  # Data validation
├── features/          # Feature engineering
│   ├── base.py       # Base transformers
│   └── parking.py    # Parking features
├── models/           # Model implementations
│   ├── base.py      # Base model interface
│   └── parking.py   # Parking models
├── pipelines/        # Training pipelines
│   ├── training.py  # Model training
│   └── prediction.py # Batch prediction
├── monitoring/       # Monitoring tools
│   ├── drift.py     # Drift detection
│   └── metrics.py   # Performance metrics
└── utils/           # Utilities
    ├── config.py    # Configuration
    └── logging.py   # Logging setup
```

## Module Descriptions

### API (`api/`)
- FastAPI implementation
- RESTful endpoints
- Request/response validation
- API documentation

### Data Processing (`data/`)
- Data ingestion from various sources
- Data validation and quality checks
- Data transformation pipelines
- Schema management

### Feature Engineering (`features/`)
- Feature transformation pipeline
- Time-based features
- Location-based features
- Weather features

### Models (`models/`)
- Model implementations
- Model versioning
- Model evaluation
- Prediction logic

### Pipelines (`pipelines/`)
- Training pipelines
- Prediction pipelines
- Pipeline configuration
- Data flow management

### Monitoring (`monitoring/`)
- Drift detection
- Performance monitoring
- Metric collection
- Alert generation

### Utilities (`utils/`)
- Configuration management
- Logging setup
- Common utilities
- Helper functions

## Development Guidelines

1. Follow modular design principles
2. Write comprehensive tests
3. Document all functions and classes
4. Use type hints
5. Follow PEP 8 style guide

## Code Organization

- Keep modules focused and single-purpose
- Use clear, descriptive names
- Maintain consistent coding style
- Write comprehensive documentation
- Include usage examples

## Testing

Each module should have corresponding tests:
- Unit tests for core functionality
- Integration tests for workflows
- Performance tests for critical paths
- Property-based tests where appropriate

## Documentation

- Use docstrings for all public interfaces
- Include usage examples
- Document parameters and return types
- Explain complex algorithms
- Maintain API documentation 