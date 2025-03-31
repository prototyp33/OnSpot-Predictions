# Data Directory

This directory contains all data assets for the OnSpot Predictive Model project.

## Directory Structure

```
data/
├── raw/               # Original, immutable data
│   └── parking/      # Parking-related raw data
├── processed/        # Cleaned, transformed data
│   └── features/    # Feature engineered datasets
├── interim/         # Intermediate processing data
│   └── validation/ # Validation datasets
├── external/        # External reference data
└── metadata/       # Dataset documentation
```

## Data Organization

### Raw Data (`raw/`)
- Original source data
- Immutable files
- Version controlled
- Documentation of origin

### Processed Data (`processed/`)
- Cleaned datasets
- Feature engineered data
- Training ready data
- Validation splits

### Interim Data (`interim/`)
- Intermediate processing results
- Temporary datasets
- Validation data
- Test datasets

### External Data (`external/`)
- Third-party datasets
- Reference data
- Benchmark datasets
- Public datasets

### Metadata (`metadata/`)
- Data dictionaries
- Schema definitions
- Data quality reports
- Dataset documentation

## Data Management

### Version Control
- Track data versions
- Document changes
- Maintain data lineage
- Archive old versions

### Data Quality
- Validation checks
- Quality metrics
- Completeness reports
- Consistency checks

### Documentation
- Data sources
- Processing steps
- Feature definitions
- Usage guidelines

### Best Practices
1. Never modify raw data
2. Document all transformations
3. Version control datasets
4. Validate data quality
5. Track data lineage

## Data Pipeline

1. Data Collection
   - Source identification
   - Data extraction
   - Quality validation

2. Data Processing
   - Cleaning
   - Transformation
   - Feature engineering

3. Data Validation
   - Quality checks
   - Schema validation
   - Distribution analysis

4. Data Storage
   - Version management
   - Access control
   - Backup strategy

## Usage Guidelines

- Document data sources
- Track data versions
- Validate data quality
- Monitor data drift
- Archive old versions

## Data Descriptions

### Raw Data
- `Estacionaments_Area_DUM.json`: Original parking data from Barcelona's open data portal
  - Format: JSON
  - Size: 911KB
  - Last Updated: 2024-02-23

### Interim Data
- `cleaned_OSM-parking_data.csv`: Cleaned OpenStreetMap parking data
  - Format: CSV
  - Size: 1.1MB
  - Last Updated: 2024-02-10
  - Processing: Basic cleaning and standardization

### Processed Data
- `cleaned_parking_data_with_features.csv`: Main dataset with engineered features
  - Format: CSV
  - Size: 229MB
  - Last Updated: 2024-02-04
  - Features: Location, time, occupancy, and derived features

- `feature_engineered_data.csv`: Complete feature engineered dataset
  - Format: CSV
  - Size: 402MB
  - Last Updated: 2024-03-18
  - Additional Features: Weather, events, temporal patterns

- `prepared_data_*.csv`: Various versions of prepared training data
  - Standard: Basic preparation
  - Improved: Enhanced feature selection
  - Advanced: Advanced feature engineering

## Data Flow
1. Raw data is collected from sources
2. Basic cleaning in interim stage
3. Feature engineering and preparation in processed stage
4. Final datasets used for model training and evaluation

## Data Quality
- All processed data includes data quality checks
- Validation results are stored in `validation_results/`
- Monitoring of data drift in `monitoring/`

## Adding New Data
1. Place raw data in `raw/`
2. Document source and timestamp
3. Create cleaning script in `src/onspot/data/`
4. Save intermediate results in `interim/`
5. Save final results in `processed/`

# Data Management

This directory contains all data-related resources for the OnSpot Predictive Model project.

## Directory Structure

```
data/
├── raw/                  # Raw data files
│   ├── historical/      # Historical parking data
│   ├── weather/        # Weather data
│   └── events/         # Event data
│
├── processed/           # Processed data files
│   ├── features/       # Engineered features
│   ├── training/       # Training datasets
│   └── validation/     # Validation datasets
│
├── external/            # External data sources
│   ├── maps/          # Map data
│   └── demographics/   # Demographic data
│
├── interim/            # Intermediate data
│   ├── cleaned/       # Cleaned data
│   └── transformed/   # Transformed data
│
└── metadata/           # Data metadata
    ├── schemas/       # Data schemas
    └── catalogs/      # Data catalogs
```

## Data Sources

### Parking Data
- Occupancy rates
- Entry/exit timestamps
- Duration of stay
- Payment information

### Weather Data
- Temperature
- Precipitation
- Wind speed
- Cloud cover

### Event Data
- Local events
- Holidays
- Special occasions

## Data Processing Pipeline

1. Data Collection
   - API integrations
   - Batch imports
   - Real-time streams

2. Data Cleaning
   - Missing value handling
   - Outlier detection
   - Duplicate removal
   - Format standardization

3. Feature Engineering
   - Time-based features
   - Weather features
   - Event features
   - Location features

4. Dataset Creation
   - Training set
   - Validation set
   - Test set

## Data Quality

### Validation Rules
- Completeness checks
- Format validation
- Range validation
- Relationship validation

### Quality Metrics
- Missing value rate
- Outlier rate
- Duplicate rate
- Data freshness

## Data Versioning

### Version Control
- Dataset versioning
- Feature set versioning
- Schema versioning

### Metadata Tracking
- Data lineage
- Processing history
- Quality metrics
- Usage statistics

## Usage Guidelines

### Loading Data

```python
from onspot.data import DataLoader

# Load raw parking data
loader = DataLoader()
parking_data = loader.load_parking_data()

# Load processed features
features = loader.load_features()
```

### Processing Data

```python
from onspot.data import DataProcessor

# Process raw data
processor = DataProcessor()
processed_data = processor.process(raw_data)

# Create features
features = processor.create_features(processed_data)
```

## Best Practices

1. Data Organization
   - Follow directory structure
   - Use consistent naming
   - Maintain documentation

2. Data Processing
   - Log all transformations
   - Validate outputs
   - Handle errors gracefully

3. Data Quality
   - Regular validation
   - Quality monitoring
   - Issue tracking

4. Data Security
   - Access control
   - Data encryption
   - Backup strategy

## Adding New Data

1. Data Source
   - Document source
   - Define schema
   - Set up collection

2. Processing Steps
   - Add cleaning rules
   - Define transformations
   - Create features

3. Integration
   - Update pipeline
   - Add validation
   - Test integration

## Data Documentation

### Required Documentation
- Data sources
- Schema definitions
- Processing steps
- Quality metrics
- Usage examples

### Metadata Fields
- Creation date
- Last update
- Version
- Owner
- License

## Data Retention

### Policies
- Retention period
- Archival rules
- Deletion criteria

### Compliance
- Privacy regulations
- Security standards
- Industry requirements 