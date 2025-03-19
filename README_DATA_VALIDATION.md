# Data Validation System for OnSpot Predictive Model

This document describes the data validation system implemented for the OnSpot Predictive Model project. The validation system ensures data quality and integrity before uploading to Supabase.

## Overview

The data validation system includes:

1. A set of data validators for different data types
2. A standalone validation tool to check data quality
3. An automatic issue fixing tool to correct common data problems
4. Integration with the data upload process

## Data Types

The validation system supports the following data types:

- **Feature Engineered Data** - Processed data with engineered features for the predictive model
- **Predictions Data** - Model predictions for parking occupancy
- **Raw Parking Data** - Original parking data from sensors

## Validation Criteria

Each data type has specific validation criteria:

### Feature Engineered Data

- `occupancy_rate`: Must be between 0 and 1
- `location_id`: Must follow valid format
- `timestamp`: Must be a valid date/time
- `is_holiday`: Must be a boolean or 0/1
- `day_of_week`: Must be between 0 and 6
- `hour_of_day`: Must be between 0 and 23

### Predictions Data

- `predicted_occupancy`: Must be between 0 and 1
- `location_id`: Must follow valid format
- `prediction_timestamp`: Must be a valid date/time
- `predicted_for_timestamp`: Must be a valid date/time
- `model_version`: Must not be empty

### Raw Parking Data

- `occupancy_rate`: Must be between 0 and 1
- `location_id`: Must follow valid format
- `timestamp`: Must be a valid date/time
- `total_spots`: Must be a positive number
- `available_spots`: Must be non-negative and ≤ total_spots
- Consistency check: `occupancy_rate` should match `(total_spots - available_spots) / total_spots`

## Usage

### Data Validation

To validate data files:

```bash
python scripts/validate_data.py [data_type]
```

Where `[data_type]` can be:
- `feature_engineered` - Validate feature engineered data
- `predictions` - Validate prediction data
- `raw_parking` - Validate raw parking data
- `all` - Validate all data types (default)

Options:
- `--file/-f` - Specify a custom file path
- `--output/-o` - Export validation results to JSON

Examples:

```bash
# Validate all data types
python scripts/validate_data.py

# Validate only feature engineered data
python scripts/validate_data.py feature_engineered

# Validate a specific predictions file and export results
python scripts/validate_data.py predictions --file path/to/predictions.csv --output validation_results.json
```

### Automatic Issue Fixing

The system includes a tool to automatically fix common data issues:

```bash
python scripts/fix_data_issues.py [data_type]
```

Where `[data_type]` can be:
- `feature_engineered` - Fix feature engineered data
- `predictions` - Fix prediction data
- `raw_parking` - Fix raw parking data
- `all` - Fix all data types (default)

Options:
- `--input/-i` - Specify a custom input file path
- `--output/-o` - Specify a custom output file path (defaults to overwriting input)

Examples:

```bash
# Fix all data types
python scripts/fix_data_issues.py

# Fix only predictions data
python scripts/fix_data_issues.py predictions

# Fix a specific file and save to a new location
python scripts/fix_data_issues.py feature_engineered --input original.csv --output fixed.csv
```

## Automatic Fixes Applied

The fix_data_issues.py script applies the following automatic fixes:

### Feature Engineered Data

- Clips `occupancy_rate` to [0,1] range
- Fills missing `occupancy_rate` with median value
- Converts `is_holiday` to 0/1 format
- Corrects `day_of_week` and `hour_of_day` values
- Adds `is_weekend` column if missing
- Fixes timestamp format issues

### Predictions Data

- Clips `predicted_occupancy` to [0,1] range
- Fills missing values with appropriate defaults
- Fixes timestamp format issues
- Adds `confidence` column if missing

### Raw Parking Data

- Recalculates and fixes `occupancy_rate` values
- Ensures `total_spots` values are positive
- Makes `available_spots` consistent with `total_spots`
- Fixes timestamp format issues
- Ensures consistency between spots and occupancy rate

## Integration with Upload Process

The validation system is integrated with the data upload process. Before uploading data to Supabase:

1. Data is validated against the criteria
2. Issues are reported in detail
3. If automatic fixing is enabled, common issues are fixed
4. Only valid data is uploaded to Supabase

## Error Handling

The validation system provides detailed error reports including:

- Count of valid and invalid records
- Specific reasons for invalidation
- Examples of invalid records
- Suggestions for fixing issues

## Implementation Details

The validation system is implemented in Python with the following components:

- `data_validators.py` - Contains validator classes for each data type
- `validate_data.py` - Command-line tool for validation
- `fix_data_issues.py` - Tool for automatic issue fixing
- Integration with upload scripts

## Best Practices

1. Always validate data before uploading to Supabase
2. Run the fix_data_issues.py script to automatically correct common issues
3. For complex issues, manual intervention may be required
4. Review validation results regularly to identify systematic data quality issues

## License

This validation system is part of the OnSpot Predictive Model project and is subject to the same license terms. 