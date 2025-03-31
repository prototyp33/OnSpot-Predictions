import pandas as pd

def validate_parking_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates a DataFrame containing parking occupancy data based on predefined rules.

    Args:
        input_df: Pandas DataFrame with parking data. Expected columns include
                  'id', 'update_timestamp', 'total_spots', 'available_spots',
                  'occupancy_rate'.

    Returns:
        A pandas DataFrame containing details of validation errors.
        Returns an empty DataFrame if no errors are found.
        Also prints a summary of errors found.
    """
    errors = []
    df = input_df.copy() # Work on a copy to preserve original index if needed
    df.reset_index(inplace=True) # Use 'index' for original row reference

    # --- Column-Specific Rules ---

    # Rule: id - Non-null
    null_ids = df[df['id'].isnull()]
    for idx, row in null_ids.iterrows():
        errors.append({
            'row_index': row['index'],
            'column_name': 'id',
            'failed_value': row['id'],
            'validation_rule_failed': 'Must be non-null'
        })

    # Rule: id - Unique string (Assuming already string, check uniqueness)
    # Note: Pandas duplicate check keeps the first occurrence, marks subsequent ones
    duplicated_ids = df[df['id'].duplicated(keep=False) & df['id'].notnull()]
    # Only report the *second* and subsequent occurrences as duplicates
    is_duplicate_marker = df['id'].duplicated(keep='first')
    for idx, row in df[is_duplicate_marker].iterrows():
         errors.append({
            'row_index': row['index'],
            'column_name': 'id',
            'failed_value': row['id'],
            'validation_rule_failed': 'Must be unique'
        })


    # Rule: update_timestamp - Non-null
    null_timestamps = df[df['update_timestamp'].isnull()]
    for idx, row in null_timestamps.iterrows():
        errors.append({
            'row_index': row['index'],
            'column_name': 'update_timestamp',
            'failed_value': row['update_timestamp'],
            'validation_rule_failed': 'Must be non-null'
        })

    # Rule: update_timestamp - ISO 8601 format (basic check using pandas)
    # Convert to datetime, coerce errors to NaT (Not a Time)
    parsed_timestamps = pd.to_datetime(df['update_timestamp'], errors='coerce')
    invalid_format_timestamps = df[parsed_timestamps.isnull() & df['update_timestamp'].notnull()]
    for idx, row in invalid_format_timestamps.iterrows():
        errors.append({
            'row_index': row['index'],
            'column_name': 'update_timestamp',
            'failed_value': row['update_timestamp'],
            'validation_rule_failed': 'Must be valid ISO 8601 format (parsable by pandas)'
        })
    # Add parsed timestamp for subsequent checks if needed
    df['parsed_timestamp'] = parsed_timestamps


    # Rule: total_spots - Non-null
    null_total_spots = df[df['total_spots'].isnull()]
    for idx, row in null_total_spots.iterrows():
         errors.append({
            'row_index': row['index'],
            'column_name': 'total_spots',
            'failed_value': row['total_spots'],
            'validation_rule_failed': 'Must be non-null'
        })

    # Rule: total_spots - Integer >= 0 (Check type and value)
    # Ensure numeric first, handling potential errors
    df['total_spots_numeric'] = pd.to_numeric(df['total_spots'], errors='coerce')
    invalid_type_total = df[df['total_spots_numeric'].isnull() & df['total_spots'].notnull()]
    for idx, row in invalid_type_total.iterrows():
        errors.append({
            'row_index': row['index'],
            'column_name': 'total_spots',
            'failed_value': row['total_spots'],
            'validation_rule_failed': 'Must be a valid number'
        })
    # Check range and integer part
    invalid_value_total = df[
        (df['total_spots_numeric'].notnull()) &
        ((df['total_spots_numeric'] < 0) | (df['total_spots_numeric'] != df['total_spots_numeric'].round()))
    ]
    for idx, row in invalid_value_total.iterrows():
        errors.append({
            'row_index': row['index'],
            'column_name': 'total_spots',
            'failed_value': row['total_spots'],
            'validation_rule_failed': 'Must be integer >= 0'
        })


    # Rule: available_spots - Non-null
    null_avail_spots = df[df['available_spots'].isnull()]
    for idx, row in null_avail_spots.iterrows():
         errors.append({
            'row_index': row['index'],
            'column_name': 'available_spots',
            'failed_value': row['available_spots'],
            'validation_rule_failed': 'Must be non-null'
        })

    # Rule: available_spots - Integer >= 0 (Check type and value)
    df['available_spots_numeric'] = pd.to_numeric(df['available_spots'], errors='coerce')
    invalid_type_avail = df[df['available_spots_numeric'].isnull() & df['available_spots'].notnull()]
    for idx, row in invalid_type_avail.iterrows():
        errors.append({
            'row_index': row['index'],
            'column_name': 'available_spots',
            'failed_value': row['available_spots'],
            'validation_rule_failed': 'Must be a valid number'
        })
    # Check range and integer part
    invalid_value_avail = df[
        (df['available_spots_numeric'].notnull()) &
        ((df['available_spots_numeric'] < 0) | (df['available_spots_numeric'] != df['available_spots_numeric'].round()))
     ]
    for idx, row in invalid_value_avail.iterrows():
        errors.append({
            'row_index': row['index'],
            'column_name': 'available_spots',
            'failed_value': row['available_spots'],
            'validation_rule_failed': 'Must be integer >= 0'
        })

    # Rule: occupancy_rate - Non-null
    null_occ_rate = df[df['occupancy_rate'].isnull()]
    for idx, row in null_occ_rate.iterrows():
         errors.append({
            'row_index': row['index'],
            'column_name': 'occupancy_rate',
            'failed_value': row['occupancy_rate'],
            'validation_rule_failed': 'Must be non-null'
        })

    # Rule: occupancy_rate - Float between 0.0 and 1.0
    df['occupancy_rate_numeric'] = pd.to_numeric(df['occupancy_rate'], errors='coerce')
    invalid_type_occ = df[df['occupancy_rate_numeric'].isnull() & df['occupancy_rate'].notnull()]
    for idx, row in invalid_type_occ.iterrows():
        errors.append({
            'row_index': row['index'],
            'column_name': 'occupancy_rate',
            'failed_value': row['occupancy_rate'],
            'validation_rule_failed': 'Must be a valid number (float)'
        })
    # Check range
    invalid_range_occ = df[
        (df['occupancy_rate_numeric'].notnull()) &
        ((df['occupancy_rate_numeric'] < 0.0) | (df['occupancy_rate_numeric'] > 1.0))
    ]
    for idx, row in invalid_range_occ.iterrows():
        errors.append({
            'row_index': row['index'],
            'column_name': 'occupancy_rate',
            'failed_value': row['occupancy_rate'],
            'validation_rule_failed': 'Must be between 0.0 and 1.0 inclusive'
        })

    # --- Row-Level/Cross-Column Rules ---

    # Rule: available_spots <= total_spots
    # Only check where both values are valid numbers already
    valid_spots_rows = df[df['available_spots_numeric'].notnull() & df['total_spots_numeric'].notnull()]
    invalid_spot_comparison = valid_spots_rows[
        valid_spots_rows['available_spots_numeric'] > valid_spots_rows['total_spots_numeric']
    ]
    for idx, row in invalid_spot_comparison.iterrows():
        errors.append({
            'row_index': row['index'],
            'column_name': 'available_spots/total_spots', # Indicate multiple columns involved
            'failed_value': f"available={row['available_spots']}, total={row['total_spots']}",
            'validation_rule_failed': 'available_spots must be <= total_spots'
        })

    # --- Prepare and Return Output ---

    error_df = pd.DataFrame(errors)

    # Print Summary Report
    print("--- Validation Summary ---")
    if error_df.empty:
        print("Validation PASSED: No errors found.")
    else:
        print(f"Validation FAILED: Found {len(error_df)} errors in {error_df['row_index'].nunique()} rows.")
        print("\nError Counts per Rule:")
        print(error_df['validation_rule_failed'].value_counts())
        print("\nError Counts per Column:")
        print(error_df['column_name'].value_counts())
        # Remove duplicates based on row_index, column, and rule to avoid overcounting
        # if the same cell violates multiple aspects (e.g., wrong type AND out of range)
        # This might not be strictly necessary depending on how you count errors.
        # Example: error_df.drop_duplicates(subset=['row_index', 'column_name', 'validation_rule_failed'])
        print("-------------------------")


    # Return DataFrame of errors
    if not error_df.empty:
        # Sort for easier reading
        error_df.sort_values(by=['row_index', 'column_name'], inplace=True)
        # Select and order columns as requested
        error_df = error_df[['row_index', 'column_name', 'failed_value', 'validation_rule_failed']]

    return error_df

# --- Example Usage --- (Commented out)
# Assuming 'input_df' is your DataFrame loaded from the JSON/CSV
# Example DataFrames for testing:
# Valid Data
# data_valid = {
#     'id': ['A1', 'B2', 'C3'],
#     'update_timestamp': ['2023-01-01T10:00:00Z', '2023-01-01T10:05:00Z', '2023-01-01T10:10:00Z'],
#     'total_spots': [100, 50, 200],
#     'available_spots': [20, 50, 10],
#     'occupancy_rate': [0.8, 0.0, 0.95]
# }
# df_valid = pd.DataFrame(data_valid)
# errors_valid = validate_parking_data(df_valid)
# print("\nValidation Errors for Valid Data:")
# print(errors_valid)

# # Invalid Data
# data_invalid = {
#     'id': ['A1', None, 'C3', 'A1', 'D4'], # Null and duplicate ID
#     'update_timestamp': ['2023-01-01T10:00:00Z', '2023-01-01T10:05:00Z', 'bad-date', '2023-01-01T10:15:00Z', '2023-01-01T10:20:00Z'], # Bad format
#     'total_spots': [100, 50, -5, 100, 20.5], # Negative and float
#     'available_spots': [20, 60, 10, 90, None], # Exceeds total, Null
#     'occupancy_rate': [0.8, 0.0, 0.95, 1.1, -0.1] # Out of range
# }
# df_invalid = pd.DataFrame(data_invalid)
# errors_invalid = validate_parking_data(df_invalid)
# print("\nValidation Errors for Invalid Data:")
# print(errors_invalid) 