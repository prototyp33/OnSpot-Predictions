import pandas as pd
import sys

file_path = 'merged_parking_data.csv'
chunk_size = 100000  # Increased chunk size slightly
print(f'--- Analyzing {file_path} in chunks of {chunk_size} ---')

total_rows = 0
min_timestamp = None
max_timestamp = None
missing_values_total = None
invalid_occ_count = 0
invalid_cap_count = 0
constant_loc2_zero_count = 0
occupancy_level_type = None
location_id_col = 'zone_id_categorical' # Use 'zone_id_categorical' as location identifier
timestamp_col = 'datetime'
occupancy_col = 'occupancy_level'
capacity_col = 'capacity'
loc_to_check = 'LOC_2' # We need to confirm if LOC_2 exists in zone_id_categorical
zero_val_to_check = 0.0 # Assumption for now, might need adjustment based on dtype

try:
    iterator = pd.read_csv(file_path, chunksize=chunk_size, iterator=True, low_memory=False) # Added low_memory=False

    for i, chunk in enumerate(iterator):
        print(f'Processing chunk {i+1}...', end='\r')

        # Check occupancy_level dtype on first chunk
        if i == 0:
            occupancy_level_type = chunk[occupancy_col].dtype
            print(f"Detected '{occupancy_col}' dtype: {occupancy_level_type}\n")
            # Add check here if loc_to_check actually exists in the location column
            if loc_to_check not in chunk[location_id_col].unique():
                 print(f"Warning: '{loc_to_check}' not found in '{location_id_col}' in the first chunk. Constant check might yield 0.")


        # --- Basic Info ---
        total_rows += len(chunk)
        chunk[timestamp_col] = pd.to_datetime(chunk[timestamp_col])
        if min_timestamp is None or chunk[timestamp_col].min() < min_timestamp:
            min_timestamp = chunk[timestamp_col].min()
        if max_timestamp is None or chunk[timestamp_col].max() > max_timestamp:
            max_timestamp = chunk[timestamp_col].max()

        # --- Missing Values ---
        current_missing = chunk.isnull().sum()
        if missing_values_total is None:
            missing_values_total = current_missing
        else:
            missing_values_total += current_missing

        # --- Range Checks ---
        # Check occupancy range only if it's numeric
        if pd.api.types.is_numeric_dtype(chunk[occupancy_col]):
             invalid_occ_count += len(chunk[(chunk[occupancy_col] < 0) | (chunk[occupancy_col] > 100)])
             # Re-evaluate the zero value to check if numeric
             zero_val_to_check = 0.0
        else:
             # If not numeric, perhaps check for specific string values like 'EMPTY'?
             # For now, we'll skip the range check and the zero check needs adjustment
             zero_val_to_check = 'UNKNOWN_ZERO' # Placeholder, won't match unless type is object and value is this
             if i == 0: print(f"Note: '{occupancy_col}' is not numeric. Skipping [0,100] range check.")

        # Check capacity range (assuming it exists and is numeric)
        if capacity_col in chunk.columns and pd.api.types.is_numeric_dtype(chunk[capacity_col]):
            invalid_cap_count += len(chunk[chunk[capacity_col] <= 0])
        elif i==0 and capacity_col not in chunk.columns:
             print(f"Warning: Column '{capacity_col}' not found.")

        # --- Constant LOC_2 Zero Check (within chunk) ---
        # Check if location column exists
        if location_id_col in chunk.columns:
            chunk_loc2 = chunk[chunk[location_id_col] == loc_to_check].sort_values(by=timestamp_col)
            if not chunk_loc2.empty and occupancy_col in chunk_loc2.columns:
                # Compare with the shifted value within the sorted LOC_2 subset
                try:
                    constant_zeros = chunk_loc2[(chunk_loc2[occupancy_col] == zero_val_to_check) & (chunk_loc2[occupancy_col].shift() == zero_val_to_check)]
                    constant_loc2_zero_count += len(constant_zeros)
                except TypeError:
                     if i == 0: print(f"Note: Cannot compare '{occupancy_col}' with {zero_val_to_check}. Adjust check if needed.")
                     pass # Cannot perform comparison
        elif i==0:
            print(f"Warning: Column '{location_id_col}' not found.")

    print('\nProcessing complete.')

    # --- Print Results ---
    print('\n--- Analysis Summary ---')
    print(f'Total rows processed: {total_rows}')
    print(f'Time range: {min_timestamp} to {max_timestamp}')
    print(f'Occupancy data type found: {occupancy_level_type}')

    print('\n--- Missing Values (Total) ---')
    if missing_values_total is not None:
        print(missing_values_total[missing_values_total > 0])
        if missing_values_total.sum() == 0:
             print('No missing values found.')
    else:
        print('No data processed.')

    print('\n--- Range Check Results ---')
    print(f'Rows with Occupancy outside [0, 100] (numeric only): {invalid_occ_count}')
    print(f'Rows with Non-Positive Capacity: {invalid_cap_count}')

    print(f"\n--- Constant Occupancy Check ('{loc_to_check}' at '{zero_val_to_check}') ---")
    print(f"Approximate instances of consecutive '{zero_val_to_check}' occupancy for '{loc_to_check}': {constant_loc2_zero_count}")
    print('(Note: This count checks within chunks and may slightly undercount sequences spanning across chunks)')


except FileNotFoundError:
    print(f'Error: File not found at {file_path}', file=sys.stderr)
except KeyError as e:
     print(f'\nError: Missing expected column in CSV: {e}. Please check column names.', file=sys.stderr)
except Exception as e:
    print(f'\nAn error occurred during processing: {e}', file=sys.stderr) 