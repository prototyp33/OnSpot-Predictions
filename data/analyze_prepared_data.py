import pandas as pd
import sys

# Use the correct file name
file_path = 'prepared_sample_data_features.csv'
print(f'--- Analyzing {file_path} ---')

try:
    df = pd.read_csv(file_path)

    # Convert timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    print('\n--- Basic Info ---')
    df.info()

    print('\n--- Missing Values (Sum) ---')
    missing_vals = df.isnull().sum()
    print(missing_vals[missing_vals > 0]) # Only show columns with missing values
    if missing_vals.sum() == 0:
        print('No missing values found.')

    print('\n--- Summary Statistics (Occupancy & Capacity) ---')
    if 'occupancy' in df.columns:
        print("Occupancy Stats:")
        print(df['occupancy'].describe())
    if 'capacity' in df.columns:
        print("\nCapacity Stats:")
        print(df['capacity'].describe())

    print('\n--- Unique Location IDs ---')
    if 'location_id' in df.columns:
        print(df['location_id'].unique())
    else:
        print('location_id column not found.')

    print('\n--- Occupancy Range Check ---')
    if 'occupancy' in df.columns:
        min_occ = df['occupancy'].min()
        max_occ = df['occupancy'].max()
        print(f'Min Occupancy: {min_occ}')
        print(f'Max Occupancy: {max_occ}')
        invalid_occ = df[(df['occupancy'] < 0) | (df['occupancy'] > 100)]
        print(f'Rows with Occupancy outside [0, 100]: {len(invalid_occ)}')
    else:
        print('occupancy column not found.')

    print('\n--- Capacity Range Check ---')
    if 'capacity' in df.columns:
        invalid_cap = df[df['capacity'] <= 0]
        print(f'Rows with Non-Positive Capacity: {len(invalid_cap)}')
    else:
        print('capacity column not found.')

    print('\n--- Checking for Constant Occupancy Periods (LOC_2) ---')
    if 'location_id' in df.columns and 'timestamp' in df.columns and 'occupancy' in df.columns:
        loc_to_check = 'LOC_2'
        if loc_to_check in df['location_id'].unique():
            df_loc = df[df['location_id'] == loc_to_check].sort_values(by='timestamp')
            df_loc['constant_occupancy'] = (df_loc['occupancy'] == df_loc['occupancy'].shift())
            constant_periods = df_loc[df_loc['constant_occupancy'] == True]
            print(f'Found {len(constant_periods)} instances where occupancy for {loc_to_check} is identical to the previous hour.')
            if not constant_periods.empty:
                # Check if constant zeros are present
                constant_zeros = constant_periods[constant_periods['occupancy'] == 0.0]
                print(f'  - Instances where the constant value was 0.0: {len(constant_zeros)}')
                if len(constant_zeros) > 0:
                     print('  - Example of constant zero occupancy:')
                     print(constant_zeros[['timestamp', 'location_id', 'occupancy']].head())
        else:
            print(f'Location {loc_to_check} not found in the data.')
    else:
        print('Required columns (location_id, timestamp, occupancy) not found for constant check.')

    print('\n--- Analyzing Durations of Constant Zero Occupancy (LOC_2) ---')
    if 'location_id' in df.columns and 'timestamp' in df.columns and 'occupancy' in df.columns:
        loc_to_check = 'LOC_2'
        if loc_to_check in df['location_id'].unique():
            df_loc = df[df['location_id'] == loc_to_check].sort_values(by='timestamp').copy()

            # Identify where occupancy is zero
            df_loc['is_zero'] = df_loc['occupancy'] == 0.0
            # Identify changes in the zero state (start of a new block)
            df_loc['zero_block_change'] = df_loc['is_zero'].diff()
            # Assign a unique ID to each consecutive block
            df_loc['zero_block_id'] = df_loc['zero_block_change'].abs().cumsum()

            # Filter for only the zero occupancy blocks
            zero_blocks = df_loc[df_loc['is_zero']]

            if not zero_blocks.empty:
                # Calculate duration of each zero block
                block_durations = zero_blocks.groupby('zero_block_id').size()

                # Filter out single-hour zero periods (we are interested in *consecutive* zeros)
                consecutive_zero_durations = block_durations[block_durations > 1]

                if not consecutive_zero_durations.empty:
                    print(f'Found {len(consecutive_zero_durations)} periods where occupancy for {loc_to_check} was 0.0 for 2 or more consecutive hours.')
                    print('\nDistribution of consecutive zero-occupancy durations (in hours):')
                    print(consecutive_zero_durations.value_counts().sort_index())
                else:
                    print(f'No periods found where occupancy for {loc_to_check} was 0.0 for 2 or more consecutive hours.')
            else:
                print(f'No zero occupancy values found for {loc_to_check}.')
        else:
            print(f'Location {loc_to_check} not found in the data.')
    else:
        print('Required columns not found for zero duration analysis.')

except FileNotFoundError:
    print(f'Error: File not found at {file_path}', file=sys.stderr)
except Exception as e:
    print(f'An error occurred: {e}', file=sys.stderr) 