import pandas as pd
import sys

file_path = 'sample_data.csv'
loc_to_check = 'LOC_2'

print(f'--- Analyzing {file_path} for consecutive zero occupancy ({loc_to_check}) ---')

try:
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_loc = df[df['location_id'] == loc_to_check].sort_values(by='timestamp').copy()

    if df_loc.empty:
        print(f'{loc_to_check} not found in data.')
        sys.exit()

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

except FileNotFoundError:
    print(f'Error: File not found at {file_path}', file=sys.stderr)
except Exception as e:
    print(f'An error occurred: {e}', file=sys.stderr) 