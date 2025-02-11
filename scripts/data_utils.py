# In src/data_utils.py

import pandas as pd
import numpy as np

START_TIME = pd.to_datetime('08:00').time()
END_TIME = pd.to_datetime('20:00').time()

def create_slot_index(timestamp):
    """
    Calculate the slot index for a given timestamp.
    
    Parameters:
      timestamp (datetime-like): A Pandas Timestamp or datetime object.
    
    Returns:
      int: Slot index (0 to 143) if the time is within regulated hours (08:00 to 20:00),
           otherwise returns -1 (or np.nan if preferred).
    """
    time_of_day = timestamp.time()
    if START_TIME <= time_of_day < END_TIME:
        minutes_since_start = (timestamp.hour - 8) * 60 + timestamp.minute
        slot_index = minutes_since_start // 5
        return slot_index
    else:
        return -1  # or np.nan
