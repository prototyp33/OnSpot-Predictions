import pandas as pd
# Assuming you have 'parking_type_osm' and 'parking_type_siu' columns
contingency_table = pd.crosstab(merged_data['parking_type_osm'], merged_data['parking_type_siu'])
print(contingency_table)