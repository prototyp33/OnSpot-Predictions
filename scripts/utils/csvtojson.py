import pandas as pd

# Step 1: Load the CSV file
df = pd.read_csv('/Users/adrianiraeguialvear/OnSpot_Predictive_Model/data/feature_engineered_data.csv')


json_data = df.to_json(orient='records', lines=True)

# Step 3: Write the JSON data to a file
with open('feature_engineered.json', 'w') as json_file:
    json_file.write(json_data)

print("Conversion completed. JSON file created as 'feature_engineered.json'")