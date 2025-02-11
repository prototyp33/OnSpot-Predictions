import pandas as pd
import os

# ✅ Ensure logs directory exists
os.makedirs("../logs", exist_ok=True)

# ✅ Load the cleaned dataset
df = pd.read_csv("/Users/adrianiraeguialvear/OnSpot_Predictive_Model/data/barcelona_parking.parking_predictions.csv")

# ✅ Convert "datetime" column to actual datetime format
df["datetime"] = pd.to_datetime(df["datetime"])

# ✅ Extract time-based features
df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.weekday  # Monday = 0, Sunday = 6
df["month"] = df["datetime"].dt.month

# ✅ Save the updated dataset
df.to_csv("/Users/adrianiraeguialvear/OnSpot_Predictive_Model/data/cleaned_parking_data_with_features.csv", index=False)

# ✅ Log the dataset check
with open("../logs/data_check.log", "w") as log_file:
    log_file.write(f"Dataset Shape: {df.shape}\n")
    log_file.write(f"Columns: {df.columns.tolist()}\n")

print("🚀 Data loaded, missing features added, and log saved!")
