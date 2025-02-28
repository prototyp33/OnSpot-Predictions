#!/usr/bin/env python
"""
Exploratory Data Analysis for the prepared_data_improved.csv dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from scipy import stats
import matplotlib.dates as mdates

# Set style for better visualizations
plt.style.use('ggplot')
sns.set_palette("deep")
sns.set_context("notebook", font_scale=1.2)

# Create output directory
output_dir = "eda_results"
os.makedirs(output_dir, exist_ok=True)

def load_and_prepare_data(file_path):
    """Load and prepare the dataset for analysis."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract time components
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_name'] = df['timestamp'].dt.day_name()
    df['month'] = df['timestamp'].dt.month
    df['month_name'] = df['timestamp'].dt.month_name()
    df['is_weekend'] = df['day_of_week'] >= 5
    df['date'] = df['timestamp'].dt.date
    
    print(f"Dataset loaded with shape: {df.shape}")
    return df

def basic_data_exploration(df):
    """Perform basic data exploration."""
    print("\n=== BASIC DATA EXPLORATION ===")
    
    # Display basic information
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check data types and missing values
    print("\nData types and missing values:")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percent
    })
    print(missing_df[missing_df['Missing Values'] > 0])
    
    # Summary statistics
    print("\nSummary statistics:")
    print(df.describe().T)
    
    # Check unique values for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\nUnique values in {col}:")
        print(df[col].value_counts())
    
    # Save summary to file
    with open(os.path.join(output_dir, "data_summary.txt"), "w") as f:
        f.write(f"Dataset shape: {df.shape}\n\n")
        f.write("Summary statistics:\n")
        f.write(df.describe().T.to_string())
        
        f.write("\n\nMissing values:\n")
        f.write(missing_df[missing_df['Missing Values'] > 0].to_string())
        
        for col in categorical_cols:
            f.write(f"\n\nUnique values in {col}:\n")
            f.write(df[col].value_counts().to_string())

def analyze_occupancy_distribution(df):
    """Analyze the distribution of occupancy values."""
    print("\n=== OCCUPANCY DISTRIBUTION ANALYSIS ===")
    
    # Overall distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['occupancy'], kde=True, bins=50)
    plt.title('Distribution of Occupancy Values')
    plt.xlabel('Occupancy')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, "occupancy_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Distribution by location
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='location_id', y='occupancy', data=df)
    plt.title('Occupancy Distribution by Location')
    plt.xlabel('Location ID')
    plt.ylabel('Occupancy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "occupancy_by_location.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Distribution by parking type
    if 'parking_type' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='parking_type', y='occupancy', data=df)
        plt.title('Occupancy Distribution by Parking Type')
        plt.xlabel('Parking Type')
        plt.ylabel('Occupancy')
        plt.savefig(os.path.join(output_dir, "occupancy_by_parking_type.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Distribution by zone type
    if 'zone_type' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='zone_type', y='occupancy', data=df)
        plt.title('Occupancy Distribution by Zone Type')
        plt.xlabel('Zone Type')
        plt.ylabel('Occupancy')
        plt.savefig(os.path.join(output_dir, "occupancy_by_zone_type.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Occupancy distribution analysis completed.")

def analyze_temporal_patterns(df):
    """Analyze temporal patterns in the data."""
    print("\n=== TEMPORAL PATTERNS ANALYSIS ===")
    
    # Occupancy by hour of day
    plt.figure(figsize=(12, 6))
    hourly_avg = df.groupby('hour')['occupancy'].mean()
    hourly_std = df.groupby('hour')['occupancy'].std()
    
    plt.plot(hourly_avg.index, hourly_avg.values, 'o-', linewidth=2, label='Mean Occupancy')
    plt.fill_between(
        hourly_avg.index, 
        hourly_avg.values - hourly_std.values, 
        hourly_avg.values + hourly_std.values, 
        alpha=0.2
    )
    plt.title('Average Occupancy by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Average Occupancy')
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "occupancy_by_hour.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Occupancy by day of week
    plt.figure(figsize=(12, 6))
    day_avg = df.groupby('day_name')['occupancy'].mean().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    day_std = df.groupby('day_name')['occupancy'].std().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    
    plt.bar(day_avg.index, day_avg.values, yerr=day_std.values, capsize=5)
    plt.title('Average Occupancy by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Occupancy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "occupancy_by_day.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Occupancy by month
    plt.figure(figsize=(12, 6))
    month_avg = df.groupby('month_name')['occupancy'].mean().reindex([
        'January', 'February', 'March', 'April', 'May', 'June', 
        'July', 'August', 'September', 'October', 'November', 'December'
    ])
    month_std = df.groupby('month_name')['occupancy'].std().reindex([
        'January', 'February', 'March', 'April', 'May', 'June', 
        'July', 'August', 'September', 'October', 'November', 'December'
    ])
    
    plt.bar(month_avg.index, month_avg.values, yerr=month_std.values, capsize=5)
    plt.title('Average Occupancy by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Occupancy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "occupancy_by_month.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Weekday vs Weekend
    plt.figure(figsize=(10, 6))
    weekend_avg = df.groupby('is_weekend')['occupancy'].mean()
    weekend_std = df.groupby('is_weekend')['occupancy'].std()
    
    plt.bar([0, 1], weekend_avg.values, yerr=weekend_std.values, capsize=5)
    plt.title('Average Occupancy: Weekday vs Weekend')
    plt.xlabel('Day Type')
    plt.ylabel('Average Occupancy')
    plt.xticks([0, 1], ['Weekday', 'Weekend'])
    plt.savefig(os.path.join(output_dir, "weekday_vs_weekend.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Temporal patterns analysis completed.")

def analyze_weather_impact(df):
    """Analyze the impact of weather on occupancy."""
    print("\n=== WEATHER IMPACT ANALYSIS ===")
    
    # Check if weather columns exist
    weather_cols = ['temperature', 'humidity', 'wind_speed', 'precipitation']
    existing_weather_cols = [col for col in weather_cols if col in df.columns]
    
    if not existing_weather_cols:
        print("No weather data found in the dataset.")
        return
    
    # Correlation heatmap for weather variables
    plt.figure(figsize=(12, 10))
    corr_cols = ['occupancy'] + existing_weather_cols
    corr = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation between Occupancy and Weather Variables')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "weather_correlation.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Scatter plots for each weather variable
    for col in existing_weather_cols:
        plt.figure(figsize=(10, 6))
        
        # Add regression line
        sns.regplot(x=col, y='occupancy', data=df, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
        
        plt.title(f'Occupancy vs {col.capitalize()}')
        plt.xlabel(col.capitalize())
        plt.ylabel('Occupancy')
        plt.savefig(os.path.join(output_dir, f"occupancy_vs_{col}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Binned analysis
        try:
            temp_df = df.copy()
            temp_df[f'{col}_bin'] = pd.qcut(temp_df[col], 10, duplicates='drop')
            
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=f'{col}_bin', y='occupancy', data=temp_df)
            plt.title(f'Occupancy by {col.capitalize()} Ranges')
            plt.xlabel(f'{col.capitalize()} Ranges')
            plt.ylabel('Occupancy')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"occupancy_by_{col}_bins.png"), dpi=300, bbox_inches='tight')
            plt.close()
        except:
            print(f"Could not create binned analysis for {col}")
    
    print("Weather impact analysis completed.")

def analyze_time_series(df):
    """Analyze time series patterns for each location."""
    print("\n=== TIME SERIES ANALYSIS ===")
    
    # Create a directory for time series analysis
    ts_dir = os.path.join(output_dir, "time_series")
    os.makedirs(ts_dir, exist_ok=True)
    
    # Time series analysis for each location
    locations = df['location_id'].unique()
    
    for location in locations:
        print(f"Analyzing time series for location {location}...")
        loc_df = df[df['location_id'] == location].sort_values('timestamp')
        
        # Plot time series
        plt.figure(figsize=(16, 8))
        plt.plot(loc_df['timestamp'], loc_df['occupancy'], '-', linewidth=1)
        plt.title(f'Occupancy Time Series - Location {location}')
        plt.xlabel('Time')
        plt.ylabel('Occupancy')
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        plt.savefig(os.path.join(ts_dir, f"timeseries_{location}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Weekly patterns
        if len(loc_df) > 168:  # At least a week of hourly data
            loc_df['day_hour'] = loc_df['day_of_week'] * 24 + loc_df['hour']
            weekly_pattern = loc_df.groupby('day_hour')['occupancy'].mean().reset_index()
            
            plt.figure(figsize=(16, 6))
            plt.plot(weekly_pattern['day_hour'], weekly_pattern['occupancy'], '-o', markersize=4)
            plt.title(f'Weekly Occupancy Pattern - Location {location}')
            plt.xlabel('Hour of Week (0=Monday 00:00, 167=Sunday 23:00)')
            plt.ylabel('Average Occupancy')
            
            # Add day separators
            for day in range(1, 7):
                plt.axvline(x=day*24, color='gray', linestyle='--', alpha=0.7)
            
            day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            plt.xticks([d*24 + 12 for d in range(7)], day_labels)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(ts_dir, f"weekly_pattern_{location}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Heatmap of weekly pattern
            weekly_data = np.zeros((7, 24))
            for day in range(7):
                for hour in range(24):
                    day_hour = day * 24 + hour
                    if day_hour in weekly_pattern['day_hour'].values:
                        weekly_data[day, hour] = weekly_pattern[
                            weekly_pattern['day_hour'] == day_hour
                        ]['occupancy'].values[0]
            
            plt.figure(figsize=(16, 8))
            sns.heatmap(
                weekly_data, 
                cmap='YlOrRd', 
                linewidths=0.5, 
                xticklabels=range(24),
                yticklabels=day_labels
            )
            plt.title(f'Weekly Occupancy Heatmap - Location {location}')
            plt.xlabel('Hour of Day')
            plt.ylabel('Day of Week')
            plt.tight_layout()
            plt.savefig(os.path.join(ts_dir, f"weekly_heatmap_{location}.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    print("Time series analysis completed.")

def analyze_anomalies(df):
    """Identify and analyze anomalies in the data."""
    print("\n=== ANOMALY DETECTION ===")
    
    # Create a directory for anomaly analysis
    anomaly_dir = os.path.join(output_dir, "anomalies")
    os.makedirs(anomaly_dir, exist_ok=True)
    
    # Anomaly detection for each location
    locations = df['location_id'].unique()
    
    for location in locations:
        print(f"Detecting anomalies for location {location}...")
        loc_df = df[df['location_id'] == location].copy().sort_values('timestamp')
        
        # Z-score method
        loc_df['occupancy_zscore'] = np.abs(stats.zscore(loc_df['occupancy']))
        anomalies = loc_df[loc_df['occupancy_zscore'] > 3]
        
        # Plot with anomalies highlighted
        plt.figure(figsize=(16, 8))
        plt.plot(loc_df['timestamp'], loc_df['occupancy'], '-', linewidth=1, label='Normal')
        plt.scatter(
            anomalies['timestamp'], 
            anomalies['occupancy'], 
            color='red', 
            s=50, 
            label=f'Anomalies ({len(anomalies)} points)'
        )
        plt.title(f'Occupancy Time Series with Anomalies - Location {location}')
        plt.xlabel('Time')
        plt.ylabel('Occupancy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        plt.savefig(os.path.join(anomaly_dir, f"anomalies_{location}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save anomalies to CSV
        if len(anomalies) > 0:
            anomalies.to_csv(os.path.join(anomaly_dir, f"anomalies_{location}.csv"), index=False)
    
    print("Anomaly detection completed.")

def analyze_location_factors(df):
    """Analyze location-specific factors."""
    print("\n=== LOCATION FACTORS ANALYSIS ===")
    
    # Create a directory for location analysis
    loc_dir = os.path.join(output_dir, "location_factors")
    os.makedirs(loc_dir, exist_ok=True)
    
    # Analyze capacity vs average occupancy
    if 'capacity' in df.columns:
        location_stats = df.groupby('location_id').agg({
            'occupancy': ['mean', 'std', 'min', 'max'],
            'capacity': 'first'
        })
        location_stats.columns = ['avg_occupancy', 'std_occupancy', 'min_occupancy', 'max_occupancy', 'capacity']
        
        plt.figure(figsize=(12, 8))
        plt.scatter(location_stats['capacity'], location_stats['avg_occupancy'], s=80, alpha=0.7)
        
        # Add location labels
        for idx, row in location_stats.iterrows():
            plt.text(row['capacity']*1.02, row['avg_occupancy'], idx, fontsize=9)
        
        plt.title('Average Occupancy vs Capacity by Location')
        plt.xlabel('Capacity')
        plt.ylabel('Average Occupancy')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(loc_dir, "occupancy_vs_capacity.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save location stats to CSV
        location_stats.to_csv(os.path.join(loc_dir, "location_stats.csv"))
    
    # Analyze zone type impact
    if 'zone_type' in df.columns:
        zone_stats = df.groupby(['location_id', 'zone_type']).agg({
            'occupancy': ['mean', 'std']
        }).reset_index()
        zone_stats.columns = ['location_id', 'zone_type', 'avg_occupancy', 'std_occupancy']
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x='location_id', y='avg_occupancy', hue='zone_type', data=zone_stats)
        plt.title('Average Occupancy by Location and Zone Type')
        plt.xlabel('Location ID')
        plt.ylabel('Average Occupancy')
        plt.legend(title='Zone Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(loc_dir, "occupancy_by_location_zone.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Location factors analysis completed.")

def create_summary_report(df):
    """Create a summary report of the analysis."""
    print("\n=== CREATING SUMMARY REPORT ===")
    
    # Calculate key metrics
    metrics = {
        'Total Observations': len(df),
        'Date Range': f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}",
        'Number of Locations': df['location_id'].nunique(),
        'Average Occupancy': f"{df['occupancy'].mean():.2f}",
        'Occupancy Range': f"{df['occupancy'].min():.2f} to {df['occupancy'].max():.2f}",
        'Missing Values': df.isnull().sum().sum()
    }
    
    # Add parking type distribution if available
    if 'parking_type' in df.columns:
        parking_type_counts = df['parking_type'].value_counts()
        for parking_type, count in parking_type_counts.items():
            metrics[f'Parking Type: {parking_type}'] = count
    
    # Add zone type distribution if available
    if 'zone_type' in df.columns:
        zone_type_counts = df['zone_type'].value_counts()
        for zone_type, count in zone_type_counts.items():
            metrics[f'Zone Type: {zone_type}'] = count
    
    # Create summary report
    with open(os.path.join(output_dir, "summary_report.txt"), "w") as f:
        f.write("=== DATASET SUMMARY REPORT ===\n\n")
        
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n=== LOCATION SUMMARY ===\n\n")
        location_summary = df.groupby('location_id')['occupancy'].agg(['count', 'mean', 'std', 'min', 'max'])
        f.write(location_summary.to_string())
        
        f.write("\n\n=== TEMPORAL PATTERNS ===\n\n")
        f.write("Hourly Patterns:\n")
        hourly_avg = df.groupby('hour')['occupancy'].mean()
        f.write(hourly_avg.to_string())
        
        f.write("\n\nDaily Patterns:\n")
        daily_avg = df.groupby('day_name')['occupancy'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        f.write(daily_avg.to_string())
        
        f.write("\n\nMonthly Patterns:\n")
        monthly_avg = df.groupby('month_name')['occupancy'].mean().reindex([
            'January', 'February', 'March', 'April', 'May', 'June', 
            'July', 'August', 'September', 'October', 'November', 'December'
        ])
        f.write(monthly_avg.to_string())
        
        # Add weather correlations if available
        weather_cols = ['temperature', 'humidity', 'wind_speed', 'precipitation']
        existing_weather_cols = [col for col in weather_cols if col in df.columns]
        
        if existing_weather_cols:
            f.write("\n\n=== WEATHER CORRELATIONS ===\n\n")
            weather_corr = df[['occupancy'] + existing_weather_cols].corr()['occupancy'].drop('occupancy')
            f.write(weather_corr.to_string())
    
    print("Summary report created.")

def main():
    """Main function to run the EDA."""
    # Load and prepare data
    df = load_and_prepare_data("data/prepared_data_improved.csv")
    
    # Run analyses
    basic_data_exploration(df)
    analyze_occupancy_distribution(df)
    analyze_temporal_patterns(df)
    analyze_weather_impact(df)
    analyze_time_series(df)
    analyze_anomalies(df)
    analyze_location_factors(df)
    create_summary_report(df)
    
    print("\nExploratory Data Analysis completed. Results saved to:", output_dir)

if __name__ == "__main__":
    main()
