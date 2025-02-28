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

def analyze_weather_impact(df):
    """Analyze how weather variables impact occupancy."""
    print("\n=== WEATHER IMPACT ANALYSIS ===")
    
    # Create a directory for weather analysis
    weather_dir = os.path.join(output_dir, "weather_impact")
    os.makedirs(weather_dir, exist_ok=True)
    
    # Check if weather columns exist
    weather_cols = ['temperature', 'humidity', 'wind_speed', 'precipitation']
    existing_weather_cols = [col for col in weather_cols if col in df.columns]
    
    if not existing_weather_cols:
        print("No weather data found in the dataset.")
        return
    
    print(f"Analyzing impact of weather variables: {', '.join(existing_weather_cols)}")
    
    # Correlation analysis
    plt.figure(figsize=(12, 10))
    corr_cols = ['occupancy'] + existing_weather_cols
    corr = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation between Occupancy and Weather Variables')
    plt.tight_layout()
    plt.savefig(os.path.join(weather_dir, "weather_correlation.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze each weather variable
    for col in existing_weather_cols:
        # Scatter plot with trend line
        plt.figure(figsize=(12, 6))
        sns.regplot(x=col, y='occupancy', data=df, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
        plt.title(f'Occupancy vs {col.capitalize()}')
        plt.xlabel(col.capitalize())
        plt.ylabel('Occupancy')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(weather_dir, f"occupancy_vs_{col}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Binned analysis for non-linear relationships
        try:
            # Create bins (handle potential errors with qcut)
            df[f'{col}_bin'] = pd.qcut(df[col], 10, duplicates='drop')
            
            # Calculate average occupancy per bin
            bin_stats = df.groupby(f'{col}_bin')['occupancy'].agg(['mean', 'std', 'count']).reset_index()
            
            # Plot binned analysis
            plt.figure(figsize=(14, 6))
            plt.errorbar(
                x=range(len(bin_stats)), 
                y=bin_stats['mean'], 
                yerr=bin_stats['std'], 
                fmt='o-', 
                capsize=5,
                linewidth=2
            )
            
            # Add bin labels
            bin_labels = [str(x) for x in bin_stats[f'{col}_bin']]
            plt.xticks(range(len(bin_stats)), bin_labels, rotation=45)
            
            plt.title(f'Average Occupancy by {col.capitalize()} Ranges')
            plt.xlabel(f'{col.capitalize()} Ranges')
            plt.ylabel('Average Occupancy')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(weather_dir, f"occupancy_by_{col}_bins.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Remove temporary bin column
            df = df.drop(f'{col}_bin', axis=1)
        except Exception as e:
            print(f"Could not create binned analysis for {col}: {e}")
    
    # Analyze weather combinations (temperature + precipitation)
    if 'temperature' in existing_weather_cols and 'precipitation' in existing_weather_cols:
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            df['temperature'], 
            df['precipitation'], 
            c=df['occupancy'], 
            cmap='viridis', 
            alpha=0.6,
            s=50
        )
        plt.colorbar(scatter, label='Occupancy')
        plt.title('Occupancy by Temperature and Precipitation')
        plt.xlabel('Temperature')
        plt.ylabel('Precipitation')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(weather_dir, "occupancy_temp_precip.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Analyze seasonal patterns with weather
    if 'month' in df.columns and 'temperature' in existing_weather_cols:
        # Monthly temperature and occupancy
        monthly_data = df.groupby('month').agg({
            'occupancy': 'mean',
            'temperature': 'mean'
        }).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot occupancy
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Average Occupancy', color='tab:blue')
        ax1.plot(monthly_data['month'], monthly_data['occupancy'], 'o-', color='tab:blue', linewidth=2)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # Create second y-axis for temperature
        ax2 = ax1.twinx()
        ax2.set_ylabel('Average Temperature', color='tab:red')
        ax2.plot(monthly_data['month'], monthly_data['temperature'], 'o-', color='tab:red', linewidth=2)
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        plt.title('Monthly Average Occupancy and Temperature')
        plt.xticks(range(1, 13))
        fig.tight_layout()
        plt.savefig(os.path.join(weather_dir, "monthly_temp_occupancy.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
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
            
            # Create weekly heatmap
            weekly_data = np.zeros((7, 24))
            day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            for day in range(7):
                for hour in range(24):
                    day_hour = day * 24 + hour
                    if day_hour in weekly_pattern['day_hour'].values:
                        weekly_data[day, hour] = weekly_pattern.loc[
                            weekly_pattern['day_hour'] == day_hour, 'occupancy'
                        ].values[0]
            
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

def analyze_additional_variables(df):
    """Analyze additional variables that might impact occupancy."""
    print("\n=== ADDITIONAL VARIABLES ANALYSIS ===")
    
    # Create a directory for additional analysis
    add_dir = os.path.join(output_dir, "additional_variables")
    os.makedirs(add_dir, exist_ok=True)
    
    # List of potential additional variables to check
    potential_vars = [
        'traffic_level', 'distance_to_center', 'parking_type', 'parking_fee',
        'special_event', 'holiday', 'capacity_utilization', 'zone_type',
        'nearby_attractions', 'nearby_businesses', 'public_transport_proximity'
    ]
    
    # Find which variables exist in the dataset
    existing_vars = [var for var in potential_vars if var in df.columns]
    
    if not existing_vars:
        print("No additional variables found for analysis.")
        return
    
    print(f"Analyzing additional variables: {', '.join(existing_vars)}")
    
    # Analyze each variable
    for var in existing_vars:
        # Check if variable is numeric
        if np.issubdtype(df[var].dtype, np.number):
            # Correlation with occupancy
            correlation = df[[var, 'occupancy']].corr().iloc[0, 1]
            
            # Scatter plot
            plt.figure(figsize=(12, 6))
            sns.regplot(x=var, y='occupancy', data=df, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
            plt.title(f'Occupancy vs {var.replace("_", " ").title()} (Correlation: {correlation:.2f})')
            plt.xlabel(var.replace("_", " ").title())
            plt.ylabel('Occupancy')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(add_dir, f"occupancy_vs_{var}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Binned analysis
            try:
                # Create bins
                df[f'{var}_bin'] = pd.qcut(df[var], 5, duplicates='drop')
                
                # Boxplot by bins
                plt.figure(figsize=(14, 6))
                sns.boxplot(x=f'{var}_bin', y='occupancy', data=df)
                plt.title(f'Occupancy by {var.replace("_", " ").title()} Ranges')
                plt.xlabel(f'{var.replace("_", " ").title()} Ranges')
                plt.ylabel('Occupancy')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(add_dir, f"occupancy_by_{var}_bins.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Remove temporary bin column
                df = df.drop(f'{var}_bin', axis=1)
            except Exception as e:
                print(f"Could not create binned analysis for {var}: {e}")
        else:
            # Categorical variable
            plt.figure(figsize=(14, 6))
            sns.boxplot(x=var, y='occupancy', data=df)
            plt.title(f'Occupancy by {var.replace("_", " ").title()}')
            plt.xlabel(var.replace("_", " ").title())
            plt.ylabel('Occupancy')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(add_dir, f"occupancy_by_{var}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Bar chart of average occupancy by category
            plt.figure(figsize=(14, 6))
            category_avg = df.groupby(var)['occupancy'].mean().sort_values(ascending=False)
            category_std = df.groupby(var)['occupancy'].std()
            
            plt.bar(
                category_avg.index, 
                category_avg.values, 
                yerr=category_std.values, 
                capsize=5
            )
            plt.title(f'Average Occupancy by {var.replace("_", " ").title()}')
            plt.xlabel(var.replace("_", " ").title())
            plt.ylabel('Average Occupancy')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(add_dir, f"avg_occupancy_by_{var}.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    # Analyze interactions between variables if multiple exist
    if len(existing_vars) > 1:
        # Create correlation matrix for all variables
        plt.figure(figsize=(12, 10))
        corr_vars = ['occupancy'] + existing_vars
        corr = df[corr_vars].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation between Occupancy and Additional Variables')
        plt.tight_layout()
        plt.savefig(os.path.join(add_dir, "additional_vars_correlation.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Additional variables analysis completed.")

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
    analyze_additional_variables(df)
    
    print("\nExploratory Data Analysis completed. Results saved to:", output_dir)

if __name__ == "__main__":
    main() 