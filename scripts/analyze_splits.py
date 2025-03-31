import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import logging
from typing import Dict, List, Tuple
import joblib
from sklearn.compose import ColumnTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_splits(splits_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the train, validation, and test splits.
    
    Args:
        splits_dir (str): Directory containing the splits
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test sets
    """
    train_df = pd.read_csv(os.path.join(splits_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(splits_dir, 'validation.csv'))
    test_df = pd.read_csv(os.path.join(splits_dir, 'test.csv'))
    
    # Convert timestamp to datetime
    for df in [train_df, val_df, test_df]:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return train_df, val_df, test_df

def calculate_basic_stats(dfs: List[pd.DataFrame], names: List[str]) -> pd.DataFrame:
    """
    Calculate basic statistics for each numerical column in each dataframe.
    
    Args:
        dfs (List[pd.DataFrame]): List of dataframes to analyze
        names (List[str]): Names of the dataframes (e.g., ['Train', 'Validation', 'Test'])
        
    Returns:
        pd.DataFrame: DataFrame containing statistics
    """
    stats_dfs = []
    
    for df, name in zip(dfs, names):
        # Select numerical columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        # Calculate statistics
        stats = df[num_cols].agg(['mean', 'std', 'min', 'max', 'median'])
        stats.columns = pd.MultiIndex.from_product([[name], stats.columns])
        stats_dfs.append(stats)
    
    return pd.concat(stats_dfs, axis=1)

def analyze_categorical_distributions(dfs: List[pd.DataFrame], names: List[str], output_dir: str):
    """
    Analyze and plot the distribution of categorical variables.
    
    Args:
        dfs (List[pd.DataFrame]): List of dataframes to analyze
        names (List[str]): Names of the dataframes
        output_dir (str): Directory to save plots
    """
    cat_cols = dfs[0].select_dtypes(include=['object', 'category']).columns
    
    for col in cat_cols:
        plt.figure(figsize=(12, 6))
        
        # Calculate and plot value counts for each split
        for df, name in zip(dfs, names):
            value_counts = df[col].value_counts(normalize=True)
            plt.bar(value_counts.index, value_counts.values, alpha=0.3, label=name)
        
        plt.title(f'Distribution of {col} across splits')
        plt.xlabel(col)
        plt.ylabel('Proportion')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'categorical_dist_{col}.png'))
        plt.close()

def analyze_numerical_distributions(dfs: List[pd.DataFrame], names: List[str], output_dir: str):
    """
    Analyze and plot the distribution of numerical variables.
    
    Args:
        dfs (List[pd.DataFrame]): List of dataframes to analyze
        names (List[str]): Names of the dataframes
        output_dir (str): Directory to save plots
    """
    num_cols = dfs[0].select_dtypes(include=[np.number]).columns
    
    for col in num_cols:
        plt.figure(figsize=(12, 6))
        
        # Create KDE plots for each split
        for df, name in zip(dfs, names):
            sns.kdeplot(data=df[col], label=name)
        
        plt.title(f'Distribution of {col} across splits')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'numerical_dist_{col}.png'))
        plt.close()

def analyze_temporal_patterns(dfs: List[pd.DataFrame], names: List[str], output_dir: str):
    """
    Analyze and plot temporal patterns in the data.
    
    Args:
        dfs (List[pd.DataFrame]): List of dataframes to analyze
        names (List[str]): Names of the dataframes
        output_dir (str): Directory to save plots
    """
    # Analyze hourly patterns
    plt.figure(figsize=(12, 6))
    for df, name in zip(dfs, names):
        hourly_mean = df.groupby(df['timestamp'].dt.hour)['occupancy'].mean()
        plt.plot(hourly_mean.index, hourly_mean.values, label=name)
    
    plt.title('Average Occupancy by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Occupancy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_hourly_pattern.png'))
    plt.close()
    
    # Analyze daily patterns
    plt.figure(figsize=(12, 6))
    for df, name in zip(dfs, names):
        daily_mean = df.groupby(df['timestamp'].dt.dayofweek)['occupancy'].mean()
        plt.plot(daily_mean.index, daily_mean.values, label=name)
    
    plt.title('Average Occupancy by Day of Week')
    plt.xlabel('Day of Week (0=Monday)')
    plt.ylabel('Average Occupancy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_daily_pattern.png'))
    plt.close()

def perform_statistical_tests(dfs: List[pd.DataFrame], names: List[str]) -> pd.DataFrame:
    """
    Perform statistical tests to compare distributions across splits.
    
    Args:
        dfs (List[pd.DataFrame]): List of dataframes to analyze
        names (List[str]): Names of the dataframes
        
    Returns:
        pd.DataFrame: DataFrame containing test results
    """
    num_cols = dfs[0].select_dtypes(include=[np.number]).columns
    test_results = []
    
    for col in num_cols:
        # Perform Kolmogorov-Smirnov test between splits
        ks_stat_train_val, p_val_train_val = stats.ks_2samp(dfs[0][col], dfs[1][col])
        ks_stat_train_test, p_val_train_test = stats.ks_2samp(dfs[0][col], dfs[2][col])
        
        test_results.append({
            'Feature': col,
            'KS_Stat_Train_Val': ks_stat_train_val,
            'P_Value_Train_Val': p_val_train_val,
            'KS_Stat_Train_Test': ks_stat_train_test,
            'P_Value_Train_Test': p_val_train_test
        })
    
    return pd.DataFrame(test_results)

def main(splits_dir: str, output_dir: str):
    """
    Main function to analyze the distribution of features across splits.
    
    Args:
        splits_dir (str): Directory containing the splits
        output_dir (str): Directory to save analysis results
    """
    logger.info("Starting split analysis...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the splits
    train_df, val_df, test_df = load_splits(splits_dir)
    dfs = [train_df, val_df, test_df]
    names = ['Train', 'Validation', 'Test']
    
    # Calculate basic statistics
    logger.info("Calculating basic statistics...")
    stats_df = calculate_basic_stats(dfs, names)
    stats_df.to_csv(os.path.join(output_dir, 'basic_statistics.csv'))
    
    # Analyze categorical distributions
    logger.info("Analyzing categorical distributions...")
    analyze_categorical_distributions(dfs, names, output_dir)
    
    # Analyze numerical distributions
    logger.info("Analyzing numerical distributions...")
    analyze_numerical_distributions(dfs, names, output_dir)
    
    # Analyze temporal patterns
    logger.info("Analyzing temporal patterns...")
    analyze_temporal_patterns(dfs, names, output_dir)
    
    # Perform statistical tests
    logger.info("Performing statistical tests...")
    test_results = perform_statistical_tests(dfs, names)
    test_results.to_csv(os.path.join(output_dir, 'statistical_tests.csv'), index=False)
    
    logger.info("Analysis completed successfully")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze the distribution of features across data splits')
    parser.add_argument('--splits-dir', type=str, required=True, help='Directory containing the splits')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    main(args.splits_dir, args.output_dir) 