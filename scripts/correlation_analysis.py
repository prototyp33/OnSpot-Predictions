#!/usr/bin/env python
"""
Script for performing comprehensive correlation analysis on parking occupancy data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import logging
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory
output_dir = "correlation_analysis_results"
os.makedirs(output_dir, exist_ok=True)

def load_data(file_path):
    """Load and prepare the dataset for correlation analysis."""
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract time components
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'] >= 5
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    # Create cyclical features for time
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    logger.info(f"Dataset loaded with shape: {df.shape}")
    return df

def identify_variable_types(df):
    """Identify numerical and categorical variables in the dataset."""
    # Exclude target variable and timestamp
    exclude_cols = ['occupancy', 'timestamp', 'date']
    
    # Identify numerical and categorical columns
    numerical_cols = []
    categorical_cols = []
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        
        if np.issubdtype(df[col].dtype, np.number):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
    
    logger.info(f"Identified {len(numerical_cols)} numerical variables: {numerical_cols}")
    logger.info(f"Identified {len(categorical_cols)} categorical variables: {categorical_cols}")
    
    return numerical_cols, categorical_cols

def pearson_correlation_analysis(df, numerical_cols):
    """Perform Pearson correlation analysis for numerical variables."""
    logger.info("Performing Pearson correlation analysis...")
    
    # Calculate Pearson correlation with occupancy
    pearson_corr = df[['occupancy'] + numerical_cols].corr(method='pearson')['occupancy'].drop('occupancy')
    
    # Sort by absolute correlation
    pearson_corr = pearson_corr.reindex(pearson_corr.abs().sort_values(ascending=False).index)
    
    # Create a DataFrame for the results
    pearson_results = pd.DataFrame({
        'variable': pearson_corr.index,
        'pearson_correlation': pearson_corr.values,
        'abs_correlation': pearson_corr.abs().values
    }).sort_values('abs_correlation', ascending=False)
    
    # Save results
    pearson_results.to_csv(os.path.join(output_dir, 'pearson_correlation.csv'), index=False)
    
    # Plot top correlations
    plt.figure(figsize=(12, 8))
    top_vars = pearson_results.head(15)['variable'].tolist()
    
    # Create correlation heatmap for top variables
    top_corr = df[['occupancy'] + top_vars].corr(method='pearson')
    mask = np.triu(np.ones_like(top_corr, dtype=bool))
    
    sns.heatmap(top_corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                vmin=-1, vmax=1, center=0, linewidths=0.5)
    plt.title('Pearson Correlation Heatmap (Top Variables)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pearson_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot correlation bar chart
    plt.figure(figsize=(12, 8))
    sns.barplot(x='pearson_correlation', y='variable', data=pearson_results.head(15))
    plt.title('Pearson Correlation with Occupancy')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Variable')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pearson_barchart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return pearson_results

def spearman_correlation_analysis(df, numerical_cols):
    """Perform Spearman correlation analysis for numerical variables."""
    logger.info("Performing Spearman correlation analysis...")
    
    # Calculate Spearman correlation with occupancy
    spearman_corr = df[['occupancy'] + numerical_cols].corr(method='spearman')['occupancy'].drop('occupancy')
    
    # Sort by absolute correlation
    spearman_corr = spearman_corr.reindex(spearman_corr.abs().sort_values(ascending=False).index)
    
    # Create a DataFrame for the results
    spearman_results = pd.DataFrame({
        'variable': spearman_corr.index,
        'spearman_correlation': spearman_corr.values,
        'abs_correlation': spearman_corr.abs().values
    }).sort_values('abs_correlation', ascending=False)
    
    # Save results
    spearman_results.to_csv(os.path.join(output_dir, 'spearman_correlation.csv'), index=False)
    
    # Plot correlation bar chart
    plt.figure(figsize=(12, 8))
    sns.barplot(x='spearman_correlation', y='variable', data=spearman_results.head(15))
    plt.title('Spearman Correlation with Occupancy')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Variable')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spearman_barchart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return spearman_results

def kendall_correlation_analysis(df, numerical_cols):
    """Perform Kendall correlation analysis for numerical variables."""
    logger.info("Performing Kendall correlation analysis...")
    
    # Calculate Kendall correlation with occupancy
    kendall_corr = df[['occupancy'] + numerical_cols].corr(method='kendall')['occupancy'].drop('occupancy')
    
    # Sort by absolute correlation
    kendall_corr = kendall_corr.reindex(kendall_corr.abs().sort_values(ascending=False).index)
    
    # Create a DataFrame for the results
    kendall_results = pd.DataFrame({
        'variable': kendall_corr.index,
        'kendall_correlation': kendall_corr.values,
        'abs_correlation': kendall_corr.abs().values
    }).sort_values('abs_correlation', ascending=False)
    
    # Save results
    kendall_results.to_csv(os.path.join(output_dir, 'kendall_correlation.csv'), index=False)
    
    # Plot correlation bar chart
    plt.figure(figsize=(12, 8))
    sns.barplot(x='kendall_correlation', y='variable', data=kendall_results.head(15))
    plt.title('Kendall Correlation with Occupancy')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Variable')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kendall_barchart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return kendall_results

def categorical_correlation_analysis(df, categorical_cols):
    """Analyze correlation between categorical variables and occupancy."""
    logger.info("Analyzing categorical variables...")
    
    categorical_results = []
    
    for col in categorical_cols:
        # Calculate mean occupancy for each category
        category_means = df.groupby(col)['occupancy'].mean().sort_values(ascending=False)
        
        # Calculate standard deviation for each category
        category_stds = df.groupby(col)['occupancy'].std()
        
        # Calculate ANOVA to test if means are significantly different
        categories = df[col].unique()
        if len(categories) > 1:  # Only perform ANOVA if there are at least 2 categories
            groups = [df[df[col] == cat]['occupancy'].values for cat in categories]
            try:
                f_stat, p_value = stats.f_oneway(*groups)
            except:
                f_stat, p_value = np.nan, np.nan
        else:
            f_stat, p_value = np.nan, np.nan
        
        # Calculate eta squared (effect size)
        if not np.isnan(f_stat) and len(df) > 0:
            eta_squared = (f_stat * (len(categories) - 1)) / (f_stat * (len(categories) - 1) + (len(df) - len(categories)))
        else:
            eta_squared = np.nan
        
        # Add to results
        categorical_results.append({
            'variable': col,
            'num_categories': len(categories),
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'max_diff': category_means.max() - category_means.min() if len(category_means) > 0 else 0
        })
        
        # Plot mean occupancy by category
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(category_means)), category_means.values, yerr=category_stds[category_means.index].values, capsize=5)
        plt.xticks(range(len(category_means)), category_means.index, rotation=45)
        plt.title(f'Mean Occupancy by {col}')
        plt.xlabel(col)
        plt.ylabel('Mean Occupancy')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'categorical_{col}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create DataFrame for results
    categorical_results_df = pd.DataFrame(categorical_results).sort_values('eta_squared', ascending=False)
    
    # Save results
    categorical_results_df.to_csv(os.path.join(output_dir, 'categorical_correlation.csv'), index=False)
    
    # Plot eta squared for categorical variables
    if len(categorical_results_df) > 0:
        plt.figure(figsize=(12, 8))
        sns.barplot(x='eta_squared', y='variable', data=categorical_results_df)
        plt.title('Effect Size (Eta Squared) for Categorical Variables')
        plt.xlabel('Eta Squared')
        plt.ylabel('Variable')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'categorical_eta_squared.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    return categorical_results_df

def compare_correlation_methods(pearson_results, spearman_results, kendall_results):
    """Compare results from different correlation methods."""
    logger.info("Comparing correlation methods...")
    
    # Merge results
    comparison = pd.merge(
        pearson_results[['variable', 'pearson_correlation']], 
        spearman_results[['variable', 'spearman_correlation']], 
        on='variable'
    )
    comparison = pd.merge(
        comparison, 
        kendall_results[['variable', 'kendall_correlation']], 
        on='variable'
    )
    
    # Calculate absolute values
    comparison['abs_pearson'] = comparison['pearson_correlation'].abs()
    comparison['abs_spearman'] = comparison['spearman_correlation'].abs()
    comparison['abs_kendall'] = comparison['kendall_correlation'].abs()
    
    # Calculate average absolute correlation
    comparison['avg_abs_corr'] = (comparison['abs_pearson'] + comparison['abs_spearman'] + comparison['abs_kendall']) / 3
    
    # Sort by average absolute correlation
    comparison = comparison.sort_values('avg_abs_corr', ascending=False)
    
    # Save comparison
    comparison.to_csv(os.path.join(output_dir, 'correlation_comparison.csv'), index=False)
    
    # Plot comparison for top variables
    top_vars = comparison.head(10)['variable'].tolist()
    
    # Reshape data for plotting
    plot_data = []
    for _, row in comparison[comparison['variable'].isin(top_vars)].iterrows():
        plot_data.append({'variable': row['variable'], 'method': 'Pearson', 'correlation': abs(row['pearson_correlation'])})
        plot_data.append({'variable': row['variable'], 'method': 'Spearman', 'correlation': abs(row['spearman_correlation'])})
        plot_data.append({'variable': row['variable'], 'method': 'Kendall', 'correlation': abs(row['kendall_correlation'])})
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create grouped bar chart
    plt.figure(figsize=(14, 8))
    sns.barplot(x='variable', y='correlation', hue='method', data=plot_df)
    plt.title('Comparison of Correlation Methods (Absolute Values)')
    plt.xlabel('Variable')
    plt.ylabel('Absolute Correlation')
    plt.xticks(rotation=45)
    plt.legend(title='Method')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_methods_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return comparison

def analyze_nonlinear_relationships(df, numerical_cols):
    """Analyze potential nonlinear relationships using scatter plots and binning."""
    logger.info("Analyzing nonlinear relationships...")
    
    nonlinear_dir = os.path.join(output_dir, 'nonlinear_analysis')
    os.makedirs(nonlinear_dir, exist_ok=True)
    
    # Get top numerical variables from Spearman correlation
    spearman_corr = df[['occupancy'] + numerical_cols].corr(method='spearman')['occupancy'].drop('occupancy')
    top_vars = spearman_corr.abs().sort_values(ascending=False).head(10).index.tolist()
    
    nonlinear_results = []
    
    for var in top_vars:
        # Create scatter plot with lowess smoothing
        plt.figure(figsize=(10, 6))
        sns.regplot(x=var, y='occupancy', data=df, scatter_kws={'alpha': 0.3}, 
                   line_kws={'color': 'red'}, lowess=True)
        plt.title(f'Relationship between {var} and Occupancy')
        plt.xlabel(var)
        plt.ylabel('Occupancy')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(nonlinear_dir, f'scatter_{var}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Bin the variable and analyze mean occupancy by bin
        try:
            # Create bins
            df[f'{var}_bin'] = pd.qcut(df[var], 10, duplicates='drop')
            
            # Calculate mean and std by bin
            bin_stats = df.groupby(f'{var}_bin')['occupancy'].agg(['mean', 'std', 'count']).reset_index()
            
            # Plot mean occupancy by bin
            plt.figure(figsize=(12, 6))
            plt.errorbar(range(len(bin_stats)), bin_stats['mean'], yerr=bin_stats['std'], fmt='o-', capsize=5)
            plt.xticks(range(len(bin_stats)), [str(x) for x in bin_stats[f'{var}_bin']], rotation=45)
            plt.title(f'Mean Occupancy by {var} Bins')
            plt.xlabel(f'{var} Bins')
            plt.ylabel('Mean Occupancy')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(nonlinear_dir, f'binned_{var}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calculate linearity metrics
            x = np.arange(len(bin_stats))
            y = bin_stats['mean'].values
            
            # Linear correlation between bin index and mean occupancy
            lin_corr, _ = stats.pearsonr(x, y)
            
            # Fit polynomial of degree 2 and calculate R²
            poly_coefs = np.polyfit(x, y, 2)
            p = np.poly1d(poly_coefs)
            y_pred = p(x)
            ss_total = np.sum((y - np.mean(y))**2)
            ss_residual = np.sum((y - y_pred)**2)
            r_squared = 1 - (ss_residual / ss_total)
            
            # Calculate improvement of quadratic fit over linear
            lin_coefs = np.polyfit(x, y, 1)
            p_lin = np.poly1d(lin_coefs)
            y_pred_lin = p_lin(x)
            ss_residual_lin = np.sum((y - y_pred_lin)**2)
            r_squared_lin = 1 - (ss_residual_lin / ss_total)
            
            improvement = r_squared - r_squared_lin
            
            # Add to results
            nonlinear_results.append({
                'variable': var,
                'linear_r_squared': r_squared_lin,
                'quadratic_r_squared': r_squared,
                'improvement': improvement,
                'bin_count': len(bin_stats)
            })
            
            # Remove temporary bin column
            df = df.drop(f'{var}_bin', axis=1)
            
        except Exception as e:
            logger.warning(f"Could not perform binned analysis for {var}: {e}")
    
    # Create DataFrame for results
    nonlinear_results_df = pd.DataFrame(nonlinear_results).sort_values('improvement', ascending=False)
    
    # Save results
    nonlinear_results_df.to_csv(os.path.join(nonlinear_dir, 'nonlinear_analysis.csv'), index=False)
    
    # Plot improvement of quadratic fit
    plt.figure(figsize=(12, 6))
    sns.barplot(x='improvement', y='variable', data=nonlinear_results_df)
    plt.title('Improvement of Quadratic Fit over Linear Fit')
    plt.xlabel('Improvement in R²')
    plt.ylabel('Variable')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(nonlinear_dir, 'quadratic_improvement.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return nonlinear_results_df

def create_summary_report(pearson_results, spearman_results, kendall_results, 
                         categorical_results, comparison, nonlinear_results):
    """Create a summary report of all correlation analyses."""
    logger.info("Creating summary report...")
    
    with open(os.path.join(output_dir, 'correlation_summary.txt'), 'w') as f:
        f.write("=== CORRELATION ANALYSIS SUMMARY ===\n\n")
        
        f.write("TOP 10 VARIABLES BY AVERAGE ABSOLUTE CORRELATION:\n")
        for i, (_, row) in enumerate(comparison.head(10).iterrows(), 1):
            f.write(f"{i}. {row['variable']}: Avg={row['avg_abs_corr']:.4f} (P={row['pearson_correlation']:.4f}, S={row['spearman_correlation']:.4f}, K={row['kendall_correlation']:.4f})\n")
        
        f.write("\nTOP 5 VARIABLES BY PEARSON CORRELATION:\n")
        for i, (_, row) in enumerate(pearson_results.head(5).iterrows(), 1):
            f.write(f"{i}. {row['variable']}: {row['pearson_correlation']:.4f}\n")
        
        f.write("\nTOP 5 VARIABLES BY SPEARMAN CORRELATION:\n")
        for i, (_, row) in enumerate(spearman_results.head(5).iterrows(), 1):
            f.write(f"{i}. {row['variable']}: {row['spearman_correlation']:.4f}\n")
        
        f.write("\nTOP 5 VARIABLES BY KENDALL CORRELATION:\n")
        for i, (_, row) in enumerate(kendall_results.head(5).iterrows(), 1):
            f.write(f"{i}. {row['variable']}: {row['kendall_correlation']:.4f}\n")
        
        if len(categorical_results) > 0:
            f.write("\nTOP CATEGORICAL VARIABLES BY EFFECT SIZE:\n")
            for i, (_, row) in enumerate(categorical_results.head(5).iterrows(), 1):
                f.write(f"{i}. {row['variable']}: Eta²={row['eta_squared']:.4f}, p-value={row['p_value']:.4f}\n")
        
        if nonlinear_results is not None and len(nonlinear_results) > 0:
            f.write("\nTOP VARIABLES WITH NONLINEAR RELATIONSHIPS:\n")
            for i, (_, row) in enumerate(nonlinear_results.head(5).iterrows(), 1):
                f.write(f"{i}. {row['variable']}: Improvement={row['improvement']:.4f}, Quadratic R²={row['quadratic_r_squared']:.4f}, Linear R²={row['linear_r_squared']:.4f}\n")
        
        f.write("\nRECOMMENDATIONS:\n")
        f.write("1. Focus on the top variables by average absolute correlation for feature selection.\n")
        f.write("2. Consider nonlinear transformations for variables with high nonlinear improvement.\n")
        f.write("3. For categorical variables with high effect size, consider one-hot encoding or target encoding.\n")
        f.write("4. Variables with high Spearman but low Pearson correlation may benefit from rank transformation.\n")
        f.write("5. Consider interaction terms between highly correlated variables.\n")

def main(data_path):
    """Main function to run the correlation analysis."""
    logger.info("Starting correlation analysis...")
    
    # Load data
    df = load_data(data_path)
    
    # Identify variable types
    numerical_cols, categorical_cols = identify_variable_types(df)
    
    # Perform correlation analyses
    pearson_results = pearson_correlation_analysis(df, numerical_cols)
    spearman_results = spearman_correlation_analysis(df, numerical_cols)
    kendall_results = kendall_correlation_analysis(df, numerical_cols)
    
    # Analyze categorical variables
    categorical_results = categorical_correlation_analysis(df, categorical_cols)
    
    # Compare correlation methods
    comparison = compare_correlation_methods(pearson_results, spearman_results, kendall_results)
    
    # Analyze nonlinear relationships
    nonlinear_results = analyze_nonlinear_relationships(df, numerical_cols)
    
    # Create summary report
    create_summary_report(pearson_results, spearman_results, kendall_results, 
                         categorical_results, comparison, nonlinear_results)
    
    logger.info(f"Correlation analysis completed. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform correlation analysis on parking occupancy data")
    parser.add_argument("--data", default="data/prepared_data_improved.csv", help="Path to the prepared data file")
    
    args = parser.parse_args()
    main(args.data) 