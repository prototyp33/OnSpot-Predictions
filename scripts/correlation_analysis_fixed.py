import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr

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
        # Create scatter plot with polynomial fit instead of lowess
        plt.figure(figsize=(10, 6))
        
        # Basic scatter plot
        plt.scatter(df[var], df['occupancy'], alpha=0.3)
        
        # Add polynomial fits
        x = df[var].values
        y = df['occupancy'].values
        
        # Sort x and y for plotting smooth curves
        idx = np.argsort(x)
        x_sorted = x[idx]
        y_sorted = y[idx]
        
        # Linear fit
        lin_coefs = np.polyfit(x, y, 1)
        lin_fit = np.poly1d(lin_coefs)
        plt.plot(x_sorted, lin_fit(x_sorted), 'r-', label='Linear fit')
        
        # Quadratic fit
        quad_coefs = np.polyfit(x, y, 2)
        quad_fit = np.poly1d(quad_coefs)
        plt.plot(x_sorted, quad_fit(x_sorted), 'g-', label='Quadratic fit')
        
        plt.title(f'Relationship between {var} and Occupancy')
        plt.xlabel(var)
        plt.ylabel('Occupancy')
        plt.grid(True, alpha=0.3)
        plt.legend()
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
            lin_corr, _ = pearsonr(x, y)
            
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