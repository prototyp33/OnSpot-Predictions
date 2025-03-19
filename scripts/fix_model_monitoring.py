#!/usr/bin/env python
"""
Fix for model_monitoring.py file.
Updates the set_baseline method to handle non-numeric columns.
"""

import os
import sys
import logging
from pathlib import Path
import json
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FixMonitoring")

def fix_model_monitoring():
    """Fix the model_monitoring.py file."""
    logger.info("Fixing model_monitoring.py")
    
    # Path to the file
    file_path = "scripts/model_monitoring.py"
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the set_baseline method
    start_marker = "    def set_baseline(self, baseline_data: pd.DataFrame):"
    end_marker = "    def calculate_drift_metrics"
    
    if start_marker not in content:
        logger.error(f"Could not find start marker: {start_marker}")
        return False
    
    if end_marker not in content:
        logger.error(f"Could not find end marker: {end_marker}")
        return False
    
    # Extract the part before and after the method
    start_index = content.find(start_marker)
    end_index = content.find(end_marker)
    
    if start_index == -1 or end_index == -1:
        logger.error("Could not find method boundaries")
        return False
    
    before_method = content[:start_index]
    after_method = content[end_index:]
    
    # New method implementation
    new_method = """    def set_baseline(self, baseline_data: pd.DataFrame):
        \"\"\"Set baseline statistics for drift comparison.\"\"\"
        # Filter out non-numeric columns
        numeric_cols = baseline_data.select_dtypes(include=['number']).columns
        categorical_cols = baseline_data.select_dtypes(exclude=['number']).columns
        
        self.logger.info(f"Using {len(numeric_cols)} numeric columns for baseline statistics")
        if len(categorical_cols) > 0:
            self.logger.info(f"Excluding {len(categorical_cols)} non-numeric columns: {list(categorical_cols)}")
        
        # Compute statistics only on numeric columns
        numeric_data = baseline_data[numeric_cols]
        
        self.baseline_stats = {
            'mean': numeric_data.mean(),
            'std': numeric_data.std(),
            'quantiles': numeric_data.quantile([0.25, 0.5, 0.75]),
            'timestamp': datetime.now().isoformat(),
            'categorical_cols': list(categorical_cols),
            'numeric_cols': list(numeric_cols)
        }
        
        # For categorical columns, store value counts
        cat_stats = {}
        for col in categorical_cols:
            try:
                # Get value counts as percentages
                value_counts = baseline_data[col].value_counts(normalize=True).to_dict()
                cat_stats[col] = value_counts
            except Exception as e:
                self.logger.warning(f"Could not compute value counts for column {col}: {e}")
        
        self.baseline_stats['categorical_stats'] = cat_stats
        
        # Save baseline stats to file
        Path("data/monitoring").mkdir(parents=True, exist_ok=True)
        with open('data/monitoring/baseline_stats.json', 'w') as f:
            # Convert any numpy types to Python native types for JSON serialization
            stats_for_json = {
                'mean': self.baseline_stats['mean'].to_dict(),
                'std': self.baseline_stats['std'].to_dict(),
                'quantiles': self.baseline_stats['quantiles'].to_dict(),
                'timestamp': self.baseline_stats['timestamp'],
                'categorical_cols': list(map(str, self.baseline_stats['categorical_cols'])),
                'numeric_cols': list(map(str, self.baseline_stats['numeric_cols'])),
                'categorical_stats': self.baseline_stats['categorical_stats']
            }
            json.dump(stats_for_json, f, indent=2)
        
        self.logger.info("Baseline statistics updated")

"""
    
    # Combine the parts
    new_content = before_method + new_method + after_method
    
    # Write back to the file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    logger.info(f"Fixed {file_path}")
    return True

def fix_calculate_drift_metrics():
    """Fix the calculate_drift_metrics method."""
    logger.info("Fixing calculate_drift_metrics method")
    
    # Path to the file
    file_path = "scripts/model_monitoring.py"
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the calculate_drift_metrics method
    start_marker = "    def calculate_drift_metrics(self, current_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:"
    end_marker = "    def detect_drift(self, current_data: pd.DataFrame) -> Tuple[bool, Dict[str, Dict[str, float]]]:"
    
    if start_marker not in content:
        logger.error(f"Could not find start marker: {start_marker}")
        return False
    
    if end_marker not in content:
        logger.error(f"Could not find end marker: {end_marker}")
        return False
    
    # Extract the part before and after the method
    start_index = content.find(start_marker)
    end_index = content.find(end_marker)
    
    if start_index == -1 or end_index == -1:
        logger.error("Could not find method boundaries")
        return False
    
    before_method = content[:start_index]
    after_method = content[end_index:]
    
    # New method implementation
    new_method = """    def calculate_drift_metrics(self, current_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        \"\"\"Calculate drift metrics for each feature.\"\"\"
        drift_metrics = {}
        
        # Process only numeric columns that are in both datasets
        if 'numeric_cols' in self.baseline_stats:
            numeric_cols = set(self.baseline_stats['numeric_cols']).intersection(
                current_data.select_dtypes(include=['number']).columns
            )
            self.logger.info(f"Calculating drift for {len(numeric_cols)} numeric columns")
        else:
            # Legacy compatibility
            numeric_cols = current_data.select_dtypes(include=['number']).columns
        
        for column in numeric_cols:
            if column in self.baseline_stats['mean']:
                try:
                    baseline_values = self.baseline_stats['mean'][column]
                    current_values = current_data[column].mean()
                    
                    # Kolmogorov-Smirnov test
                    ks_statistic, p_value = ks_2samp(
                        current_data[column].dropna(),
                        pd.Series([baseline_values], name=column)
                    )
                    
                    # Calculate relative differences
                    mean_diff = abs(current_values - baseline_values) / (abs(baseline_values) if baseline_values != 0 else 1)
                    
                    if column in self.baseline_stats['std']:
                        std_baseline = self.baseline_stats['std'][column]
                        std_current = current_data[column].std()
                        std_diff = abs(std_current - std_baseline) / (abs(std_baseline) if std_baseline != 0 else 1)
                    else:
                        std_diff = 0
                    
                    drift_metrics[column] = {
                        'ks_statistic': float(ks_statistic),
                        'p_value': float(p_value),
                        'mean_difference': float(mean_diff),
                        'std_difference': float(std_diff)
                    }
                except Exception as e:
                    self.logger.warning(f"Error calculating drift metrics for column {column}: {e}")
        
        # Process categorical columns
        if 'categorical_cols' in self.baseline_stats:
            cat_cols = set(self.baseline_stats['categorical_cols']).intersection(
                current_data.select_dtypes(exclude=['number']).columns
            )
            
            if cat_cols:
                self.logger.info(f"Calculating distribution changes for {len(cat_cols)} categorical columns")
            
            for column in cat_cols:
                try:
                    if column in self.baseline_stats.get('categorical_stats', {}):
                        baseline_dist = self.baseline_stats['categorical_stats'][column]
                        current_dist = current_data[column].value_counts(normalize=True).to_dict()
                        
                        # Calculate distribution difference
                        total_diff = 0
                        for category, baseline_freq in baseline_dist.items():
                            current_freq = current_dist.get(category, 0)
                            total_diff += abs(baseline_freq - current_freq)
                        
                        # Add new categories
                        for category in current_dist:
                            if category not in baseline_dist:
                                total_diff += current_dist[category]
                        
                        # Normalize to [0, 1]
                        total_diff = min(1.0, total_diff / 2.0)
                        
                        drift_metrics[column] = {
                            'distribution_difference': float(total_diff),
                            'new_categories': list(set(current_dist.keys()) - set(baseline_dist.keys())),
                            'missing_categories': list(set(baseline_dist.keys()) - set(current_dist.keys()))
                        }
                except Exception as e:
                    self.logger.warning(f"Error calculating categorical drift for column {column}: {e}")
        
        return drift_metrics

"""
    
    # Combine the parts
    new_content = before_method + new_method + after_method
    
    # Write back to the file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    logger.info(f"Fixed calculate_drift_metrics method in {file_path}")
    return True

def fix_detect_drift():
    """Fix the detect_drift method."""
    logger.info("Fixing detect_drift method")
    
    # Path to the file
    file_path = "scripts/model_monitoring.py"
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the detect_drift method
    start_marker = "    def detect_drift(self, current_data: pd.DataFrame) -> Tuple[bool, Dict[str, Dict[str, float]]]:"
    end_marker = "class PerformanceMonitor:"
    
    if start_marker not in content:
        logger.error(f"Could not find start marker: {start_marker}")
        return False
    
    if end_marker not in content:
        logger.error(f"Could not find end marker: {end_marker}")
        return False
    
    # Extract the part before and after the method
    start_index = content.find(start_marker)
    end_index = content.find(end_marker)
    
    if start_index == -1 or end_index == -1:
        logger.error("Could not find method boundaries")
        return False
    
    before_method = content[:start_index]
    after_method = content[end_index:]
    
    # New method implementation
    new_method = """    def detect_drift(self, current_data: pd.DataFrame) -> Tuple[bool, Dict[str, Dict[str, float]]]:
        \"\"\"Detect if there is significant drift in the current data.\"\"\"
        drift_metrics = self.calculate_drift_metrics(current_data)
        drift_detected = False
        thresholds = self.config['drift_thresholds']
        
        # Check for drift in numeric features
        for feature, metrics in drift_metrics.items():
            if 'ks_statistic' in metrics:  # Numeric feature
                if (metrics['ks_statistic'] > thresholds['ks_statistic'] or
                    metrics['mean_difference'] > thresholds['mean_difference'] or
                    metrics['std_difference'] > thresholds['std_difference']):
                    drift_detected = True
                    self.logger.warning(f"Drift detected in numeric feature {feature}")
                    self.logger.info(f"  KS statistic: {metrics['ks_statistic']:.3f} (threshold: {thresholds['ks_statistic']:.3f})")
                    self.logger.info(f"  Mean difference: {metrics['mean_difference']:.1%} (threshold: {thresholds['mean_difference']:.1%})")
                    self.logger.info(f"  Std difference: {metrics['std_difference']:.1%} (threshold: {thresholds['std_difference']:.1%})")
            elif 'distribution_difference' in metrics:  # Categorical feature
                # Use a default threshold if not specified
                cat_threshold = thresholds.get('distribution_difference', 0.2)
                if metrics['distribution_difference'] > cat_threshold:
                    drift_detected = True
                    self.logger.warning(f"Drift detected in categorical feature {feature}")
                    self.logger.info(f"  Distribution difference: {metrics['distribution_difference']:.1%}")
                    if metrics['new_categories']:
                        self.logger.info(f"  New categories: {metrics['new_categories']}")
                    if metrics['missing_categories']:
                        self.logger.info(f"  Missing categories: {metrics['missing_categories']}")
                
        return drift_detected, drift_metrics

"""
    
    # Combine the parts
    new_content = before_method + new_method + after_method
    
    # Write back to the file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    logger.info(f"Fixed detect_drift method in {file_path}")
    return True

def fix_monitoring_init():
    """Fix the DataDriftMonitor and PerformanceMonitor to accept dictionaries."""
    logger.info("Fixing monitor initialization methods")
    
    # Path to the file
    file_path = "scripts/model_monitoring.py"
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the DataDriftMonitor __init__ method
    start_marker = "class DataDriftMonitor:"
    end_marker = "    def set_baseline"
    
    if start_marker not in content:
        logger.error(f"Could not find start marker: {start_marker}")
        return False
    
    if end_marker not in content:
        logger.error(f"Could not find end marker: {end_marker}")
        return False
    
    # Extract the part before and after the method
    start_index = content.find(start_marker)
    first_method_index = content.find(end_marker, start_index)
    
    # Skip to after the class declaration
    class_end_index = content.find("\n", start_index) + 1
    init_start = content.find("    def __init__", class_end_index)
    
    before_init = content[:init_start]
    after_init = content[first_method_index:]
    
    # New __init__ method implementation
    new_init = '''    def __init__(self, config=None):
        """Monitor data drift in model inputs."""
        self.logger = logging.getLogger('DataDriftMonitor')
        
        # Handle both file path and direct config dictionary
        if config is None:
            config = 'config/monitoring_config.json'
            
        if isinstance(config, str):
            self.config = self._load_config(config)
        else:
            self.logger.info("Using provided configuration dictionary")
            self.config = config
            
            # Set default thresholds if not provided
            if 'drift_thresholds' not in self.config:
                self.config['drift_thresholds'] = {
                    'ks_statistic': 0.1,
                    'mean_difference': 0.2,
                    'std_difference': 0.2,
                    'distribution_difference': 0.3
                }
                
        self.baseline_stats = {}
        
        # Create monitoring directory
        Path('data/monitoring').mkdir(parents=True, exist_ok=True)
        
        # Load baseline stats if available
        baseline_path = 'data/monitoring/baseline_stats.json'
        if os.path.exists(baseline_path):
            try:
                with open(baseline_path, 'r') as f:
                    stats = json.load(f)
                    self.logger.info(f"Loaded baseline statistics from {baseline_path}")
                    
                    # Convert dictionary stats back to pandas Series
                    self.baseline_stats = {
                        'mean': pd.Series(stats['mean']),
                        'std': pd.Series(stats['std']),
                        'quantiles': pd.DataFrame(stats['quantiles']),
                        'timestamp': stats['timestamp'],
                        'categorical_cols': stats['categorical_cols'],
                        'numeric_cols': stats['numeric_cols']
                    }
                    
                    if 'categorical_stats' in stats:
                        self.baseline_stats['categorical_stats'] = stats['categorical_stats']
            except Exception as e:
                self.logger.warning(f"Failed to load baseline stats: {e}")
'''
    
    # Combine the parts
    new_content = before_init + new_init + after_init
    
    # Write back to the file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    # Now fix the PerformanceMonitor __init__ method
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the PerformanceMonitor __init__ method
    start_marker = "class PerformanceMonitor:"
    end_marker = "    def evaluate_model"
    
    if start_marker not in content:
        logger.error(f"Could not find start marker: {start_marker}")
        return False
    
    if end_marker not in content:
        logger.error(f"Could not find end marker: {end_marker}")
        return False
    
    # Extract the part before and after the method
    start_index = content.find(start_marker)
    first_method_index = content.find(end_marker, start_index)
    
    # Skip to after the class declaration
    class_end_index = content.find("\n", start_index) + 1
    init_start = content.find("    def __init__", class_end_index)
    
    before_init = content[:init_start]
    after_init = content[first_method_index:]
    
    # New __init__ method implementation
    new_init = '''    def __init__(self, config=None):
        """Monitor model performance metrics."""
        self.logger = logging.getLogger('PerformanceMonitor')
        
        # Handle both file path and direct config dictionary
        if config is None:
            config = 'config/monitoring_config.json'
            
        if isinstance(config, str):
            self.config = self._load_config(config)
        else:
            self.logger.info("Using provided configuration dictionary")
            self.config = config
            
            # Set default thresholds if not provided
            if 'performance_thresholds' not in self.config:
                self.config['performance_thresholds'] = {
                    'accuracy': 0.8,
                    'precision': 0.7,
                    'recall': 0.7,
                    'f1': 0.75
                }
        
        # Create monitoring directory
        Path('data/monitoring').mkdir(parents=True, exist_ok=True)
'''
    
    # Combine the parts
    new_content = before_init + new_init + after_init
    
    # Write back to the file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    logger.info(f"Fixed monitor initialization methods in {file_path}")
    return True

def main():
    """Run the fixes."""
    logger.info("Starting fixes for model_monitoring.py")
    
    # Fix the init methods
    if fix_monitoring_init():
        logger.info("Successfully fixed monitor initialization methods")
    else:
        logger.error("Failed to fix monitor initialization methods")
    
    # Fix the set_baseline method
    if fix_model_monitoring():
        logger.info("Successfully fixed set_baseline method")
    else:
        logger.error("Failed to fix set_baseline method")
    
    # Fix the calculate_drift_metrics method
    if fix_calculate_drift_metrics():
        logger.info("Successfully fixed calculate_drift_metrics method")
    else:
        logger.error("Failed to fix calculate_drift_metrics method")
    
    # Fix the detect_drift method
    if fix_detect_drift():
        logger.info("Successfully fixed detect_drift method")
    else:
        logger.error("Failed to fix detect_drift method")
    
    logger.info("All fixes applied")

if __name__ == "__main__":
    main() 