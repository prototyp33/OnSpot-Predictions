#!/usr/bin/env python
"""
Main execution script for feature validation pipeline.

This script serves as the entry point for running the entire validation process,
integrating all components (ValidationConfig, DistributionAnalyzer, ReportGenerator)
to analyze and report on feature distributions across data splits.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
import sys
import json
from typing import Dict, List, Optional, Tuple
import yaml
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from validation import (
    ValidationConfig,
    DistributionAnalyzer,
    ReportGenerator
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_logging(config: ValidationConfig) -> None:
    """
    Set up logging configuration based on settings.
    
    Args:
        config: Validation configuration object
    """
    log_config = config.config.get('logging', {})
    log_file = Path(config.get_output_dir()) / 'logs' / f'validation_{datetime.now():%Y%m%d_%H%M%S}.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter(log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    )
    
    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter('%(levelname)s: %(message)s')
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_config.get('level', 'INFO'))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

def load_data_splits(config: ValidationConfig, columns: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Load all dataset splits based on configuration.
    
    Args:
        config: Validation configuration object
        columns: Optional list of columns to load (memory optimization)
        
    Returns:
        Dictionary mapping split names to DataFrames
    """
    logger.info("Loading data splits...")
    splits_dir = Path(config.config['data']['splits_dir'])
    
    data_splits = {}
    expected_splits = ['reference', 'comparison']
    
    for split in expected_splits:
        split_path = splits_dir / f'{split}.csv'
        try:
            # Load only required columns if specified
            df = pd.read_csv(split_path, usecols=columns) if columns else pd.read_csv(split_path)
            logger.info(f"Loaded {split} data: {df.shape}")
            data_splits[split] = df
        except Exception as e:
            logger.error(f"Failed to load {split} data from {split_path}: {str(e)}")
            raise
    
    # Validate splits have same columns
    ref_cols = set(data_splits['reference'].columns)
    for split_name, split_df in data_splits.items():
        if set(split_df.columns) != ref_cols:
            extra = set(split_df.columns) - ref_cols
            missing = ref_cols - set(split_df.columns)
            error_msg = f"Column mismatch in {split_name} split."
            if extra:
                error_msg += f"\nExtra columns: {extra}"
            if missing:
                error_msg += f"\nMissing columns: {missing}"
            raise ValueError(error_msg)
    
    return data_splits

def analyze_feature_parallel(feature: str, data_splits: Dict[str, pd.DataFrame], analyzer: DistributionAnalyzer) -> Tuple[str, dict]:
    """Analyze a single feature in parallel."""
    try:
        distributions = {
            name: split[feature] for name, split in data_splits.items()
        }
        result = analyzer.analyze_feature(distributions, feature)
        return feature, result
    except Exception as e:
        logger.error(f"Error analyzing feature {feature}: {str(e)}")
        return feature, None

def validate_features(
    config_path: str,
    output_format: str = 'html',
    generate_feature_reports: bool = True,
    feature_subset: Optional[List[str]] = None,
    n_jobs: int = -1,
    skip_visualizations: bool = False
) -> None:
    """
    Run the complete feature validation pipeline.
    
    Args:
        config_path: Path to validation configuration file
        output_format: Format for reports ('html' or 'markdown')
        generate_feature_reports: Whether to generate individual feature reports
        feature_subset: Optional list of features to analyze (if None, analyzes all)
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        skip_visualizations: Whether to skip generating visualizations during analysis
    """
    try:
        # 1. Initialize configuration
        logger.info(f"Initializing validation with config from: {config_path}")
        config = ValidationConfig(config_path)
        
        if skip_visualizations:
            # Temporarily disable visualizations in config
            config.config['analysis']['visualization']['enabled'] = False
        
        # Set up logging
        setup_logging(config)
        
        # 2. Determine features to analyze
        if feature_subset is None:
            # Read CSV header only to get column names
            reference_path = Path(config.config['data']['splits_dir']) / 'reference.csv'
            feature_subset = pd.read_csv(reference_path, nrows=0).columns.tolist()
        
        # 3. Load data splits with only required columns
        data_splits = load_data_splits(config, columns=feature_subset)
        
        # 4. Initialize analyzer
        logger.info("Initializing DistributionAnalyzer...")
        analyzer = DistributionAnalyzer(config)
        
        # 5. Run parallel analysis
        logger.info("Running batch analysis...")
        n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        
        with mp.Pool(processes=n_jobs) as pool:
            analyze_func = partial(analyze_feature_parallel, data_splits=data_splits, analyzer=analyzer)
            results = list(tqdm(
                pool.imap(analyze_func, feature_subset),
                total=len(feature_subset),
                desc="Analyzing features"
            ))
        
        # Convert results to dictionary
        analysis_results = {
            feature: result for feature, result in results if result is not None
        }
        
        # 6. Save raw analysis results
        results_dir = Path(config.get_output_dir()) / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = results_dir / f'analysis_results_{timestamp}.json'
        
        logger.info(f"Saving analysis results to: {results_path}")
        analyzer.save_analysis_results(analysis_results, results_path)
        
        # 7. Generate reports
        logger.info("Initializing ReportGenerator...")
        reporter = ReportGenerator(config)
        
        logger.info(f"Generating summary report in {output_format} format...")
        summary_report = reporter.generate_summary_report(
            results=analysis_results,
            output_format=output_format
        )
        
        summary_path = (
            Path(config.get_output_dir()) / 
            'reports' / 
            f'summary_report_{timestamp}.{output_format}'
        )
        
        logger.info(f"Exporting summary report to: {summary_path}")
        reporter.export_report(
            report_content=summary_report,
            output_path=summary_path,
            report_type='summary'
        )
        
        if generate_feature_reports:
            logger.info("Generating individual feature reports...")
            for feature, result in tqdm(analysis_results.items(), desc="Generating feature reports"):
                report = reporter.generate_feature_report(
                    feature=feature,
                    result=result,
                    output_format=output_format
                )
                
                report_path = (
                    Path(config.get_output_dir()) / 
                    'reports' / 
                    'features' /
                    f'{feature}_{timestamp}.{output_format}'
                )
                
                reporter.export_report(
                    report_content=report,
                    output_path=report_path,
                    report_type='feature'
                )
                
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run feature validation pipeline to analyze distribution shifts."
    )
    
    parser.add_argument(
        "config_path",
        help="Path to validation configuration file"
    )
    
    parser.add_argument(
        "--output-format",
        choices=['html', 'markdown'],
        default='html',
        help="Output format for reports (default: html)"
    )
    
    parser.add_argument(
        "--skip-feature-reports",
        action="store_true",
        help="Skip generating individual feature reports"
    )
    
    parser.add_argument(
        "--features",
        nargs='+',
        help="Optional list of specific features to analyze"
    )
    
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 for all CPUs)"
    )
    
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip generating visualizations during analysis"
    )
    
    args = parser.parse_args()
    
    validate_features(
        config_path=args.config_path,
        output_format=args.output_format,
        generate_feature_reports=not args.skip_feature_reports,
        feature_subset=args.features,
        n_jobs=args.jobs,
        skip_visualizations=args.skip_viz
    )

if __name__ == "__main__":
    main() 