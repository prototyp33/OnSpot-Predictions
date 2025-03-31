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

def load_data_splits(config: ValidationConfig) -> Dict[str, pd.DataFrame]:
    """
    Load all dataset splits based on configuration.
    
    Args:
        config: Validation configuration object
        
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
            df = pd.read_csv(split_path)
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

def validate_features(
    config_path: str,
    output_format: str = 'html',
    generate_feature_reports: bool = True,
    feature_subset: Optional[List[str]] = None
) -> None:
    """
    Run the complete feature validation pipeline.
    
    Args:
        config_path: Path to validation configuration file
        output_format: Format for reports ('html' or 'markdown')
        generate_feature_reports: Whether to generate individual feature reports
        feature_subset: Optional list of features to analyze (if None, analyzes all)
    """
    try:
        # 1. Initialize configuration
        logger.info(f"Initializing validation with config from: {config_path}")
        config = ValidationConfig(config_path)
        
        # Set up logging
        setup_logging(config)
        
        # 2. Load data splits
        data_splits = load_data_splits(config)
        
        # 3. Initialize analyzer
        logger.info("Initializing DistributionAnalyzer...")
        analyzer = DistributionAnalyzer(config)
        
        # Get features to analyze
        all_features = data_splits['reference'].columns.tolist()
        features_to_analyze = (
            feature_subset if feature_subset is not None
            else all_features
        )
        
        # Validate feature subset
        if feature_subset:
            invalid_features = set(feature_subset) - set(all_features)
            if invalid_features:
                raise ValueError(f"Invalid features specified: {invalid_features}")
        
        # 4. Run analysis with progress bar
        logger.info("Running batch analysis...")
        analysis_results = {}
        
        for feature in tqdm(features_to_analyze, desc="Analyzing features"):
            try:
                # Extract feature distributions
                distributions = {
                    name: split[feature] for name, split in data_splits.items()
                }
                
                # Analyze feature
                result = analyzer.analyze_feature(distributions, feature)
                analysis_results[feature] = result
                
            except Exception as e:
                logger.error(f"Error analyzing feature {feature}: {str(e)}")
                continue
        
        # 5. Save raw analysis results
        results_dir = Path(config.get_output_dir()) / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = results_dir / f'analysis_results_{timestamp}.json'
        
        logger.info(f"Saving analysis results to: {results_path}")
        analyzer.save_analysis_results(analysis_results, results_path)
        
        # 6. Initialize report generator
        logger.info("Initializing ReportGenerator...")
        reporter = ReportGenerator(config)
        
        # 7. Generate and export summary report
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
        
        # 8. Optionally generate individual feature reports
        if generate_feature_reports:
            logger.info("Generating individual feature reports...")
            feature_reports_dir = Path(config.get_output_dir()) / 'reports' / 'features'
            feature_reports_dir.mkdir(parents=True, exist_ok=True)
            
            for feature_name, result in tqdm(analysis_results.items(), desc="Generating feature reports"):
                try:
                    feature_report = reporter.generate_feature_report(
                        result=result,
                        output_format=output_format
                    )
                    
                    feature_path = (
                        feature_reports_dir / 
                        f'{feature_name}_report_{timestamp}.{output_format}'
                    )
                    
                    reporter.export_report(
                        report_content=feature_report,
                        output_path=feature_path,
                        report_type='feature'
                    )
                except Exception as e:
                    logger.error(f"Error generating report for feature {feature_name}: {str(e)}")
                    continue
        
        # 9. Generate validation summary
        validation_summary = {
            'timestamp': timestamp,
            'config_path': str(config_path),
            'features_analyzed': len(analysis_results),
            'features_with_issues': sum(1 for r in analysis_results.values() if r.summary.get('has_issues', False)),
            'output_files': {
                'results': str(results_path),
                'summary_report': str(summary_path),
                'feature_reports_dir': str(feature_reports_dir) if generate_feature_reports else None
            }
        }
        
        # Save validation summary
        summary_path = Path(config.get_output_dir()) / f'validation_summary_{timestamp}.json'
        with open(summary_path, 'w') as f:
            json.dump(validation_summary, f, indent=2)
        
        logger.info("Feature validation pipeline completed successfully!")
        logger.info(f"Results saved to: {config.get_output_dir()}")
        
    except Exception as e:
        logger.error(f"Feature validation pipeline failed: {str(e)}")
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
    
    args = parser.parse_args()
    
    validate_features(
        config_path=args.config_path,
        output_format=args.output_format,
        generate_feature_reports=not args.skip_feature_reports,
        feature_subset=args.features
    )

if __name__ == "__main__":
    main() 