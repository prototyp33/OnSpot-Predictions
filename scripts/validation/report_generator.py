"""
Report generator module for creating validation reports.

This module handles the generation of HTML and Markdown reports
summarizing the results of feature distribution analysis.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Union, Any, Optional
from datetime import datetime
import logging
import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader
import matplotlib.pyplot as plt
import seaborn as sns
from .analyzer import AnalysisResult

# Set up logging
logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Class for generating human-readable reports from analysis results.
    """
    
    def __init__(self, config: 'ValidationConfig'):  # Forward reference
        """
        Initialize the report generator.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.output_dir = Path(self.config.get_output_dir()) / 'reports'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up Jinja2 environment
        template_dir = Path(__file__).parent / 'templates'
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html'])
        )
        
        logger.info("Initialized ReportGenerator")
    
    def _load_results(self, results_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load analysis results from a file.
        
        Args:
            results_path: Path to JSON or YAML results file
            
        Returns:
            List of analysis result dictionaries
        """
        results_path = Path(results_path)
        
        try:
            with open(results_path, 'r') as f:
                if results_path.suffix == '.json':
                    return json.load(f)
                elif results_path.suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported file format: {results_path.suffix}")
        except Exception as e:
            logger.error(f"Failed to load results from {results_path}: {str(e)}")
            raise
    
    def _format_test_results(self, test_results: Dict[str, Any]) -> str:
        """Format statistical test results into a readable string."""
        formatted = []
        for test_name, result in test_results.items():
            # Format p-value with appropriate precision
            p_value = f"{result['p_value']:.4f}" if result['p_value'] is not None else "N/A"
            
            # Create test summary
            summary = f"{test_name}: "
            if result['is_significant']:
                summary += "⚠️ Significant difference detected "
            else:
                summary += "✓ No significant difference "
            
            summary += f"(p={p_value})"
            
            # Add interpretation if available
            if result['interpretation']:
                summary += f"\n   {result['interpretation']}"
            
            formatted.append(summary)
        
        return "\n".join(formatted)
    
    def _format_basic_stats(self, stats: Dict[str, Any], feature_type: str) -> pd.DataFrame:
        """Format basic statistics into a pandas DataFrame for easy comparison."""
        if feature_type == 'continuous':
            metrics = ['count', 'missing', 'mean', 'std', 'min', 'max', 'median']
        else:
            metrics = ['count', 'missing', 'unique']
            # Add top categories if available
            if 'top_categories' in stats[next(iter(stats.keys()))]:
                for split_stats in stats.values():
                    for cat, prop in split_stats['top_categories'].items():
                        if f"prop_{cat}" not in metrics:
                            metrics.append(f"prop_{cat}")
        
        # Create DataFrame
        stats_df = pd.DataFrame(index=metrics, columns=stats.keys())
        
        # Fill values
        for split_name, split_stats in stats.items():
            for metric in metrics:
                if metric.startswith('prop_'):
                    cat = metric[5:]  # Remove 'prop_' prefix
                    value = split_stats.get('top_categories', {}).get(cat, 0)
                else:
                    value = split_stats.get(metric, None)
                stats_df.loc[metric, split_name] = value
        
        return stats_df
    
    def generate_feature_report(
        self,
        result: Union[AnalysisResult, Dict[str, Any]],
        output_format: str = 'html'
    ) -> str:
        """
        Generate a detailed report for a single feature.
        
        Args:
            result: Analysis result for the feature
            output_format: Output format ('html' or 'markdown')
            
        Returns:
            str: Generated report in the specified format
        """
        # Convert AnalysisResult to dict if needed
        if isinstance(result, AnalysisResult):
            result = {
                'feature_name': result.feature_name,
                'feature_type': result.feature_type,
                'statistical_results': result.statistical_results,
                'visualization_paths': result.visualization_paths,
                'summary': result.summary
            }
        
        # Format statistical results
        formatted_tests = self._format_test_results(result['statistical_results'])
        
        # Format basic statistics
        stats_df = self._format_basic_stats(
            result['summary']['basic_stats'],
            result['feature_type']
        )
        
        # Prepare template context
        context = {
            'feature_name': result['feature_name'],
            'feature_type': result['feature_type'],
            'statistical_results': formatted_tests,
            'basic_stats': stats_df.to_html(
                float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
            ),
            'visualization_paths': result['visualization_paths'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Generate report
        try:
            if output_format == 'html':
                template = self.jinja_env.get_template('feature_report.html')
                return template.render(**context)
            elif output_format == 'markdown':
                template = self.jinja_env.get_template('feature_report.md')
                return template.render(**context)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
        except Exception as e:
            logger.error(f"Failed to generate feature report: {str(e)}")
            raise
    
    def generate_summary_report(
        self,
        results: Union[List[AnalysisResult], List[Dict[str, Any]], str, Path],
        output_format: str = 'html'
    ) -> str:
        """
        Generate a summary report highlighting potential issues across all features.
        
        Args:
            results: List of analysis results or path to results file
            output_format: Output format ('html' or 'markdown')
            
        Returns:
            str: Generated summary report in the specified format
        """
        # Load results if path provided
        if isinstance(results, (str, Path)):
            results = self._load_results(results)
        
        # Convert AnalysisResult objects to dicts if needed
        results_dicts = []
        for result in results:
            if isinstance(result, AnalysisResult):
                results_dicts.append({
                    'feature_name': result.feature_name,
                    'feature_type': result.feature_type,
                    'statistical_results': result.statistical_results,
                    'visualization_paths': result.visualization_paths,
                    'summary': result.summary
                })
            else:
                results_dicts.append(result)
        
        # Analyze results for issues
        issues = []
        warnings = []
        for result in results_dicts:
            feature_name = result['feature_name']
            
            # Check for significant statistical differences
            significant_tests = [
                test_name for test_name, test_result in result['statistical_results'].items()
                if test_result['is_significant']
            ]
            if significant_tests:
                issues.append({
                    'feature': feature_name,
                    'type': 'statistical_difference',
                    'details': f"Significant differences detected in tests: {', '.join(significant_tests)}"
                })
            
            # Check missing value rates
            stats = result['summary']['basic_stats']
            for split_name, split_stats in stats.items():
                missing_rate = split_stats['missing'] / split_stats['count']
                if missing_rate > self.config.get_thresholds()['missing_rate']:
                    warnings.append({
                        'feature': feature_name,
                        'type': 'high_missing_rate',
                        'details': f"High missing rate in {split_name} split: {missing_rate:.2%}"
                    })
        
        # Prepare template context
        context = {
            'issues': issues,
            'warnings': warnings,
            'total_features': len(results_dicts),
            'features_with_issues': len(set(issue['feature'] for issue in issues)),
            'features_with_warnings': len(set(warning['feature'] for warning in warnings)),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Generate report
        try:
            if output_format == 'html':
                template = self.jinja_env.get_template('summary_report.html')
                return template.render(**context)
            elif output_format == 'markdown':
                template = self.jinja_env.get_template('summary_report.md')
                return template.render(**context)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
        except Exception as e:
            logger.error(f"Failed to generate summary report: {str(e)}")
            raise
    
    def export_report(
        self,
        report_content: str,
        output_path: Union[str, Path],
        report_type: str = 'feature'
    ) -> None:
        """
        Save a report to file.
        
        Args:
            report_content: Generated report content
            output_path: Path to save the report
            report_type: Type of report ('feature' or 'summary')
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Saved {report_type} report to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {str(e)}")
            raise 