"""
Test module for the ReportGenerator class.

This module contains tests for verifying the functionality of the ReportGenerator,
including report generation, formatting, and file handling.
"""

import json
import yaml
from pathlib import Path
import tempfile
import shutil
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from . import ValidationConfig
from .report_generator import ReportGenerator
from .analyzer import AnalysisResult

@pytest.fixture
def test_config():
    """Create a test configuration."""
    # Create temporary directories for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        splits_dir = temp_path / 'splits'
        output_dir = temp_path / 'output'
        splits_dir.mkdir()
        output_dir.mkdir()
        
        config_dict = {
            'data': {
                'splits_dir': str(splits_dir),
                'reference_path': str(splits_dir / 'reference.csv'),
                'comparison_path': str(splits_dir / 'comparison.csv')
            },
            'analysis': {
                'statistical_tests': {
                    'ks_test': {
                        'enabled': True,
                        'significance_level': 0.05
                    },
                    'chi_squared': {
                        'enabled': True,
                        'significance_level': 0.05
                    },
                    'mann_whitney': {
                        'enabled': True,
                        'significance_level': 0.05
                    },
                    'psi': {
                        'enabled': True,
                        'threshold': 0.2,
                        'bins': 10
                    }
                }
            },
            'thresholds': {
                'missing_rate': 0.1,
                'significance_level': 0.05,
                'psi_threshold': 0.2
            },
            'output': {
                'base_dir': str(output_dir),
                'format': ['json', 'yaml'],
                'subdirs': {
                    'plots': 'plots',
                    'reports': 'reports',
                    'results': 'results'
                }
            },
            'visualization': {
                'style': 'seaborn',
                'color_palette': 'deep',
                'figsize': [10, 6],
                'dpi': 100,
                'continuous_plots': ['histogram', 'kde', 'box', 'qq'],
                'categorical_plots': ['bar', 'pie']
            },
            'logging': {
                'level': 'INFO',
                'file': str(output_dir / 'validation.log'),
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
        
        # Create test data files
        pd.DataFrame({'A': [1, 2, 3]}).to_csv(splits_dir / 'reference.csv', index=False)
        pd.DataFrame({'A': [2, 3, 4]}).to_csv(splits_dir / 'comparison.csv', index=False)
        
        # Create config file
        config_path = temp_path / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        
        config = ValidationConfig(str(config_path))
        yield config

@pytest.fixture
def test_results():
    """Create test analysis results."""
    # Create a continuous feature result
    continuous_result = {
        'feature_name': 'test_continuous',
        'feature_type': 'continuous',
        'statistical_results': {
            'ks_test': {
                'statistic': 0.15,
                'p_value': 0.03,
                'is_significant': True,
                'interpretation': 'Distributions are significantly different'
            },
            'psi': {
                'statistic': 0.25,
                'p_value': None,
                'is_significant': True,
                'interpretation': 'High PSI value indicates significant drift'
            }
        },
        'visualization_paths': {
            'distribution_plot': 'plots/test_continuous_dist.png',
            'qq_plot': 'plots/test_continuous_qq.png'
        },
        'summary': {
            'basic_stats': {
                'reference': {
                    'count': 1000,
                    'missing': 50,
                    'mean': 10.5,
                    'std': 2.3,
                    'min': 5.0,
                    'max': 15.0,
                    'median': 10.2
                },
                'comparison': {
                    'count': 1000,
                    'missing': 45,
                    'mean': 11.2,
                    'std': 2.5,
                    'min': 5.2,
                    'max': 16.0,
                    'median': 11.0
                }
            }
        }
    }
    
    # Create a categorical feature result
    categorical_result = {
        'feature_name': 'test_categorical',
        'feature_type': 'categorical',
        'statistical_results': {
            'chi2_test': {
                'statistic': 12.5,
                'p_value': 0.02,
                'is_significant': True,
                'interpretation': 'Category proportions are significantly different'
            },
            'psi': {
                'statistic': 0.18,
                'p_value': None,
                'is_significant': False,
                'interpretation': 'PSI value indicates stable distributions'
            }
        },
        'visualization_paths': {
            'category_distribution': 'plots/test_categorical_dist.png'
        },
        'summary': {
            'basic_stats': {
                'reference': {
                    'count': 1000,
                    'missing': 30,
                    'unique': 4,
                    'top_categories': {
                        'A': 0.4,
                        'B': 0.3,
                        'C': 0.2,
                        'D': 0.1
                    }
                },
                'comparison': {
                    'count': 1000,
                    'missing': 25,
                    'unique': 4,
                    'top_categories': {
                        'A': 0.35,
                        'B': 0.35,
                        'C': 0.2,
                        'D': 0.1
                    }
                }
            }
        }
    }
    
    return [continuous_result, categorical_result]

def test_report_generator_initialization(test_config):
    """Test ReportGenerator initialization."""
    generator = ReportGenerator(test_config)
    assert generator.config == test_config
    assert generator.output_dir == Path(test_config.get_output_dir()) / 'reports'
    assert generator.output_dir.exists()

def test_format_test_results(test_config, test_results):
    """Test formatting of statistical test results."""
    generator = ReportGenerator(test_config)
    
    # Test continuous feature formatting
    continuous_formatted = generator._format_test_results(
        test_results[0]['statistical_results']
    )
    assert '⚠️' in continuous_formatted  # Should show warning for significant difference
    assert 'ks_test' in continuous_formatted
    assert 'psi' in continuous_formatted
    
    # Test categorical feature formatting
    categorical_formatted = generator._format_test_results(
        test_results[1]['statistical_results']
    )
    assert '⚠️' in categorical_formatted  # Should show warning for chi2 test
    assert 'chi2_test' in categorical_formatted
    assert 'psi' in categorical_formatted

def test_format_basic_stats(test_config, test_results):
    """Test formatting of basic statistics."""
    generator = ReportGenerator(test_config)
    
    # Test continuous feature stats
    continuous_stats = generator._format_basic_stats(
        test_results[0]['summary']['basic_stats'],
        'continuous'
    )
    assert isinstance(continuous_stats, pd.DataFrame)
    assert all(metric in continuous_stats.index for metric in 
              ['count', 'missing', 'mean', 'std', 'min', 'max', 'median'])
    
    # Test categorical feature stats
    categorical_stats = generator._format_basic_stats(
        test_results[1]['summary']['basic_stats'],
        'categorical'
    )
    assert isinstance(categorical_stats, pd.DataFrame)
    assert all(metric in categorical_stats.index for metric in 
              ['count', 'missing', 'unique'])
    assert all(f"prop_{cat}" in categorical_stats.index for cat in ['A', 'B', 'C', 'D'])

def test_generate_feature_report(test_config, test_results):
    """Test generation of feature-level reports."""
    generator = ReportGenerator(test_config)
    
    # Test HTML report generation
    html_report = generator.generate_feature_report(test_results[0], 'html')
    assert isinstance(html_report, str)
    assert 'test_continuous' in html_report
    assert 'distribution_plot' in html_report
    assert 'qq_plot' in html_report
    
    # Test Markdown report generation
    md_report = generator.generate_feature_report(test_results[1], 'markdown')
    assert isinstance(md_report, str)
    assert 'test_categorical' in md_report
    assert 'category_distribution' in md_report

def test_generate_summary_report(test_config, test_results):
    """Test generation of summary reports."""
    generator = ReportGenerator(test_config)
    
    # Test HTML summary report
    html_summary = generator.generate_summary_report(test_results, 'html')
    assert isinstance(html_summary, str)
    assert 'Total Features' in html_summary
    assert str(len(test_results)) in html_summary
    assert 'Critical Issues' in html_summary
    
    # Test Markdown summary report
    md_summary = generator.generate_summary_report(test_results, 'markdown')
    assert isinstance(md_summary, str)
    assert '## Overview' in md_summary
    assert str(len(test_results)) in md_summary

def test_export_report(test_config, test_results):
    """Test report export functionality."""
    generator = ReportGenerator(test_config)
    
    # Create temporary directory for test outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test HTML export
        html_report = generator.generate_feature_report(test_results[0], 'html')
        html_path = Path(temp_dir) / 'test_report.html'
        generator.export_report(html_report, html_path, 'feature')
        assert html_path.exists()
        assert html_path.stat().st_size > 0
        
        # Test Markdown export
        md_report = generator.generate_summary_report(test_results, 'markdown')
        md_path = Path(temp_dir) / 'test_summary.md'
        generator.export_report(md_report, md_path, 'summary')
        assert md_path.exists()
        assert md_path.stat().st_size > 0

def test_load_results(test_config, test_results):
    """Test loading results from files."""
    generator = ReportGenerator(test_config)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test JSON loading
        json_path = Path(temp_dir) / 'test_results.json'
        with open(json_path, 'w') as f:
            json.dump(test_results, f)
        loaded_json = generator._load_results(json_path)
        assert loaded_json == test_results
        
        # Test YAML loading
        yaml_path = Path(temp_dir) / 'test_results.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(test_results, f)
        loaded_yaml = generator._load_results(yaml_path)
        assert loaded_yaml == test_results
        
        # Test invalid format
        invalid_path = Path(temp_dir) / 'test_results.txt'
        invalid_path.touch()
        with pytest.raises(ValueError):
            generator._load_results(invalid_path) 