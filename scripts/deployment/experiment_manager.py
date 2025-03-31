"""Experiment Manager for handling multiple A/B tests."""

import yaml
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from .ab_testing import ABTest

logger = logging.getLogger(__name__)

class ExperimentManager:
    """Manages multiple A/B test experiments."""
    
    def __init__(self, config_path: str = "config/ab_testing.yaml"):
        """Initialize the experiment manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.active_experiments: Dict[str, ABTest] = {}
        self.load_config()
        
    def load_config(self) -> None:
        """Load experiment configurations from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise
            
    def create_experiment(self, experiment_name: str) -> ABTest:
        """Create a new A/B test experiment from configuration.
        
        Args:
            experiment_name: Name of the experiment in config
            
        Returns:
            Newly created ABTest instance
            
        Raises:
            ValueError: If experiment configuration is not found
        """
        if experiment_name not in self.config["experiments"]:
            raise ValueError(f"Experiment {experiment_name} not found in config")
            
        exp_config = self.config["experiments"][experiment_name]
        variants = [v["name"] for v in exp_config["variants"]]
        
        test = ABTest(
            name=exp_config["name"],
            variants=variants,
            traffic_split=exp_config.get("traffic_split"),
            min_sample_size=self.config["defaults"]["min_sample_size"],
            confidence_level=self.config["defaults"]["confidence_level"]
        )
        
        self.active_experiments[experiment_name] = test
        logger.info(f"Created experiment: {experiment_name}")
        return test
        
    def get_experiment(self, experiment_name: str) -> Optional[ABTest]:
        """Get an active experiment by name.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            ABTest instance or None if not found
        """
        return self.active_experiments.get(experiment_name)
        
    def list_experiments(self) -> List[str]:
        """List all active experiments.
        
        Returns:
            List of experiment names
        """
        return list(self.active_experiments.keys())
        
    def check_completion_criteria(self, experiment_name: str) -> bool:
        """Check if an experiment meets completion criteria.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            True if experiment should be completed
        """
        test = self.get_experiment(experiment_name)
        if not test:
            return False
            
        exp_config = self.config["experiments"][experiment_name]
        max_duration = timedelta(days=self.config["defaults"]["max_duration_days"])
        
        # Check duration
        if datetime.now() - test.start_time > max_duration:
            logger.info(f"Experiment {experiment_name} exceeded maximum duration")
            return True
            
        # Check sample size
        stats = test.get_statistics()
        if all(s["has_sufficient_data"] for s in stats.values()):
            # Check success criteria
            significance = test.calculate_significance()
            if significance:
                success_criteria = exp_config.get("success_criteria", [])
                for criterion in success_criteria:
                    metric = criterion["metric"]
                    threshold = criterion["improvement_threshold"]
                    
                    if metric in significance and significance[metric]["is_significant"]:
                        return True
                        
        return False
        
    def end_experiment(self, experiment_name: str) -> Optional[Dict]:
        """End an experiment and get results.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Dictionary containing experiment results or None if experiment not found
        """
        test = self.get_experiment(experiment_name)
        if not test:
            return None
            
        test.end_test()
        winner, improvements = test.get_winner()
        
        results = {
            "experiment": experiment_name,
            "duration": (test.end_time - test.start_time).total_seconds(),
            "winner": winner,
            "improvements": improvements,
            "statistics": test.get_statistics(),
            "significance": test.calculate_significance()
        }
        
        # Remove from active experiments
        del self.active_experiments[experiment_name]
        
        # Log results
        logger.info(f"Ended experiment {experiment_name}. Winner: {winner}")
        return results
        
    def check_and_complete_experiments(self) -> List[Dict]:
        """Check all active experiments and complete those meeting criteria.
        
        Returns:
            List of results for completed experiments
        """
        completed_results = []
        
        for experiment_name in list(self.active_experiments.keys()):
            if self.check_completion_criteria(experiment_name):
                results = self.end_experiment(experiment_name)
                if results:
                    completed_results.append(results)
                    
        return completed_results 