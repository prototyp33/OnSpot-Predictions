"""Module for monitoring code performance and resource usage."""

import time
import logging
import psutil
import functools
from typing import Any, Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitors execution time and resource usage."""
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.process = psutil.Process()
        self.timings = {}
        self.memory_usage = {}
    
    @contextmanager
    def monitor(self, section_name: str):
        """
        Context manager to monitor a section of code.
        
        Args:
            section_name: Name of the code section being monitored
        """
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Store metrics
            if section_name not in self.timings:
                self.timings[section_name] = []
            if section_name not in self.memory_usage:
                self.memory_usage[section_name] = []
                
            self.timings[section_name].append(execution_time)
            self.memory_usage[section_name].append(memory_delta)
            
            # Log performance metrics
            logger.debug(f"Performance - {section_name}:")
            logger.debug(f"  Time: {execution_time:.3f} seconds")
            logger.debug(f"  Memory delta: {memory_delta:.1f} MB")
    
    def log_summary(self):
        """Log summary of all monitored sections."""
        logger.info("\nPerformance Summary:")
        for section in self.timings:
            times = self.timings[section]
            memory = self.memory_usage[section]
            
            logger.info(f"\n{section}:")
            logger.info(f"  Executions: {len(times)}")
            logger.info(f"  Average time: {sum(times)/len(times):.3f} seconds")
            logger.info(f"  Total time: {sum(times):.3f} seconds")
            logger.info(f"  Average memory delta: {sum(memory)/len(memory):.1f} MB")
            logger.info(f"  Max memory delta: {max(memory):.1f} MB")

def monitor_performance(section_name: str):
    """
    Decorator to monitor function performance.
    
    Args:
        section_name: Name for the monitored section
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            monitor = PerformanceMonitor()
            with monitor.monitor(section_name):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator 