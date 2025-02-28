"""Module for profiling code performance and identifying bottlenecks."""

import cProfile
import pstats
import io
import time
import logging
import functools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from memory_profiler import profile as memory_profile

logger = logging.getLogger(__name__)

class CodeProfiler:
    """
    Comprehensive code profiling tools.
    
    This class provides multiple profiling methods:
    1. Function-level profiling with cProfile
    2. Line-by-line profiling with memory_profiler
    3. Memory usage tracking
    4. Visualization of profiling results
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the code profiler.
        
        Args:
            output_dir: Directory to save profiling results
        """
        self.output_dir = output_dir or Path('profiling_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.function_stats = {}
        
    def profile_function(self, func: Callable) -> Callable:
        """
        Decorator to profile a function using cProfile.
        
        Args:
            func: Function to profile
            
        Returns:
            Wrapped function with profiling
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            profiler.disable()
            
            # Process stats
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            
            # Store stats
            self.function_stats[func.__name__] = {
                'text': s.getvalue(),
                'time': execution_time,
                'stats': ps
            }
            
            # Save to file
            stats_path = self.output_dir / f"{func.__name__}_profile.txt"
            with open(stats_path, 'w') as f:
                f.write(s.getvalue())
                
            logger.info(f"Function profile saved to {stats_path}")
            
            return result
        return wrapper
    
    def profile_line(self, func: Callable) -> Callable:
        """
        Profile a function line by line.
        
        Args:
            func: Function to profile
            
        Returns:
            Profiled function
        """
        # Use memory_profiler instead of line_profiler
        profiled_func = memory_profile(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = profiled_func(*args, **kwargs)
            return result
            
        return wrapper
    
    def compare_implementations(self, 
                              funcs: Dict[str, Callable], 
                              args_list: List[Dict],
                              num_runs: int = 5) -> Dict[str, Dict]:
        """
        Compare multiple implementations of the same functionality.
        
        Args:
            funcs: Dictionary of function names to functions
            args_list: List of argument dictionaries to pass to each function
            num_runs: Number of times to run each function
            
        Returns:
            Dictionary of performance metrics for each function
        """
        results = {}
        
        for name, func in funcs.items():
            times = []
            
            for _ in range(num_runs):
                for args in args_list:
                    start = time.time()
                    func(**args)
                    times.append(time.time() - start)
            
            results[name] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'times': times
            }
            
            logger.info(f"{name}: {results[name]['mean']:.4f}s ± {results[name]['std']:.4f}s")
        
        # Create comparison chart
        self._plot_comparison(results)
        
        return results
    
    def _plot_comparison(self, results: Dict[str, Dict]) -> None:
        """Create a bar chart comparing function performance."""
        names = list(results.keys())
        means = [results[name]['mean'] for name in names]
        stds = [results[name]['std'] for name in names]
        
        plt.figure(figsize=(10, 6))
        plt.bar(names, means, yerr=stds, capsize=10)
        plt.ylabel('Execution Time (s)')
        plt.title('Performance Comparison')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add text labels
        for i, (name, mean) in enumerate(zip(names, means)):
            plt.text(i, mean + stds[i] + 0.01, f"{mean:.4f}s", 
                    ha='center', va='bottom', fontweight='bold')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png')
        plt.close()
    
    def generate_report(self) -> None:
        """Generate a comprehensive profiling report."""
        report_path = self.output_dir / 'profiling_report.html'
        
        with open(report_path, 'w') as f:
            f.write('<html><head><title>Profiling Report</title>')
            f.write('<style>body{font-family:Arial;max-width:1200px;margin:0 auto;padding:20px}')
            f.write('pre{background:#f5f5f5;padding:10px;overflow:auto}')
            f.write('h2{color:#333;border-bottom:1px solid #ccc}')
            f.write('table{border-collapse:collapse;width:100%}')
            f.write('th,td{text-align:left;padding:8px;border:1px solid #ddd}')
            f.write('th{background-color:#f2f2f2}</style></head><body>')
            
            f.write('<h1>Code Profiling Report</h1>')
            
            # Function profiles
            f.write('<h2>Function Profiles</h2>')
            for func_name, data in self.function_stats.items():
                f.write(f'<h3>{func_name} ({data["time"]:.4f}s)</h3>')
                f.write(f'<pre>{data["text"]}</pre>')
            
            # Images
            comparison_img = self.output_dir / 'performance_comparison.png'
            if comparison_img.exists():
                f.write('<h2>Performance Comparison</h2>')
                f.write(f'<img src="{comparison_img.name}" alt="Performance Comparison">')
            
            f.write('</body></html>')
        
        logger.info(f"Profiling report generated at {report_path}")

# Example usage
def profile_code():
    """Profile the parking simulation code."""
    from .generator import ParkingGenerator
    from pathlib import Path
    
    profiler = CodeProfiler()
    
    # Load generator with profiling
    config_path = Path('config/generator_config.json')
    generator = ParkingGenerator(config_path)
    
    # Profile dataset generation
    profiled_generate = profiler.profile_function(generator.generate_dataset)
    df = profiled_generate()
    
    # Generate report
    profiler.generate_report()
    
    return df

if __name__ == "__main__":
    profile_code() 