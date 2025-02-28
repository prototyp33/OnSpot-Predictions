import argparse
from pathlib import Path
import logging
from parking_sim.generator import ParkingGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Parking data generator')
    parser.add_argument('--config', type=str, default='config/generator_config.json',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='data/synthetic_parking_data.csv',
                       help='Path to output CSV file')
    parser.add_argument('--profile', action='store_true',
                       help='Run with profiling enabled')
    args = parser.parse_args()
    
    if args.profile:
        from parking_sim.profiling import CodeProfiler
        profiler = CodeProfiler()
        
        # Create generator
        generator = ParkingGenerator(Path(args.config))
        
        # Profile dataset generation
        profiled_generate = profiler.profile_function(generator.generate_dataset)
        df = profiled_generate(Path(args.output))
        
        # Profile specific functions
        feature_engineering = generator.feature_engineering
        profiled_simulate = profiler.profile_line(feature_engineering.simulate_parking_durations)
        
        # Replace original method with profiled version
        feature_engineering.simulate_parking_durations = profiled_simulate
        
        # Generate report
        profiler.generate_report()
    else:
        # Normal execution
        generator = ParkingGenerator(Path(args.config))
        generator.generate_dataset(Path(args.output))

if __name__ == "__main__":
    main() 