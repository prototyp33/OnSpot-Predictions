"""
Example usage of the data pipeline module with enhanced robustness features
"""

import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
from ..config.monitoring_config import MetricsConfig, load_config
from .data_pipeline import (
    DataPipeline,
    MetricRecord,
    DataPipelineError,
    DataValidationError,
    DatabaseError
)

async def process_metrics_with_error_handling(pipeline: DataPipeline, model_id: str, start_time: datetime, end_time: datetime):
    """Process metrics with comprehensive error handling"""
    
    print(f"\nStarting robust processing for model {model_id}")
    print(f"Time range: {start_time} to {end_time}")
    
    try:
        # Extract and process metrics with validation
        metrics_generator = pipeline.extract_metrics_in_batches(
            model_id=model_id,
            start_time=start_time,
            end_time=end_time,
            batch_size=1000,
            time_chunk_size=timedelta(days=1)
        )
        
        # Calculate aggregated metrics with validation
        aggregated_metrics = await pipeline.calculate_aggregated_metrics_in_batches(
            records_generator=metrics_generator,
            window_size=timedelta(hours=24),
            percentiles=[0.25, 0.5, 0.75, 0.90, 0.95, 0.99],
            decay_factor=0.1,
            chunk_size=500
        )
        
        return aggregated_metrics
        
    except DatabaseError as e:
        print(f"\nDatabase error occurred: {str(e)}")
        print("Attempting to use backup data...")
        
        # Try to load from latest backup
        backup_data = await load_from_latest_backup(pipeline)
        if backup_data is not None:
            print("Successfully loaded data from backup")
            return backup_data
        else:
            print("No valid backup data found")
            raise
        
    except DataValidationError as e:
        print(f"\nData validation error: {str(e)}")
        print("Please check the data quality and validation rules")
        raise
        
    except DataPipelineError as e:
        print(f"\nPipeline error: {str(e)}")
        print("Please check the pipeline configuration and logs")
        raise
        
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        raise

async def load_from_latest_backup(pipeline: DataPipeline) -> pd.DataFrame:
    """Load data from the latest available backup"""
    try:
        # Check hourly backups first
        backup_dir = pipeline.backup_dir / "hourly"
        backup_files = sorted(backup_dir.glob("metrics_*.csv"), reverse=True)
        
        if not backup_files:
            # Try daily backups
            backup_dir = pipeline.backup_dir / "daily"
            backup_files = sorted(backup_dir.glob("metrics_*.csv"), reverse=True)
        
        if backup_files:
            latest_backup = backup_files[0]
            print(f"Loading backup from: {latest_backup}")
            return pd.read_csv(latest_backup)
        
        return None
        
    except Exception as e:
        print(f"Error loading backup: {str(e)}")
        return None

async def generate_test_metrics(count: int, error_rate: float = 0.1) -> List[MetricRecord]:
    """Generate test metrics with some intentional errors for testing"""
    metrics = []
    for i in range(count):
        try:
            # Introduce random errors based on error_rate
            if np.random.random() < error_rate:
                # Generate invalid metrics for testing
                if np.random.random() < 0.5:
                    value = float('inf')  # Invalid value
                else:
                    value = "invalid"  # Invalid type
            else:
                value = 0.45 + np.random.normal(0, 0.1)
            
            metrics.append(MetricRecord(
                model_id="model_v1",
                metric_name="rmse",
                value=value,
                timestamp=datetime.now() - timedelta(minutes=i),
                prediction_id=f"pred_{i}",
                metadata={"batch_size": 1000}
            ))
            
        except Exception as e:
            print(f"Error generating test metric: {str(e)}")
            continue
    
    return metrics

async def main():
    # Load configuration
    config = load_config()
    
    # Initialize pipeline with robustness features
    pipeline = DataPipeline(
        db_connection_string="postgresql://user:password@localhost:5432/monitoring_db",
        metrics_config=config.metrics,
        backup_dir="data/backups",
        max_retries=3,
        retry_delay=1.0,
        auto_backup_interval=timedelta(hours=1)
    )
    
    try:
        # Define time range for extraction
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)
        
        # Process metrics with error handling
        model_id = "model_v1"
        aggregated_metrics = await process_metrics_with_error_handling(
            pipeline=pipeline,
            model_id=model_id,
            start_time=start_time,
            end_time=end_time
        )
        
        if aggregated_metrics is not None and not aggregated_metrics.empty:
            # Display summary statistics
            print("\nAggregated Metrics Summary (Last values per metric):")
            
            print("\nBasic Statistics:")
            basic_cols = ["mean", "median", "std", "min", "max", "count"]
            print(aggregated_metrics[["metric_name", *basic_cols]].groupby("metric_name").last())
            
            print("\nPercentiles:")
            percentile_cols = [col for col in aggregated_metrics.columns if "percentile" in col]
            print(aggregated_metrics[["metric_name", *percentile_cols]].groupby("metric_name").last())
            
            print("\nAdvanced Statistics:")
            advanced_cols = ["skew", "kurtosis", "volatility", "rate_of_change"]
            print(aggregated_metrics[["metric_name", *advanced_cols]].groupby("metric_name").last())
            
            # Save results with error handling
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = Path(f"data/aggregated_metrics_{timestamp}.csv")
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                aggregated_metrics.to_csv(csv_path, index=False)
                print(f"\nSaved aggregated metrics to: {csv_path}")
            except Exception as e:
                print(f"Error saving results: {str(e)}")
        
        # Demonstrate error handling with test data
        print("\nTesting error handling with invalid data...")
        test_metrics = await generate_test_metrics(count=100, error_rate=0.2)
        
        # Validate and store test metrics
        for i, metric in enumerate(test_metrics):
            try:
                # Validate single metric
                is_valid, errors = metric.validate()
                if is_valid:
                    # Store valid metric
                    await pipeline.store_metrics([metric], storage_type="database")
                else:
                    print(f"\nValidation errors for metric {i}:")
                    for error in errors:
                        print(f"- {error}")
            except Exception as e:
                print(f"Error processing test metric {i}: {str(e)}")
                continue
        
        # Verify automated backups
        print("\nVerifying automated backups...")
        backup_dir = Path("data/backups")
        hourly_backups = list((backup_dir / "hourly").glob("metrics_*.csv"))
        daily_backups = list((backup_dir / "daily").glob("metrics_*.csv"))
        
        print(f"Found {len(hourly_backups)} hourly backups")
        print(f"Found {len(daily_backups)} daily backups")
        
        # Clean up old metrics with error handling
        try:
            deleted_count = await pipeline.cleanup_old_metrics(retention_days=90)
            print(f"\nDeleted {deleted_count} old metric records")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
        
    except Exception as e:
        print(f"Critical error: {str(e)}")
        # In a production environment, you might want to:
        # 1. Send alerts
        # 2. Write to error log
        # 3. Trigger fallback procedures
        raise

if __name__ == "__main__":
    asyncio.run(main()) 