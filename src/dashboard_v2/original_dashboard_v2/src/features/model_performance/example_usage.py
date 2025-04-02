"""
Example usage of the model performance monitoring module with alerts, comparison, health monitoring,
data quality monitoring, and automated responses
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from src.core.database import DatabaseConnection
from src.features.model_performance.business_logic.metrics_calculator import ModelMetricsCalculator
from src.features.model_performance.business_logic.drift_detector import DriftDetector
from src.features.model_performance.business_logic.alert_manager import AlertManager
from src.features.model_performance.business_logic.model_comparison import ModelComparator
from src.features.model_performance.business_logic.health_monitor import HealthMonitor
from src.features.model_performance.business_logic.data_quality import DataQualityMonitor
from src.features.model_performance.business_logic.auto_response import AutoResponseManager
from src.features.model_performance.data_access.metrics_repository import MetricsRepository
from src.features.model_performance.visualization.monitoring_dashboard import MonitoringDashboard
from src.features.model_performance.visualization.quality_visualizations import DataQualityVisualizer

async def main():
    # Initialize components
    db_connection = DatabaseConnection()
    metrics_repository = MetricsRepository(db_connection)
    alert_manager = AlertManager()
    health_monitor = HealthMonitor()
    data_quality_monitor = DataQualityMonitor()
    quality_visualizer = DataQualityVisualizer()
    
    # Initialize auto response manager
    auto_response_manager = AutoResponseManager(
        metrics_repository=metrics_repository,
        alert_manager=alert_manager,
        health_monitor=health_monitor,
        data_quality_monitor=data_quality_monitor
    )
    
    # Create multiple model instances for comparison
    model_ids = ["model_v1", "model_v2", "model_v3"]
    calculators = {
        model_id: ModelMetricsCalculator(model_id)
        for model_id in model_ids
    }
    drift_detector = DriftDetector()
    model_comparator = ModelComparator(metrics_repository)
    
    # Generate example data for multiple models
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    for model_id in model_ids:
        # Simulate different model versions with varying performance
        noise_level = 0.1 if model_id == "model_v3" else 0.2 if model_id == "model_v2" else 0.3
        
        # Reference data (clean)
        reference_features = np.random.normal(0, 1, (n_samples, n_features))
        reference_target = np.sum(reference_features, axis=1) + np.random.normal(0, 0.1, n_samples)
        reference_predictions = reference_target + np.random.normal(0, noise_level, n_samples)
        
        # Current data (with issues)
        drift_magnitude = 0.5 if model_id in ["model_v1", "model_v2"] else 0.0
        current_features = np.random.normal(drift_magnitude, 1.2, (n_samples, n_features))
        
        # Introduce some missing values
        mask = np.random.random(current_features.shape) < 0.05
        current_features[mask] = np.nan
        
        # Introduce some outliers
        outlier_mask = np.random.random(current_features.shape) < 0.02
        current_features[outlier_mask] = np.random.normal(10, 2, np.sum(outlier_mask))
        
        current_target = np.sum(np.nan_to_num(current_features), axis=1) + np.random.normal(0, 0.1, n_samples)
        current_predictions = current_target + np.random.normal(0, noise_level * 1.2, n_samples)
        
        try:
            # Set reference data for quality monitoring
            data_quality_monitor.set_reference_data(reference_features, feature_names)
            
            # Check data quality
            quality_metrics = data_quality_monitor.check_data_quality(
                current_features,
                feature_names
            )
            
            # Calculate feature importance based on quality issues
            feature_importance = data_quality_monitor.get_feature_importance(quality_metrics)
            
            # Create quality visualizations
            quality_overview = quality_visualizer.create_quality_overview(
                {
                    "overall_score": data_quality_monitor.get_quality_score(quality_metrics),
                    "missing_rate": quality_metrics.missing_rate,
                    "out_of_range_rate": quality_metrics.out_of_range_rate
                },
                feature_importance
            )
            
            missing_values_heatmap = quality_visualizer.create_missing_values_heatmap(
                {f: [float(np.isnan(current_features[:, i]).mean())]
                 for i, f in enumerate(feature_names)},
                feature_names,
                [datetime.now()]
            )
            
            distribution_shifts = quality_visualizer.create_distribution_shifts(
                quality_metrics.distribution_metrics,
                feature_names
            )
            
            # Calculate and store performance metrics
            current_metrics = calculators[model_id].calculate_regression_metrics(
                current_target,
                current_predictions
            )
            
            for metric_name, result in current_metrics.items():
                await metrics_repository.store_metric(
                    model_id=model_id,
                    metric_name=metric_name,
                    metric_result=result
                )
            
            print(f"\nStored performance metrics for {model_id}:")
            for metric_name, result in current_metrics.items():
                print(f"{metric_name}: {result.value:.4f}")
            
            # Check for alerts
            new_alerts = alert_manager.check_performance_metrics(
                {name: result.value for name, result in current_metrics.items()},
                model_id
            )
            
            if new_alerts:
                print(f"\nNew alerts for {model_id}:")
                for alert in new_alerts:
                    print(f"- {alert.message} (severity: {alert.severity.value})")
            
            # Detect drift
            drift_metrics = drift_detector.detect_feature_drift(
                reference_features,
                current_features,
                feature_names
            )
            
            # Check for drift alerts
            drift_alerts = alert_manager.check_drift_metrics(
                {name: result.value for name, result in drift_metrics.items()},
                model_id
            )
            
            if drift_alerts:
                print(f"\nDrift alerts for {model_id}:")
                for alert in drift_alerts:
                    print(f"- {alert.message} (severity: {alert.severity.value})")
            
            # Simulate health metrics
            for _ in range(10):
                latency = np.random.normal(50, 10)  # Simulate varying latency
                is_error = np.random.random() < 0.05  # 5% error rate
                
                health_metrics = health_monitor.record_prediction_metrics(
                    model_id=model_id,
                    latency_ms=latency,
                    is_error=is_error
                )
            
            # Get health summary
            health_summary = health_monitor.get_health_summary(model_id)
            print(f"\nHealth summary for {model_id}:")
            print(f"Health score: {health_summary.value:.2f}")
            print(f"Status: {health_summary.metadata['status']}")
            
            # Print data quality summary
            print(f"\nData quality summary for {model_id}:")
            print(f"Quality score: {data_quality_monitor.get_quality_score(quality_metrics):.2f}")
            print(f"Missing rate: {quality_metrics.missing_rate:.2%}")
            print(f"Out of range rate: {quality_metrics.out_of_range_rate:.2%}")
            print("\nFeature importance:")
            for feature, importance in feature_importance.items():
                print(f"{feature}: {importance:.4f}")
            
            # Get automated recommendations
            recommended_actions = auto_response_manager.analyze_and_respond(
                model_id=model_id,
                current_metrics={
                    name: result.value for name, result in current_metrics.items()
                },
                quality_metrics={
                    "overall_score": data_quality_monitor.get_quality_score(quality_metrics),
                    "missing_rate": quality_metrics.missing_rate,
                    "out_of_range_rate": quality_metrics.out_of_range_rate,
                    "drift_magnitude": drift_magnitude
                },
                health_metrics={
                    "error_rate": health_summary.metadata.get("error_rate", 0),
                    "latency_ms": health_summary.metadata.get("avg_latency", 0)
                }
            )
            
            print(f"\nRecommended actions for {model_id}:")
            for action in recommended_actions:
                print(f"- {action.description} (Priority: {action.priority}, Confidence: {action.confidence:.2%})")
                
                # Simulate action execution for high-confidence actions
                if action.confidence >= 0.9:
                    success = auto_response_manager.execute_action(action)
                    print(f"  Action execution: {'Successful' if success else 'Failed'}")
        
        except Exception as e:
            print(f"Error processing {model_id}: {str(e)}")
    
    # Compare models
    print("\nModel Comparison:")
    comparison_results = await model_comparator.compare_models(
        model_ids=model_ids,
        metrics=["rmse", "mae", "r2"]
    )
    
    for metric_name, result in comparison_results.items():
        print(f"\n{metric_name}:")
        print(f"Best model: {result.best_model}")
        print(f"Improvement: {result.improvement:.2f}%")
        print("Values:")
        for model_id, value in result.values.items():
            print(f"  {model_id}: {value:.4f}")
    
    # Create and run dashboard
    dashboard = MonitoringDashboard(
        model_ids=model_ids,
        metrics_repository=metrics_repository,
        alert_manager=alert_manager,
        health_monitor=health_monitor,
        model_comparator=model_comparator,
        data_quality_monitor=data_quality_monitor,
        auto_response_manager=auto_response_manager
    )
    
    # Run the dashboard
    dashboard.run(host="0.0.0.0", port=8050)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 