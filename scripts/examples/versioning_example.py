#!/usr/bin/env python3
"""
Example script demonstrating the OnSpot model versioning system.

This script shows how to:
1. Initialize the versioning system
2. Train and register a model
3. Track model metadata
4. Load models from the registry
5. Compare model versions
6. Promote models to production
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('versioning_example')

# Add the project root to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)

# Import the model versioning system
from scripts.models import ModelVersioning, ModelIntegration, create_model_metadata

def generate_synthetic_data(n_samples=1000, n_features=10):
    """Generate synthetic parking data for demonstration."""
    np.random.seed(42)
    
    # Generate feature data
    X = np.random.randn(n_samples, n_features)
    
    # Generate target (parking occupancy)
    # Base occupancy on time of day pattern plus some features
    hour_of_day = np.random.randint(0, 24, n_samples)
    
    # Create time-of-day pattern (higher occupancy during working hours)
    base_occupancy = np.sin((hour_of_day - 6) * np.pi / 12) * 0.4 + 0.5
    
    # Add feature effects
    feature_effect = X[:, 0] * 0.1 + X[:, 1] * -0.05 + X[:, 2] * 0.15
    
    # Add noise
    noise = np.random.randn(n_samples) * 0.1
    
    # Compute final occupancy (clipped to 0-1 range)
    occupancy = np.clip(base_occupancy + feature_effect + noise, 0, 1)
    
    # Create a dataframe
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['hour_of_day'] = hour_of_day
    df['day_of_week'] = np.random.randint(0, 7, n_samples)
    df['is_holiday'] = np.random.randint(0, 2, n_samples)
    df['occupancy'] = occupancy
    
    return df

def train_model(X_train, y_train, n_estimators=100, max_depth=10):
    """Train a random forest model for parking occupancy prediction."""
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model, "random_forest"

def main():
    """Run the versioning example."""
    # Create example directories
    os.makedirs("example_registry", exist_ok=True)
    os.makedirs("example_production", exist_ok=True)
    
    logger.info("Initializing model versioning system...")
    
    # Initialize the model versioning system
    versioning = ModelVersioning(
        registry_path="example_registry",
        production_models_path="example_production"
    )
    
    # Initialize the model integration
    integration = ModelIntegration(
        registry_path="example_registry",
        production_models_path="example_production"
    )
    
    # Generate synthetic data
    logger.info("Generating synthetic parking data...")
    data = generate_synthetic_data(n_samples=1000, n_features=10)
    
    # Split data
    logger.info("Preparing data...")
    features = [col for col in data.columns if col != 'occupancy']
    target = 'occupancy'
    
    X = data[features]
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train a model
    logger.info("Training base model...")
    model, model_type = train_model(X_train, y_train, n_estimators=100)
    
    # Create a scaler for preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Register the model using the integration class
    logger.info("Registering base model...")
    base_model_id = integration.register_trained_model(
        model=model,
        model_name="parking_occupancy",
        model_type=model_type,
        training_data=data,
        features=features,
        target=target,
        X_test=X_test,
        y_test=y_test,
        training_config={
            "n_estimators": 100,
            "max_depth": 10
        },
        task_type="regression",
        additional_artifacts={
            "scaler": scaler
        },
        additional_metadata={
            "description": "Base parking occupancy prediction model",
            "created_by": "versioning_example.py"
        }
    )
    
    logger.info(f"Base model registered with ID: {base_model_id}")
    
    # Train an improved model
    logger.info("Training improved model...")
    improved_model, model_type = train_model(X_train, y_train, n_estimators=200, max_depth=15)
    
    # Register the improved model
    logger.info("Registering improved model...")
    improved_model_id = integration.register_trained_model(
        model=improved_model,
        model_name="parking_occupancy",
        model_type=model_type,
        training_data=data,
        features=features,
        target=target,
        X_test=X_test,
        y_test=y_test,
        training_config={
            "n_estimators": 200,
            "max_depth": 15
        },
        task_type="regression",
        additional_artifacts={
            "scaler": scaler
        },
        additional_metadata={
            "description": "Improved parking occupancy prediction model",
            "created_by": "versioning_example.py"
        },
        change_type="minor"  # This is a minor change (more estimators, deeper trees)
    )
    
    logger.info(f"Improved model registered with ID: {improved_model_id}")
    
    # List all models in the registry
    logger.info("Listing all models in the registry:")
    models = versioning.registry.list_models()
    for model_info in models:
        logger.info(f"- {model_info['name']} v{model_info['version']} (ID: {model_info['model_id']})")
    
    # Compare the two models
    logger.info("Comparing models:")
    comparison = versioning.compare_models(base_model_id, improved_model_id)
    
    # Print key comparison results
    for metric, values in comparison["metrics_comparison"].items():
        if isinstance(values, dict) and "difference" in values:
            logger.info(f"- {metric}: {values['model_1']:.4f} -> {values['model_2']:.4f} (diff: {values['difference']:.4f})")
    
    # Promote the improved model to production
    logger.info("Promoting improved model to production...")
    production_path = versioning.promote_to_production(improved_model_id)
    logger.info(f"Model promoted to production at: {production_path}")
    
    # Demonstrate loading a production model
    logger.info("Loading the production model...")
    prod_model, prod_metadata = versioning.load_production_model("parking_occupancy")
    
    logger.info(f"Loaded production model: {prod_metadata['name']} v{prod_metadata['version']}")
    
    # Test the production model
    logger.info("Testing production model...")
    predictions = prod_model.predict(X_test)
    
    # Calculate mean absolute error
    mae = np.mean(np.abs(predictions - y_test))
    logger.info(f"Production model MAE on test data: {mae:.4f}")
    
    # List all production models
    logger.info("Listing all production models:")
    production_models = versioning.list_production_models()
    for prod_model in production_models:
        logger.info(f"- {prod_model['name']} v{prod_model['version']} (dir: {prod_model['directory']})")
    
    # Demonstrate model lineage
    logger.info("Model lineage for 'parking_occupancy':")
    lineage = versioning.get_model_lineage("parking_occupancy")
    for model_version in lineage:
        logger.info(f"- v{model_version['version']} (ID: {model_version['model_id']})")
    
    # Demonstrate using the high-level retrain function
    logger.info("Demonstrating model retraining...")
    
    # Generate new synthetic data (slightly different distribution)
    new_data = generate_synthetic_data(n_samples=1200, n_features=10)
    new_data['feature_0'] = new_data['feature_0'] * 1.2  # Introduce some drift
    
    # Use the integration class to retrain and register
    retrained_model_id = integration.retrain_and_register(
        training_function=train_model,
        model_name="parking_occupancy",
        training_data=new_data,
        features=features,
        target=target,
        test_size=0.2,
        task_type="regression",
        training_args={
            "n_estimators": 150,
            "max_depth": 12
        },
        auto_promote=True,
        change_type="patch"
    )
    
    logger.info(f"Retrained model registered with ID: {retrained_model_id}")
    
    # Demonstrate evaluating a model on new data
    logger.info("Evaluating model on new data...")
    
    # Generate evaluation data
    eval_data = generate_synthetic_data(n_samples=500, n_features=10)
    
    # Evaluate the model
    evaluation_results = integration.evaluate_and_monitor(
        model_id=retrained_model_id,
        evaluation_data=eval_data,
        features=features,
        target=target,
        task_type="regression"
    )
    
    logger.info("Evaluation results:")
    for metric, value in evaluation_results["performance_metrics"].items():
        logger.info(f"- {metric}: {value:.4f}")
    
    logger.info(f"Data drift score: {evaluation_results['data_drift_metrics']['overall_drift_score']:.4f}")
    logger.info(f"Model drift score: {evaluation_results['model_drift_metrics']['overall_drift_score']:.4f}")
    
    logger.info("Example completed successfully!")

if __name__ == "__main__":
    main() 