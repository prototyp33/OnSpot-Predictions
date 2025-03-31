#!/usr/bin/env python3
"""
Enhanced prediction service for OnSpot Predictive Model.

This service is designed to run in containers with support for:
- Loading models from model registry
- Multiple deployment patterns (production, shadow, canary, A/B testing)
- Performance monitoring and metrics reporting
- Health checks and diagnostics
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import logging
import sys
import json
import time
from datetime import datetime
import uuid
import threading
import traceback
import requests
from prometheus_client import Counter, Histogram, start_http_server
from typing import Dict, List, Any, Optional, Tuple, Union

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from model versioning system
from scripts.models import ModelVersioning, ModelRegistry
from scripts.models.metadata import compute_performance_metrics

# Set up logging
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Get environment variables
MODEL_REGISTRY_PATH = os.environ.get('MODEL_REGISTRY_PATH', './model_registry')
PRODUCTION_MODELS_PATH = os.environ.get('PRODUCTION_MODELS_PATH', './production_models')
DEPLOYMENT_TYPE = os.environ.get('DEPLOYMENT_TYPE', 'production')  # production, shadow, canary, a_b_test
SHADOW_MODEL_VERSION = os.environ.get('SHADOW_MODEL_VERSION', 'latest_candidate')
CANARY_MODEL_VERSION = os.environ.get('CANARY_MODEL_VERSION', 'latest_candidate')
ENABLE_METRICS = os.environ.get('ENABLE_METRICS', 'true').lower() == 'true'
METRICS_PORT = int(os.environ.get('METRICS_PORT', '8001'))

# Initialize model versioning
versioning = ModelVersioning(
    registry_path=MODEL_REGISTRY_PATH,
    production_models_path=PRODUCTION_MODELS_PATH
)

# Store loaded models
loaded_models = {}

# Prometheus metrics
if ENABLE_METRICS:
    prediction_requests = Counter('prediction_requests_total', 'Total prediction requests', ['model_name', 'model_version', 'deployment_type'])
    prediction_errors = Counter('prediction_errors_total', 'Total prediction errors', ['model_name', 'model_version', 'error_type'])
    prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency in seconds', ['model_name', 'model_version'])
    ground_truth_metrics = Counter('ground_truth_metrics', 'Metrics comparing predictions to ground truth', ['model_name', 'model_version', 'metric_name'])
    
    # Start metrics server
    start_http_server(METRICS_PORT)
    logger.info(f"Prometheus metrics server started on port {METRICS_PORT}")

def load_model(model_name: str, model_version: str = 'latest') -> Tuple[Any, Dict[str, Any]]:
    """
    Load a model from the registry or production directory.
    
    Args:
        model_name: Name of the model to load
        model_version: Version of the model (or 'latest')
        
    Returns:
        Tuple of (model, metadata)
    """
    model_key = f"{model_name}_{model_version}"
    
    if model_key in loaded_models:
        logger.debug(f"Using cached model {model_key}")
        return loaded_models[model_key]
    
    try:
        if model_version == 'latest':
            logger.info(f"Loading latest production model for {model_name}")
            model, metadata = versioning.load_production_model(model_name)
        else:
            logger.info(f"Loading specific model version {model_version} for {model_name}")
            # Find the model ID for this version
            models = versioning.get_model_lineage(model_name)
            model_id = None
            
            for model_info in models:
                if model_info['version'] == model_version:
                    model_id = model_info['model_id']
                    break
            
            if not model_id:
                raise ValueError(f"Model version {model_version} not found for {model_name}")
            
            model, metadata = versioning.registry.load_model(model_id, with_metadata=True)
        
        # Cache the model
        loaded_models[model_key] = (model, metadata)
        logger.info(f"Successfully loaded model {model_name} version {metadata.get('version', 'unknown')}")
        
        return model, metadata
    
    except Exception as e:
        logger.error(f"Error loading model {model_name} version {model_version}: {str(e)}")
        raise

def get_model_for_deployment(model_name: str) -> Tuple[Any, Dict[str, Any], str]:
    """
    Get the appropriate model based on deployment type.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Tuple of (model, metadata, model_version)
    """
    if DEPLOYMENT_TYPE == 'production':
        model, metadata = load_model(model_name, 'latest')
        return model, metadata, metadata.get('version', 'unknown')
    
    elif DEPLOYMENT_TYPE == 'shadow':
        try:
            model, metadata = load_model(model_name, SHADOW_MODEL_VERSION)
            return model, metadata, metadata.get('version', 'unknown')
        except Exception as e:
            logger.error(f"Error loading shadow model, falling back to production: {str(e)}")
            model, metadata = load_model(model_name, 'latest')
            return model, metadata, metadata.get('version', 'unknown')
    
    elif DEPLOYMENT_TYPE == 'canary':
        try:
            model, metadata = load_model(model_name, CANARY_MODEL_VERSION)
            return model, metadata, metadata.get('version', 'unknown')
        except Exception as e:
            logger.error(f"Error loading canary model, falling back to production: {str(e)}")
            model, metadata = load_model(model_name, 'latest')
            return model, metadata, metadata.get('version', 'unknown')
    
    else:
        # Default to production
        model, metadata = load_model(model_name, 'latest')
        return model, metadata, metadata.get('version', 'unknown')

def log_prediction_metrics(model_name: str, model_version: str, latency: float, error: Optional[str] = None):
    """
    Log prediction metrics to Prometheus.
    
    Args:
        model_name: Name of the model
        model_version: Version of the model
        latency: Prediction latency in seconds
        error: Error message if prediction failed
    """
    if not ENABLE_METRICS:
        return
    
    prediction_requests.labels(model_name=model_name, model_version=model_version, deployment_type=DEPLOYMENT_TYPE).inc()
    prediction_latency.labels(model_name=model_name, model_version=model_version).observe(latency)
    
    if error:
        prediction_errors.labels(model_name=model_name, model_version=model_version, error_type=type(error).__name__).inc()

def update_ground_truth_metrics(model_name: str, model_version: str, y_true: np.ndarray, y_pred: np.ndarray, task_type: str = 'regression'):
    """
    Update metrics comparing predictions to ground truth.
    
    Args:
        model_name: Name of the model
        model_version: Version of the model
        y_true: True values
        y_pred: Predicted values
        task_type: Type of task ('regression', 'classification', 'binary_classification')
    """
    if not ENABLE_METRICS:
        return
    
    metrics = compute_performance_metrics(y_true, y_pred, task_type)
    
    for metric_name, value in metrics.items():
        ground_truth_metrics.labels(model_name=model_name, model_version=model_version, metric_name=metric_name).inc(value)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    health_info = {
        'status': 'healthy',
        'deployment_type': DEPLOYMENT_TYPE,
        'registry_path': MODEL_REGISTRY_PATH,
        'production_models_path': PRODUCTION_MODELS_PATH,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
    }
    
    # Add information about loaded models
    models_info = []
    for model_key, (_, metadata) in loaded_models.items():
        models_info.append({
            'key': model_key,
            'name': metadata.get('name', 'unknown'),
            'version': metadata.get('version', 'unknown'),
            'created_at': metadata.get('created_at', 'unknown'),
        })
    
    health_info['loaded_models'] = models_info
    health_info['loaded_models_count'] = len(models_info)
    
    return jsonify(health_info)

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction based on input data."""
    request_id = str(uuid.uuid4())
    request_start = time.time()
    
    try:
        # Get input data
        data = request.json
        
        # Get model name from request or use default
        model_name = data.get('model_name', 'parking_occupancy')
        
        # Load the appropriate model
        model, metadata, model_version = get_model_for_deployment(model_name)
        
        # Convert to DataFrame
        input_data = {}
        for key, value in data.items():
            if key not in ['model_name', 'model_version']:
                input_data[key] = value
        
        df = pd.DataFrame([input_data])
        
        # Add timestamp if not present
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now().isoformat()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Get feature list from metadata
        features = metadata.get('features', [])
        
        # Ensure all required features are present
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            
            # Fill with defaults if needed
            for feature in missing_features:
                df[feature] = 0
        
        # Handle post-processing artifacts if needed
        scaler_data = None
        artifacts = metadata.get('artifacts', {})
        if 'scaler' in artifacts:
            # This assumes the scaler is in the artifacts directory
            artifact_path = os.path.join(
                versioning.registry.models_path, 
                metadata.get('model_id', ''), 
                'artifacts', 
                'scaler.pkl'
            )
            if os.path.exists(artifact_path):
                scaler_data = joblib.load(artifact_path)
                # Apply scaling to features
                num_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
                if num_features:
                    df[num_features] = scaler_data.transform(df[num_features])
        
        # Prepare features for prediction
        X = df[features]
        
        # Make prediction
        prediction_start = time.time()
        prediction = model.predict(X)
        prediction_end = time.time()
        
        # Calculate latency
        prediction_latency_value = prediction_end - prediction_start
        request_latency = time.time() - request_start
        
        # Log metrics
        log_prediction_metrics(model_name, model_version, prediction_latency_value)
        
        # Format response
        response = {
            'request_id': request_id,
            'prediction': float(prediction[0]),
            'model_name': model_name,
            'model_version': model_version,
            'deployment_type': DEPLOYMENT_TYPE,
            'prediction_time': prediction_latency_value,
            'request_time': request_latency,
            'timestamp': datetime.now().isoformat()
        }
        
        # If this is a shadow deployment, compare with production prediction
        if DEPLOYMENT_TYPE == 'shadow' and 'compare_with_production' in request.args:
            try:
                # Load production model
                prod_model, prod_metadata = load_model(model_name, 'latest')
                prod_version = prod_metadata.get('version', 'unknown')
                
                # Make prediction with production model
                prod_prediction = prod_model.predict(X)
                
                # Add to response
                response['production_prediction'] = float(prod_prediction[0])
                response['production_model_version'] = prod_version
                response['prediction_difference'] = float(prediction[0] - prod_prediction[0])
            except Exception as e:
                logger.error(f"Error comparing with production model: {str(e)}")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        traceback.print_exc()
        
        # Log error metrics
        if 'model_name' in locals() and 'model_version' in locals():
            log_prediction_metrics(model_name, model_version, time.time() - request_start, str(e))
        
        return jsonify({
            'request_id': request_id,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Make predictions for a batch of input data."""
    request_id = str(uuid.uuid4())
    request_start = time.time()
    
    try:
        # Get input data
        data = request.json
        batch_data = data.get('data', [])
        model_name = data.get('model_name', 'parking_occupancy')
        
        # Load the appropriate model
        model, metadata, model_version = get_model_for_deployment(model_name)
        
        # Convert to DataFrame
        df = pd.DataFrame(batch_data)
        
        # Add timestamp if not present
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now().isoformat()
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Get feature list from metadata
        features = metadata.get('features', [])
        
        # Ensure all required features are present
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            
            # Fill with defaults if needed
            for feature in missing_features:
                df[feature] = 0
        
        # Handle post-processing artifacts if needed
        scaler_data = None
        artifacts = metadata.get('artifacts', {})
        if 'scaler' in artifacts:
            # This assumes the scaler is in the artifacts directory
            artifact_path = os.path.join(
                versioning.registry.models_path, 
                metadata.get('model_id', ''), 
                'artifacts', 
                'scaler.pkl'
            )
            if os.path.exists(artifact_path):
                scaler_data = joblib.load(artifact_path)
                # Apply scaling to features
                num_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
                if num_features:
                    df[num_features] = scaler_data.transform(df[num_features])
        
        # Prepare features for prediction
        X = df[features]
        
        # Make predictions
        prediction_start = time.time()
        predictions = model.predict(X)
        prediction_end = time.time()
        
        # Calculate latency
        prediction_latency_value = prediction_end - prediction_start
        request_latency = time.time() - request_start
        
        # Log metrics
        log_prediction_metrics(model_name, model_version, prediction_latency_value)
        
        # Format predictions
        results = []
        for i in range(len(predictions)):
            results.append({
                'index': i,
                'prediction': float(predictions[i]),
                'input_data': batch_data[i] if i < len(batch_data) else {}
            })
        
        # Format response
        response = {
            'request_id': request_id,
            'predictions': results,
            'count': len(results),
            'model_name': model_name,
            'model_version': model_version,
            'deployment_type': DEPLOYMENT_TYPE,
            'prediction_time': prediction_latency_value,
            'request_time': request_latency,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        traceback.print_exc()
        
        # Log error metrics
        if 'model_name' in locals() and 'model_version' in locals():
            log_prediction_metrics(model_name, model_version, time.time() - request_start, str(e))
        
        return jsonify({
            'request_id': request_id,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 400

@app.route('/feedback', methods=['POST'])
def record_feedback():
    """Record feedback with ground truth for a prediction."""
    try:
        data = request.json
        request_id = data.get('request_id')
        model_name = data.get('model_name')
        model_version = data.get('model_version')
        prediction = data.get('prediction')
        ground_truth = data.get('ground_truth')
        features = data.get('features', {})
        
        if not all([request_id, model_name, prediction, ground_truth]):
            return jsonify({
                'error': 'Missing required fields: request_id, model_name, prediction, ground_truth'
            }), 400
        
        # Record feedback for metrics
        if ENABLE_METRICS:
            update_ground_truth_metrics(
                model_name, 
                model_version, 
                np.array([ground_truth]), 
                np.array([prediction])
            )
        
        # Store the feedback in a file for future analysis
        feedback_dir = os.path.join(PRODUCTION_MODELS_PATH, 'feedback')
        os.makedirs(feedback_dir, exist_ok=True)
        
        feedback_file = os.path.join(feedback_dir, f"feedback_{model_name}.jsonl")
        
        with open(feedback_file, 'a') as f:
            feedback_data = {
                'request_id': request_id,
                'model_name': model_name,
                'model_version': model_version,
                'prediction': prediction,
                'ground_truth': ground_truth,
                'features': features,
                'timestamp': datetime.now().isoformat()
            }
            f.write(json.dumps(feedback_data) + '\n')
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback recorded'
        })
    
    except Exception as e:
        logger.error(f"Error recording feedback: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e)
        }), 400

@app.route('/models', methods=['GET'])
def list_models():
    """List available models."""
    try:
        # Get models from registry
        registry_models = versioning.registry.list_models()
        
        # Get production models
        production_models = versioning.list_production_models()
        
        # Format response
        response = {
            'registry_models': registry_models,
            'production_models': production_models,
            'total_registry_models': len(registry_models),
            'total_production_models': len(production_models),
            'deployment_type': DEPLOYMENT_TYPE
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 400

if __name__ == '__main__':
    # Log startup info
    logger.info(f"Starting prediction service in {DEPLOYMENT_TYPE} mode")
    logger.info(f"Model registry path: {MODEL_REGISTRY_PATH}")
    logger.info(f"Production models path: {PRODUCTION_MODELS_PATH}")
    
    # Start the Flask app
    app.run(debug=False, host='0.0.0.0', port=5000) 