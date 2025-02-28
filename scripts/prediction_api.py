#!/usr/bin/env python
"""
API for serving parking occupancy predictions.
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from your existing modules
from scripts.parking_sim.advanced_features import engineer_advanced_features

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load models
MODEL_DIR = "production_models"
GLOBAL_MODEL_PATH = os.path.join(MODEL_DIR, "global_model_advanced_features.pkl")
LOCATION_MODELS = {}

# Load global model
if os.path.exists(GLOBAL_MODEL_PATH):
    GLOBAL_MODEL = joblib.load(GLOBAL_MODEL_PATH)
    logger.info(f"Loaded global model from {GLOBAL_MODEL_PATH}")
else:
    GLOBAL_MODEL = None
    logger.warning(f"Global model not found at {GLOBAL_MODEL_PATH}")

# Load location-specific models
if os.path.exists(MODEL_DIR):
    for model_file in os.listdir(MODEL_DIR):
        if model_file.startswith("location_") and model_file.endswith(".pkl"):
            location_id = model_file.split("_")[1]
            model_path = os.path.join(MODEL_DIR, model_file)
            LOCATION_MODELS[location_id] = joblib.load(model_path)
            logger.info(f"Loaded model for location {location_id} from {model_path}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'global_model': GLOBAL_MODEL is not None,
        'location_models': list(LOCATION_MODELS.keys())
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction based on input data."""
    try:
        # Get input data
        data = request.json
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Add timestamp if not present
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now().isoformat()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Generate advanced features
        df_advanced = engineer_advanced_features(df)
        
        # Determine which model to use
        location_id = data.get('location_id')
        model = None
        model_type = "global"
        
        if location_id and location_id in LOCATION_MODELS:
            model = LOCATION_MODELS[location_id]
            model_type = f"location_{location_id}"
        elif GLOBAL_MODEL is not None:
            model = GLOBAL_MODEL
        else:
            return jsonify({
                'error': 'No suitable model found'
            }), 404
        
        # Prepare features
        exclude_cols = ['timestamp', 'date', 'occupancy']
        X = df_advanced.drop(columns=[col for col in exclude_cols if col in df_advanced.columns])
        
        # Make prediction
        prediction = float(model.predict(X)[0])
        
        # Return prediction
        return jsonify({
            'prediction': prediction,
            'model_used': model_type
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Make predictions for a batch of input data."""
    try:
        # Get input data
        data = request.json
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Generate advanced features
        df_advanced = engineer_advanced_features(df)
        
        # Prepare features
        exclude_cols = ['timestamp', 'date', 'occupancy']
        X = df_advanced.drop(columns=[col for col in exclude_cols if col in df_advanced.columns])
        
        # Make predictions
        predictions = []
        
        # Group by location if location_id is present
        if 'location_id' in df.columns:
            for location_id, group in df.groupby('location_id'):
                # Get indices for this location
                indices = group.index
                
                # Determine which model to use
                if location_id in LOCATION_MODELS:
                    model = LOCATION_MODELS[location_id]
                    model_type = f"location_{location_id}"
                elif GLOBAL_MODEL is not None:
                    model = GLOBAL_MODEL
                    model_type = "global"
                else:
                    model = None
                    model_type = "none"
                
                if model is not None:
                    # Make predictions for this location
                    X_loc = X.loc[indices]
                    loc_preds = model.predict(X_loc)
                    
                    # Add predictions to results
                    for i, idx in enumerate(indices):
                        predictions.append({
                            'index': int(idx),
                            'prediction': float(loc_preds[i]),
                            'model_used': model_type
                        })
                else:
                    # No model available for this location
                    for idx in indices:
                        predictions.append({
                            'index': int(idx),
                            'prediction': None,
                            'model_used': "none",
                            'error': f"No model available for location {location_id}"
                        })
        else:
            # No location_id, use global model for all
            if GLOBAL_MODEL is not None:
                preds = GLOBAL_MODEL.predict(X)
                
                for i in range(len(preds)):
                    predictions.append({
                        'index': i,
                        'prediction': float(preds[i]),
                        'model_used': "global"
                    })
            else:
                return jsonify({
                    'error': 'Global model not available'
                }), 404
        
        # Return predictions
        return jsonify({
            'predictions': predictions
        })
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 