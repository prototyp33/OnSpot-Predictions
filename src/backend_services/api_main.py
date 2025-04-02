"""
FastAPI implementation for serving parking occupancy predictions.
"""
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import os
import sys
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def engineer_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer advanced features for the model."""
    df = df.copy()
    
    # Extract time-based features if timestamp is present
    if "timestamp" in df.columns:
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6])
    
    # Calculate interaction features
    if "temperature" in df.columns and "humidity" in df.columns:
        df["temp_humidity_interaction"] = df["temperature"] * df["humidity"]
    
    if "precipitation" in df.columns and "wind_speed" in df.columns:
        df["weather_severity"] = df["precipitation"] * df["wind_speed"]
    
    return df

# Create FastAPI app
app = FastAPI(
    title="Parking Occupancy Prediction API",
    description="API for predicting parking occupancy",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths
MODEL_DIR = "production_models"
GLOBAL_MODEL_PATH = os.path.join(MODEL_DIR, "global_model_advanced_features.pkl")

# Pydantic models for request/response validation
class PredictionInput(BaseModel):
    """Input data for prediction."""
    location_id: Optional[str] = None
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    precipitation: Optional[float] = None
    wind_speed: Optional[float] = None
    day_of_week: Optional[int] = None
    hour_of_day: Optional[int] = None
    is_weekend: Optional[bool] = None
    is_holiday: Optional[bool] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "location_id": "downtown_1",
                "temperature": 22.5,
                "humidity": 65.0,
                "precipitation": 0.0,
                "wind_speed": 5.2,
                "day_of_week": 2,
                "hour_of_day": 14,
                "is_weekend": False,
                "is_holiday": False
            }
        }

class PredictionOutput(BaseModel):
    """Output data for prediction."""
    prediction: float
    model_used: str
    timestamp: datetime = Field(default_factory=datetime.now)

class BatchPredictionInput(BaseModel):
    """Input data for batch prediction."""
    data: List[PredictionInput]

class BatchPredictionOutput(BaseModel):
    """Output data for batch prediction."""
    predictions: List[Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.now)

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    global_model: bool
    location_models: List[str]
    timestamp: datetime = Field(default_factory=datetime.now)

# Model loading dependency
def get_models():
    """Load models as a dependency."""
    global_model = None
    location_models = {}
    
    # Load global model
    if os.path.exists(GLOBAL_MODEL_PATH):
        global_model = joblib.load(GLOBAL_MODEL_PATH)
        logger.info(f"Loaded global model from {GLOBAL_MODEL_PATH}")
    else:
        logger.warning(f"Global model not found at {GLOBAL_MODEL_PATH}")
    
    # Load location-specific models
    if os.path.exists(MODEL_DIR):
        for model_file in os.listdir(MODEL_DIR):
            if model_file.startswith("location_") and model_file.endswith(".pkl"):
                location_id = model_file.split("_")[1]
                model_path = os.path.join(MODEL_DIR, model_file)
                location_models[location_id] = joblib.load(model_path)
                logger.info(f"Loaded model for location {location_id} from {model_path}")
    
    return {"global_model": global_model, "location_models": location_models}

@app.get("/health", response_model=HealthResponse)
async def health_check(models: Dict = Depends(get_models)):
    """Health check endpoint."""
    return {
        "status": "healthy",
        "global_model": models["global_model"] is not None,
        "location_models": list(models["location_models"].keys()),
        "timestamp": datetime.now()
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput, models: Dict = Depends(get_models)):
    """Make a prediction based on input data."""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data.dict()])
        
        # Convert timestamp to datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Generate advanced features
        df_advanced = engineer_advanced_features(df)
        
        # Determine which model to use
        location_id = data.location_id
        model = None
        model_type = "global"
        
        if location_id and location_id in models["location_models"]:
            model = models["location_models"][location_id]
            model_type = f"location_{location_id}"
        elif models["global_model"] is not None:
            model = models["global_model"]
        else:
            raise HTTPException(
                status_code=404,
                detail="No suitable model found"
            )
        
        # Prepare features
        exclude_cols = ["timestamp", "date", "occupancy"]
        X = df_advanced.drop(columns=[col for col in exclude_cols if col in df_advanced.columns])
        
        # Make prediction
        prediction = float(model.predict(X)[0])
        
        # Return prediction
        return {
            "prediction": prediction,
            "model_used": model_type,
            "timestamp": datetime.now()
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@app.post("/batch_predict", response_model=BatchPredictionOutput)
async def batch_predict(data: BatchPredictionInput, models: Dict = Depends(get_models)):
    """Make predictions for a batch of input data."""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([item.dict() for item in data.data])
        
        # Convert timestamp to datetime if present
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Generate advanced features
        df_advanced = engineer_advanced_features(df)
        
        # Prepare features
        exclude_cols = ["timestamp", "date", "occupancy"]
        X = df_advanced.drop(columns=[col for col in exclude_cols if col in df_advanced.columns])
        
        # Make predictions
        predictions = []
        
        # Group by location if location_id is present
        if "location_id" in df.columns:
            for location_id, group in df.groupby("location_id"):
                # Get indices for this location
                indices = group.index
                
                # Determine which model to use
                if location_id in models["location_models"]:
                    model = models["location_models"][location_id]
                    model_type = f"location_{location_id}"
                elif models["global_model"] is not None:
                    model = models["global_model"]
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
                            "index": int(idx),
                            "prediction": float(loc_preds[i]),
                            "model_used": model_type
                        })
                else:
                    # No model available for this location
                    for idx in indices:
                        predictions.append({
                            "index": int(idx),
                            "prediction": None,
                            "model_used": "none",
                            "error": f"No model available for location {location_id}"
                        })
        else:
            # No location_id, use global model for all
            if models["global_model"] is not None:
                preds = models["global_model"].predict(X)
                
                for i in range(len(preds)):
                    predictions.append({
                        "index": i,
                        "prediction": float(preds[i]),
                        "model_used": "global"
                    })
            else:
                raise HTTPException(
                    status_code=404,
                    detail="Global model not available"
                )
        
        # Return predictions
        return {
            "predictions": predictions,
            "timestamp": datetime.now()
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        ) 