"""
FastAPI Application for Model Monitoring Dashboard
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel

from src.core.config import settings
from src.features.model_performance.business_logic.metrics_calculator import ModelMetricsCalculator
from src.features.model_performance.business_logic.data_quality import DataQualityMonitor
from src.features.model_performance.business_logic.health_monitor import HealthMonitor
from src.features.model_performance.data_access.metrics_repository import MetricsRepository

# Configure logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Model Monitoring API",
    description="API for model performance monitoring and metrics tracking",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class MetricResponse(BaseModel):
    model_id: str
    metric_name: str
    value: float
    timestamp: datetime
    confidence_interval: Optional[tuple[float, float]]
    metadata: Optional[Dict]

class DataQualityResponse(BaseModel):
    model_id: str
    timestamp: datetime
    missing_rate: float
    out_of_range_rate: float
    correlation_changes: Dict[str, float]
    distribution_metrics: Dict[str, Dict[str, float]]
    sample_size: int

class HealthResponse(BaseModel):
    model_id: str
    timestamp: datetime
    status: str
    metrics: Dict[str, float]
    alerts: List[Dict]

# Dependencies
async def get_metrics_repository():
    # In a real app, you would initialize this with proper DB connection
    return MetricsRepository(settings.database)

async def get_metrics_calculator(model_id: str):
    return ModelMetricsCalculator(model_id)

# Routes
@app.get("/api/v1/metrics/{model_id}", response_model=List[MetricResponse])
async def get_model_metrics(
    model_id: str,
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    metric_names: Optional[List[str]] = Query(None),
    repository: MetricsRepository = Depends(get_metrics_repository)
):
    """Get model performance metrics"""
    try:
        metrics = await repository.get_metric_history(
            model_id=model_id,
            start_time=start_time or (datetime.now() - timedelta(days=7)),
            end_time=end_time or datetime.now(),
            metric_names=metric_names
        )
        return metrics
    except Exception as e:
        logger.error(f"Error fetching metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/metrics/{model_id}")
async def store_model_metrics(
    model_id: str,
    metrics: Dict[str, float],
    repository: MetricsRepository = Depends(get_metrics_repository),
    calculator: ModelMetricsCalculator = Depends(get_metrics_calculator)
):
    """Store new model metrics"""
    try:
        metric_results = {}
        for name, value in metrics.items():
            metric_results[name] = calculator.create_metric_result(value)
        
        for name, result in metric_results.items():
            await repository.store_metric(model_id, name, result)
        
        return {"status": "success", "stored_metrics": len(metrics)}
    except Exception as e:
        logger.error(f"Error storing metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/quality/{model_id}", response_model=DataQualityResponse)
async def get_data_quality(
    model_id: str,
    repository: MetricsRepository = Depends(get_metrics_repository)
):
    """Get data quality metrics"""
    try:
        quality_metrics = await repository.get_latest_quality_metrics(model_id)
        return quality_metrics
    except Exception as e:
        logger.error(f"Error fetching data quality metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health/{model_id}", response_model=HealthResponse)
async def get_model_health(
    model_id: str,
    repository: MetricsRepository = Depends(get_metrics_repository)
):
    """Get model health status"""
    try:
        health_metrics = await repository.get_latest_health_metrics(model_id)
        return health_metrics
    except Exception as e:
        logger.error(f"Error fetching health metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models")
async def list_monitored_models(
    repository: MetricsRepository = Depends(get_metrics_repository)
):
    """Get list of all monitored models"""
    try:
        models = await repository.get_monitored_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error fetching model list: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """API health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()} 