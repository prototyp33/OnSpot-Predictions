"""Main FastAPI application for OnSpot Predictive Model."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from onspot.utils.config import load_config
from onspot.api.routes import weather

# Load configuration
config = load_config()

# Create FastAPI application
app = FastAPI(
    title="OnSpot Predictive Model API",
    description="API for parking occupancy prediction and weather impact analysis",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(weather.router)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"} 