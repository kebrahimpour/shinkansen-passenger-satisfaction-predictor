"""FastAPI serving scaffold for Shinkansen passenger satisfaction prediction.

This module provides a REST API endpoint for predicting passenger satisfaction
scores using the trained SatisfactionPredictor model. It includes automatic
Swagger/OpenAPI documentation and model loading via pickle.
"""
import os
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from .predictor import SatisfactionPredictor


# Pydantic models for request/response validation
class JourneyData(BaseModel):
    """Journey data model for prediction requests."""

    duration: float = Field(
        ..., description="Journey duration in minutes", gt=0, example=120.0
    )
    service_class: str = Field(
        ..., description="Service class: Ordinary, Green, or GranClass", example="Green"
    )
    on_time_performance: Optional[float] = Field(
        0.95, description="On-time performance ratio (0-1)", ge=0, le=1, example=0.95
    )
    weather_condition: Optional[str] = Field(
        "clear", description="Weather condition during journey", example="clear"
    )
    seat_occupancy: Optional[float] = Field(
        0.8, description="Seat occupancy ratio (0-1)", ge=0, le=1, example=0.8
    )

    @validator("service_class")
    def validate_service_class(cls, v):
        """Validate service class values."""
        allowed_classes = ["Ordinary", "Green", "GranClass"]
        if v not in allowed_classes:
            raise ValueError(f"Service class must be one of: {allowed_classes}")
        return v

    class Config:
        """Pydantic configuration."""

        schema_extra = {
            "example": {
                "duration": 120.0,
                "service_class": "Green",
                "on_time_performance": 0.95,
                "weather_condition": "clear",
                "seat_occupancy": 0.8,
            }
        }


class PredictionResponse(BaseModel):
    """Response model for satisfaction predictions."""

    satisfaction_score: float = Field(
        ..., description="Predicted satisfaction score (0-5)", ge=0, le=5, example=4.2
    )
    model_version: str = Field(
        ..., description="Version of the model used for prediction", example="v1.0"
    )
    confidence: Optional[float] = Field(
        None,
        description="Prediction confidence (if available)",
        ge=0,
        le=1,
        example=0.85,
    )


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status", example="healthy")
    model_loaded: bool = Field(..., description="Whether model is loaded", example=True)
    version: str = Field(..., description="API version", example="1.0.0")


# FastAPI app initialization
app = FastAPI(
    title="Shinkansen Passenger Satisfaction Predictor API",
    description="""
    REST API for predicting passenger satisfaction on Shinkansen trains.

    This API provides a `/predict` endpoint that accepts journey data and returns
    a satisfaction score prediction using a trained machine learning model.

    ## Features
    - Real-time satisfaction prediction
    - Model loading via pickle
    - Automatic input validation
    - Swagger/OpenAPI documentation
    - Health check endpoint

    ## Example Usage
    ```bash
    curl -X POST "http://localhost:8000/predict" \
         -H "Content-Type: application/json" \
         -d '{"duration": 120, "service_class": "Green", "on_time_performance": 0.95}'
    ```
    """,
    version="1.0.0",
    contact={
        "name": "Shinkansen Predictor Team",
        "url": (
            "https://github.com/kebrahimpour/"
            "shinkansen-passenger-satisfaction-predictor"
        ),
    },
    license_info={
        "name": "CC0-1.0",
        "url": "https://creativecommons.org/publicdomain/zero/1.0/",
    },
)

# Global predictor instance
predictor: Optional[SatisfactionPredictor] = None


@app.on_event("startup")
async def load_model():
    """Load the trained model on startup."""
    global predictor

    # Look for model file in common locations
    model_paths = [
        "models/satisfaction_model.pkl",
        "../models/satisfaction_model.pkl",
        "../../models/satisfaction_model.pkl",
        "/app/models/satisfaction_model.pkl",
        os.getenv("MODEL_PATH", "models/satisfaction_model.pkl"),
    ]

    predictor = SatisfactionPredictor()

    # Try to load existing model
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                predictor.load_model(model_path)
                print(f"✅ Model loaded successfully from {model_path}")
                return
            except Exception as e:
                print(f"⚠️ Failed to load model from {model_path}: {e}")
                continue

    # If no model found, create a dummy trained model for demo purposes
    print("📝 No trained model found, creating demo model...")
    try:
        # Create sample training data
        X_train = [
            {
                "duration": 120,
                "service_class": "Green",
                "on_time_performance": 0.95,
                "weather_condition": "clear",
                "seat_occupancy": 0.8,
            },
            {
                "duration": 90,
                "service_class": "Ordinary",
                "on_time_performance": 0.88,
                "weather_condition": "rain",
                "seat_occupancy": 0.6,
            },
            {
                "duration": 180,
                "service_class": "GranClass",
                "on_time_performance": 0.99,
                "weather_condition": "clear",
                "seat_occupancy": 0.9,
            },
            {
                "duration": 60,
                "service_class": "Ordinary",
                "on_time_performance": 0.92,
                "weather_condition": "cloudy",
                "seat_occupancy": 0.7,
            },
        ]
        y_train = [4.5, 3.8, 4.9, 4.1]

        predictor.fit(X_train, y_train)
        print("✅ Demo model trained successfully")

        # Save the demo model
        os.makedirs("models", exist_ok=True)
        predictor.save_model("models/satisfaction_model.pkl")
        print("💾 Demo model saved to models/satisfaction_model.pkl")

    except Exception as e:
        print(f"❌ Failed to create demo model: {e}")
        predictor = None


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Shinkansen Passenger Satisfaction Predictor API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=(predictor is not None and predictor._is_fitted),
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_satisfaction(journey: JourneyData):
    """Predict passenger satisfaction score for a journey.

    Args:
        journey: Journey data including duration, service class, etc.

    Returns:
        Prediction response with satisfaction score and metadata.

    Raises:
        HTTPException: If model is not loaded or prediction fails.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs.",
        )

    if not predictor._is_fitted:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not trained. Please check server configuration.",
        )

    try:
        # Convert Pydantic model to dict for prediction
        journey_dict = journey.dict()

        # Make prediction
        satisfaction_score = predictor.predict(journey_dict)

        # Ensure score is within valid range
        satisfaction_score = max(0.0, min(5.0, float(satisfaction_score)))

        return PredictionResponse(
            satisfaction_score=satisfaction_score,
            model_version="v1.0",
            confidence=None,  # Could be extended with model confidence estimation
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": f"Validation error: {str(exc)}"},
    )


if __name__ == "__main__":
    # Run the API server
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
