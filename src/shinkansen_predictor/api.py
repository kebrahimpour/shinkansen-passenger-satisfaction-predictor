"""FastAPI application for the Shinkansen Passenger Satisfaction Predictor.

This module defines the API endpoints for:
- Health checks.
- Making predictions using the trained model.
"""

import os
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .predictor import SatisfactionPredictor

# Use a dictionary to hold the application's state
ml_models: Dict[str, SatisfactionPredictor] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages loading the ML model on startup and cleanup on shutdown."""
    model_path = os.environ.get("MODEL_PATH", "model.pkl")
    ml_models["predictor"] = SatisfactionPredictor()
    if os.path.exists(model_path):
        ml_models["predictor"].load_model(model_path)
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(
    lifespan=lifespan,
    title="Shinkansen Passenger Satisfaction Predictor",
    description="API to predict passenger satisfaction scores.",
    version="1.0.0",
)


def get_predictor() -> SatisfactionPredictor:
    """Dependency function to get the predictor model."""
    return ml_models["predictor"]


class PredictionRequest(BaseModel):
    """Defines the schema for a prediction request."""

    duration: int = Field(..., example=120)
    service_class: str = Field(..., example="Green")
    on_time_performance: float = Field(..., example=0.95)
    weather_condition: str = Field(..., example="clear")
    seat_occupancy: float = Field(..., example=0.8)

    @field_validator("service_class")
    def validate_service_class(cls, v):
        allowed = {"Ordinary", "Green", "GranClass"}
        if v not in allowed:
            raise ValueError(f"service_class must be one of {allowed}")
        return v


class PredictionResponse(BaseModel):
    """Defines the schema for a prediction response."""

    prediction: float = Field(..., example=4.2)


@app.get("/")
def read_root() -> Dict[str, str]:
    """Root endpoint providing a welcome message."""
    return {"message": "Welcome to the Shinkansen Predictor API"}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    predictor: SatisfactionPredictor = Depends(get_predictor),
) -> PredictionResponse:
    """Endpoint to make a satisfaction prediction."""
    if not predictor or not predictor._is_fitted:
        raise HTTPException(status_code=503, detail="Model is not fitted or available")

    prediction_result = predictor.predict(request.model_dump())
    return PredictionResponse(prediction=prediction_result)
