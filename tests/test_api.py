"""Unit tests for the FastAPI application."""

from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from shinkansen_predictor.api import app, get_predictor


def test_read_root():
    """Test the root endpoint returns a welcome message."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "welcome" in response.json().get("message", "").lower()


def test_predict_success():
    """Test the /predict endpoint with a valid, mocked predictor."""
    # Create a mock predictor
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = 4.2
    mock_predictor._is_fitted = True

    # Use FastAPI's dependency override to replace the model for this test
    app.dependency_overrides[get_predictor] = lambda: mock_predictor

    test_data = {
        "duration": 120,
        "service_class": "Green",
        "on_time_performance": 0.95,
        "weather_condition": "clear",
        "seat_occupancy": 0.8,
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=test_data)

    assert response.status_code == 200
    assert response.json() == {"prediction": 4.2}

    # Clear the override after the test
    app.dependency_overrides.clear()


def test_predict_invalid_data():
    """Test the /predict endpoint returns a 422 for incomplete data."""
    with TestClient(app) as client:
        test_data = {"duration": 120, "service_class": "Green"}
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422


def test_predict_unfitted_model():
    """Test the /predict endpoint when the model is not fitted."""
    mock_predictor = MagicMock()
    mock_predictor._is_fitted = False

    # Override the dependency with the unfitted mock
    app.dependency_overrides[get_predictor] = lambda: mock_predictor

    test_data = {
        "duration": 120,
        "service_class": "Green",
        "on_time_performance": 0.95,
        "weather_condition": "clear",
        "seat_occupancy": 0.8,
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=test_data)

    assert response.status_code == 503
    assert "model is not fitted" in response.json().get("detail", "").lower()

    # Clean up the override
    app.dependency_overrides.clear()
