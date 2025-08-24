"""Unit tests for SatisfactionPredictor class.

This module contains comprehensive tests covering:
- Model fitting and prediction
- Model saving and loading
- Error handling and edge cases
- Data validation
"""

import os
import tempfile

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from shinkansen_predictor import SatisfactionPredictor


class TestSatisfactionPredictor:
    """Test suite for SatisfactionPredictor class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.predictor = SatisfactionPredictor()
        self.sample_X = [
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
        self.sample_y = [4.5, 3.8, 4.9, 4.1]

    def test_init(self):
        """Test SatisfactionPredictor initialization."""
        predictor = SatisfactionPredictor()
        assert predictor is not None
        assert not predictor._is_fitted
        assert predictor.model is None
        assert predictor.feature_names is None

    def test_fit_basic(self):
        """Test basic model fitting functionality."""
        self.predictor.fit(self.sample_X, self.sample_y)

        assert self.predictor._is_fitted
        assert self.predictor.model is not None
        assert self.predictor.feature_names is not None
        assert len(self.predictor.feature_names) > 0

    def test_fit_empty_data(self):
        """Test fitting with empty data raises appropriate error."""
        with pytest.raises(ValueError, match="Training data cannot be empty"):
            self.predictor.fit([], [])

    def test_fit_mismatched_lengths(self):
        """Test fitting with mismatched X and y lengths."""
        with pytest.raises(ValueError, match="X and y must have the same length"):
            self.predictor.fit(self.sample_X, [1, 2])  # Wrong length

    def test_fit_invalid_target_values(self):
        """Test fitting with invalid target values."""
        invalid_y = [-1, 6, 2.5, 3.0]  # Values outside 0-5 range
        with pytest.raises(ValueError, match="Target values must be between 0 and 5"):
            self.predictor.fit(self.sample_X, invalid_y)

    def test_predict_basic(self):
        """Test basic prediction functionality."""
        self.predictor.fit(self.sample_X, self.sample_y)

        test_data = {
            "duration": 100,
            "service_class": "Green",
            "on_time_performance": 0.93,
            "weather_condition": "clear",
            "seat_occupancy": 0.75,
        }

        prediction = self.predictor.predict(test_data)

        assert isinstance(prediction, (int, float))
        assert 0 <= prediction <= 5

    def test_predict_unfitted_model(self):
        """Test prediction with unfitted model raises error."""
        test_data = {"duration": 100, "service_class": "Green"}

        with pytest.raises(
            ValueError, match="Model must be fitted before making prediction"
        ):
            self.predictor.predict(test_data)

    def test_predict_missing_features(self):
        """Test prediction with missing features."""
        self.predictor.fit(self.sample_X, self.sample_y)

        incomplete_data = {"duration": 100}  # Missing other features

        with pytest.raises(ValueError):
            self.predictor.predict(incomplete_data)

    def test_predict_invalid_service_class(self):
        """Test prediction with invalid service class."""
        self.predictor.fit(self.sample_X, self.sample_y)

        invalid_data = {
            "duration": 100,
            "service_class": "InvalidClass",
            "on_time_performance": 0.93,
            "weather_condition": "clear",
            "seat_occupancy": 0.75,
        }

        # Should handle gracefully - model should still make prediction
        prediction = self.predictor.predict(invalid_data)
        assert isinstance(prediction, (int, float))

    def test_save_model(self):
        """Test model saving functionality."""
        self.predictor.fit(self.sample_X, self.sample_y)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
            tmp_path = tmp_file.name

        try:
            self.predictor.save_model(tmp_path)
            assert os.path.exists(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_save_unfitted_model(self):
        """Test saving unfitted model raises error."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
            tmp_path = tmp_file.name

        try:
            with pytest.raises(ValueError, match="Model must be fitted before saving"):
                self.predictor.save_model(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_load_model(self):
        """Test model loading functionality."""
        # First fit and save a model
        self.predictor.fit(self.sample_X, self.sample_y)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
            tmp_path = tmp_file.name

        try:
            self.predictor.save_model(tmp_path)

            # Create new predictor and load the model
            new_predictor = SatisfactionPredictor()
            new_predictor.load_model(tmp_path)

            assert new_predictor._is_fitted
            assert new_predictor.model is not None
            assert new_predictor.feature_names is not None

            # Test that loaded model can make predictions
            test_data = {
                "duration": 100,
                "service_class": "Green",
                "on_time_performance": 0.93,
                "weather_condition": "clear",
                "seat_occupancy": 0.75,
            }
            prediction = new_predictor.predict(test_data)
            assert isinstance(prediction, (int, float))

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_load_nonexistent_model(self):
        """Test loading non-existent model file raises error."""
        with pytest.raises(FileNotFoundError):
            self.predictor.load_model("nonexistent_model.pkl")

    def test_load_invalid_model_file(self):
        """Test loading invalid model file raises error."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(b"invalid pickle data")

        try:
            with pytest.raises(Exception):  # Could be pickle.UnpicklingError or similar
                self.predictor.load_model(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_feature_extraction(self):
        """Test internal feature extraction consistency."""
        self.predictor.fit(self.sample_X, self.sample_y)

        # Test that the same data produces consistent features
        test_data = self.sample_X[0]

        # Extract features multiple times
        features1 = self.predictor._extract_features(test_data)
        features2 = self.predictor._extract_features(test_data)

        assert np.array_equal(features1, features2)
        assert features1.shape[1] == len(self.predictor.feature_names)

    def test_prediction_consistency(self):
        """Test that predictions are consistent for the same input."""
        self.predictor.fit(self.sample_X, self.sample_y)

        test_data = {
            "duration": 100,
            "service_class": "Green",
            "on_time_performance": 0.93,
            "weather_condition": "clear",
            "seat_occupancy": 0.75,
        }

        # Make multiple predictions with the same data
        pred1 = self.predictor.predict(test_data)
        pred2 = self.predictor.predict(test_data)

        assert pred1 == pred2

    def test_prediction_bounds(self):
        """Test that all predictions are within valid bounds."""
        self.predictor.fit(self.sample_X, self.sample_y)

        # Test various edge cases
        edge_cases = [
            {
                "duration": 1,
                "service_class": "Ordinary",
                "on_time_performance": 0.0,
                "weather_condition": "storm",
                "seat_occupancy": 1.0,
            },
            {
                "duration": 1000,
                "service_class": "GranClass",
                "on_time_performance": 1.0,
                "weather_condition": "clear",
                "seat_occupancy": 0.1,
            },
        ]

        for case in edge_cases:
            prediction = self.predictor.predict(case)
            assert (
                0 <= prediction <= 5
            ), f"Prediction {prediction} out of bounds for case {case}"

    @patch("pickle.dump")
    def test_save_model_io_error(self, mock_dump):
        """Test save model handles IO errors gracefully."""
        self.predictor.fit(self.sample_X, self.sample_y)
        mock_dump.side_effect = IOError("Disk full")

        with pytest.raises(IOError):
            self.predictor.save_model("test.pkl")

    @patch("shinkansen_predictor.predictor.os.path.exists")
    @patch("shinkansen_predictor.predictor.pickle.load")
    @patch("builtins.open")
    def test_load_model_io_error(self, mock_open, mock_load, mock_exists):
        """Test load model handles IO errors gracefully."""
        mock_exists.return_value = True
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None
        mock_load.side_effect = IOError("File corrupted")

        with pytest.raises(IOError):
            self.predictor.load_model("test.pkl")

    def test_model_attributes_after_fit(self):
        """Test that model attributes are properly set after fitting."""
        assert (
            not hasattr(self.predictor, "_is_fitted") or not self.predictor._is_fitted
        )

        self.predictor.fit(self.sample_X, self.sample_y)

        assert self.predictor._is_fitted
        assert hasattr(self.predictor, "model")
        assert hasattr(self.predictor, "feature_names")
        assert self.predictor.model is not None
        assert self.predictor.feature_names is not None

    def test_multiple_fits_overwrite(self):
        """Test that multiple fits properly overwrite the previous model."""
        self.predictor.fit(self.sample_X, self.sample_y)
        first_model = self.predictor.model

        # Fit again with different data
        new_X = [
            {
                "duration": 200,
                "service_class": "Green",
                "on_time_performance": 0.95,
                "weather_condition": "clear",
                "seat_occupancy": 0.8,
            }
        ]
        new_y = [4.0]

        self.predictor.fit(new_X, new_y)
        second_model = self.predictor.model

        # Models should be different objects
        assert first_model is not second_model
        assert self.predictor._is_fitted

    def test_predict_without_fit_raises(self):
        p = SatisfactionPredictor()
        with pytest.raises(
            ValueError, match="Model must be fitted before making predictions"
        ):
            p.predict([1, 2, 3])


# Standalone test functions (move these OUTSIDE the class)
def test_extract_features_dict_with_string_values():
    """Test _extract_features with dict input containing string values."""
    predictor = SatisfactionPredictor()
    feature_names = [
        "duration",
        "service_class",
        "on_time_performance",
        "weather_condition",
        "seat_occupancy",
    ]
    X = [
        {
            "duration": 120,
            "service_class": "Green",
            "on_time_performance": 0.95,
            "weather_condition": "clear",
            "seat_occupancy": 0.8,
        }
    ]
    # Should raise ValueError as "service_class" and "weather_condition" are strings
    with pytest.raises(ValueError):
        predictor._extract_features(X, feature_names)


def test_extract_features_dict_with_all_numeric():
    """Test _extract_features with dict input containing only numeric values."""
    predictor = SatisfactionPredictor()
    feature_names = ["a", "b", "c"]
    X = [
        {"a": 1, "b": 2, "c": 3},
        {"a": 4, "b": 5, "c": 6},
    ]
    arr = predictor._extract_features(X, feature_names)
    assert arr.shape == (2, 3)
    assert (arr == [[1, 2, 3], [4, 5, 6]]).all()


def test_predict_unfitted_model_message():
    """Test error message for unfitted model matches expected string."""
    predictor = SatisfactionPredictor()
    with pytest.raises(
        ValueError, match="Model must be fitted before making predictions"
    ):
        predictor.predict([1, 2, 3])


def test_fit_with_non_numeric_dict_values():
    """Test fit raises ValueError if dict features contain non-numeric values."""
    predictor = SatisfactionPredictor()
    X = [
        {"a": 1, "b": "foo", "c": 3},
        {"a": 4, "b": 5, "c": 6},
    ]
    y = [1, 2]
    with pytest.raises(ValueError):
        predictor.fit(X, y)


def test_fit_with_numeric_sequence():
    """Test fit works with numeric sequence input."""
    predictor = SatisfactionPredictor()
    X = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    y = [1, 2]
    predictor.fit(X, y)
    assert predictor.model is not None
    assert predictor.feature_names == ["f0", "f1", "f2"]


def test_extract_features_empty_feature_names():
    """Test _extract_features with empty feature_names returns correct shape."""
    predictor = SatisfactionPredictor()
    arr = predictor._extract_features([], [])
    assert arr.shape == (0, 0)


def test_load_model_file_not_found():
    """Test load_model raises FileNotFoundError."""
    predictor = SatisfactionPredictor()
    with pytest.raises(FileNotFoundError):
        predictor.load_model("does_not_exist.json")


def test_save_model_unfitted():
    """Test save_model raises ValueError if model is not fitted."""
    predictor = SatisfactionPredictor()
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        with pytest.raises(ValueError, match="Model must be fitted before saving"):
            predictor.save_model(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_extract_features_dict_missing_key():
    predictor = SatisfactionPredictor()
    feature_names = ["a", "b", "c"]
    X = [
        {"a": 1, "b": 2},
        {"a": 4, "b": 5, "c": 6},
    ]
    with pytest.raises(
        KeyError,
        match=(
            r"(?s).*The feature names should match those that were passed"
            r".*during fit.*"
        ),
    ):
        predictor._extract_features(X, feature_names)


def test_extract_features_sequence():
    predictor = SatisfactionPredictor()
    feature_names = ["f0", "f1", "f2"]
    X = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    arr = predictor._extract_features(X, feature_names)
    assert arr.shape == (2, 3)
    assert (arr == [[1, 2, 3], [4, 5, 6]]).all()


def test_extract_features_empty():
    predictor = SatisfactionPredictor()
    feature_names = ["a", "b", "c"]
    arr = predictor._extract_features([], feature_names)
    assert arr.shape == (0, 3)


def test_extract_features_single_dict():
    predictor = SatisfactionPredictor()
    feature_names = ["a", "b", "c"]
    X = {"a": 7, "b": 8, "c": 9}
    arr = predictor._extract_features([X], feature_names)
    assert arr.shape == (1, 3)
    assert (arr == [[7, 8, 9]]).all()


def test_extract_features_single_sequence():
    predictor = SatisfactionPredictor()
    feature_names = ["f0", "f1", "f2"]
    X = [7, 8, 9]
    arr = predictor._extract_features([X], feature_names)
    assert arr.shape == (1, 3)
    assert (arr == [[7, 8, 9]]).all()


def test_extract_features_non_numeric_sequence():
    predictor = SatisfactionPredictor()
    feature_names = ["f0", "f1", "f2"]
    X = [["a", "b", "c"]]
    with pytest.raises(ValueError):
        predictor._extract_features(X, feature_names)


def test_extract_features_feature_names_mismatch():
    predictor = SatisfactionPredictor()
    feature_names = ["a", "b", "d"]
    X = [
        {"a": 1, "b": 2, "c": 3},
        {"a": 4, "b": 5, "c": 6},
    ]
    with pytest.raises(KeyError):
        predictor._extract_features(X, feature_names)
