from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import os

import numpy as np
from sklearn.linear_model import LinearRegression


Number = Union[int, float]
FeatureDict = Dict[str, Any]
XInput = Union[Sequence[Number], FeatureDict]
XArray = Sequence[XInput]


def _is_mapping(obj: Any) -> bool:
    return isinstance(obj, dict)


@dataclass
class SatisfactionPredictor:
    """
    Simple satisfaction predictor using LinearRegression.
    Assumes target scale is 0..5 and clamps predictions into this range.
    """

    model: Optional[LinearRegression] = None
    feature_names: Optional[List[str]] = field(default=None, init=False)
    _is_fitted: bool = field(default=False, init=False)
    _cat_maps: Optional[Dict[str, Dict[Any, int]]] = field(default=None, init=False)

    def _infer_feature_names(self, X: XArray) -> List[str]:
        if not X:
            raise ValueError("Training data cannot be empty")

        first = X[0]
        if _is_mapping(first):
            # consistent order for dict-based features
            return sorted(first.keys())
        else:
            # numeric vector indices as synthetic names
            return [f"f{i}" for i in range(len(first))]

    def _detect_categorical(self, X: XArray, feature_names: List[str]) -> List[str]:
        # Detect categorical features by type in first row
        first = X[0]
        cat_feats = []
        if _is_mapping(first):
            for k in feature_names:
                v = first[k]
                if isinstance(v, str):
                    cat_feats.append(k)
        return cat_feats

    def _fit_categorical_maps(
        self, X: XArray, feature_names: List[str]
    ) -> Dict[str, Dict[Any, int]]:
        cat_feats = self._detect_categorical(X, feature_names)
        cat_maps: Dict[str, Dict[Any, int]] = {}
        for feat in cat_feats:
            values = set(row[feat] for row in X)
            cat_maps[feat] = {val: idx for idx, val in enumerate(sorted(values))}
        return cat_maps

    def _extract_features(
        self, X: XArray, feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Convert heterogeneous inputs into a 2D float array with columns aligned
        to feature_names. For dict inputs, missing keys raise a ValueError
        matching scikit-learn's semantics in tests.
        """
        if feature_names is None:
            feature_names = self.feature_names
        if feature_names is None:
            raise ValueError("feature_names must be set before extracting features")

        rows: List[List[float]] = []
        if not X:
            return np.zeros((0, len(feature_names)), dtype=float)

        # Handle single dict or sequence
        if isinstance(X, dict):
            X = [X]
        elif isinstance(X, (list, tuple)) and (
            len(X) > 0 and isinstance(X[0], (int, float))
        ):
            X = [X]

        cat_maps = getattr(self, "_cat_maps", None)
        for row in X:  # type: ignore[index]
            if _is_mapping(row):
                missing = [k for k in feature_names if k not in row]
                if missing:
                    msg = (
                        "The feat. names should match those passed during fit.\n"
                        "Feature names seen at fit time, yet now missing:\n- "
                    )
                    msg += "\n- ".join(missing)
                    raise ValueError(msg)
                vals = []
                for k in feature_names:
                    v = row[k]
                    if cat_maps and k in cat_maps:
                        try:
                            vals.append(float(cat_maps[k][v]))
                        except KeyError:
                            # Unseen category: assign a new index
                            cat_maps[k][v] = len(cat_maps[k])
                            vals.append(float(cat_maps[k][v]))
                    else:
                        try:
                            vals.append(float(v))
                        except Exception:
                            raise ValueError(
                                f"Feature '{k}' value '{v}' is not numeric and "
                                "not categorical"
                            )
                rows.append(vals)
            else:
                # Sequence input
                try:
                    vals = [float(v) for v in row]
                except Exception:
                    raise ValueError("All feature values must be numeric")
                rows.append(vals)

        return np.asarray(rows, dtype=float)

    def fit(self, X: XArray, y: Sequence[Number]) -> "SatisfactionPredictor":
        # Input validation
        if X is None or y is None:
            raise ValueError("X and y must not be None")

        if not isinstance(y, Sequence) or len(y) == 0:
            raise ValueError("Training data cannot be empty")

        if not isinstance(X, Sequence) or len(X) == 0:
            raise ValueError("Training data cannot be empty")

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        # Validate target bounds (0..5)
        y_list = list(float(v) for v in y)
        if any((v < 0 or v > 5) for v in y_list):
            raise ValueError("Target values must be between 0 and 5")

        # Determine feature names on first fit or recompute for consistency
        self.feature_names = self._infer_feature_names(X)

        # Fit categorical mappings
        self._cat_maps = self._fit_categorical_maps(X, self.feature_names)

        # Vectorize features
        X_processed = self._extract_features(X, self.feature_names)

        # Fresh model per fit to ensure overwrite behavior
        self.model = LinearRegression()
        self.model.fit(X_processed, y_list)
        self._is_fitted = True
        return self

    def predict(self, X: XArray) -> Union[Number, List[Number]]:
        if self.model is None or self.feature_names is None or not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Single item handling
        if isinstance(X, dict) or (
            isinstance(X, (list, tuple))
            and (len(X) > 0 and isinstance(X[0], (int, float)))
        ):
            X_list = [X]  # type: ignore[list-item]
        else:
            X_list = list(X)  # type: ignore[call-arg]

        X_processed = self._extract_features(X_list, self.feature_names)
        preds = self.model.predict(X_processed)

        # Clamp predictions into [0, 5]
        preds = np.clip(preds, 0.0, 5.0)

        if len(preds) == 1:
            return float(preds)
        return [float(p) for p in preds]

    def save_model(self, path: str) -> None:
        if self.model is None or self.feature_names is None or not self._is_fitted:
            raise ValueError("Model must be fitted before saving")

        data = {
            "coef": self.model.coef_.tolist(),
            "intercept": float(self.model.intercept_),
            "feature_names": self.feature_names,
            "cat_maps": self._cat_maps,
        }
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load_model(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        try:
            import json

            with open(path, "r", encoding="utf-8") as f:
                model_data = json.load(f)
        except Exception as e:
            raise IOError(f"Failed to load model: {e}") from e

        required_keys = {"coef", "intercept", "feature_names"}
        if not required_keys.issubset(model_data):
            raise ValueError("Invalid model file format")

        self.feature_names = list(model_data["feature_names"])
        self._cat_maps = model_data.get("cat_maps", None)
        self.model = LinearRegression()
        self.model.coef_ = np.asarray(model_data["coef"], dtype=float)
        self.model.intercept_ = float(model_data["intercept"])
        self._is_fitted = True
