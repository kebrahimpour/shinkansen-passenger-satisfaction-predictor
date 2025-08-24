"""
Optimized pipeline for training, tuning, and creating a submission.
- robust preprocessing (imputation, scaling, one-hot)
- automatic problem type detection (binary/multi-class/regression)
- RandomizedSearchCV hyperparameter tuning with cross-validation
- Refit on full data and save model & submission
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    make_scorer,
)
from xgboost import XGBClassifier, XGBRegressor

# Config
RANDOM_STATE = 42
N_ITER_SEARCH = 24
CV_FOLDS = 5
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def load_and_merge(train_travel_path, train_survey_path, test_travel_path, test_survey_path):
    tt = pd.read_csv(train_travel_path)
    ts = pd.read_csv(train_survey_path)
    ttest = pd.read_csv(test_travel_path)
    stest = pd.read_csv(test_survey_path)
    train = pd.merge(tt, ts, on="ID")
    test = pd.merge(ttest, stest, on="ID")
    return train, test


def basic_map_and_impute(train, test):
    # Impute Arrival_Delay_in_Mins with training median
    if "Arrival_Delay_in_Mins" in train.columns:
        med = train["Arrival_Delay_in_Mins"].median()
        train["Arrival_Delay_in_Mins"] = train["Arrival_Delay_in_Mins"].fillna(
            med)
        test["Arrival_Delay_in_Mins"] = test["Arrival_Delay_in_Mins"].fillna(
            med)

    # Map textual survey ratings -> numeric if present
    satisfaction_mapping = {
        "Extremely Poor": 0,
        "Poor": 1,
        "Needs Improvement": 2,
        "Acceptable": 3,
        "Good": 4,
        "Excellent": 5,
    }
    survey_cols = [
        "Seat_Comfort",
        "Arrival_Time_Convenient",
        "Catering",
        "Platform_Location",
        "Onboard_Wifi_Service",
        "Onboard_Entertainment",
        "Online_Support",
        "Ease_of_Online_Booking",
        "Onboard_Service",
        "Legroom",
        "Baggage_Handling",
        "CheckIn_Service",
        "Cleanliness",
        "Online_Boarding",
    ]
    for c in survey_cols:
        if c in train.columns:
            train[c] = train[c].map(satisfaction_mapping).astype(float)
            test[c] = test[c].map(satisfaction_mapping).astype(float)
            # fill missing with train median
            m = train[c].median()
            train[c] = train[c].fillna(m)
            test[c] = test[c].fillna(m)
    return train, test


def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # drop ID / target if present in numeric_cols
    numeric_cols = [c for c in numeric_cols if c not in (
        "ID", "Overall_Experience")]
    categorical_cols = X.select_dtypes(
        include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
    return preprocessor


def choose_model_and_scoring(y: pd.Series):
    # Decide classification vs regression
    if pd.api.types.is_integer_dtype(y) or pd.api.types.is_categorical_dtype(y):
        n_classes = int(y.nunique())
        if n_classes <= 2:
            model = XGBClassifier(
                objective="binary:logistic",
                use_label_encoder=False,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            scoring = "accuracy"
            cv = StratifiedKFold(
                n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        else:
            model = XGBClassifier(
                objective="multi:softprob",
                use_label_encoder=False,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            scoring = "accuracy"
            cv = StratifiedKFold(
                n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    else:
        # regression fallback
        model = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        scoring = make_scorer(mean_absolute_error, greater_is_better=False)
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return model, scoring, cv


def get_param_dist():
    return {
        "model__n_estimators": [100, 300, 500, 800, 1200],
        "model__learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1, 0.2],
        "model__max_depth": [3, 4, 6, 8, 10],
        "model__subsample": [0.5, 0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.5, 0.7, 0.8, 1.0],
        "model__min_child_weight": [1, 3, 5],
    }


def main():
    print("🚀 Starting optimized pipeline...")
    # load
    try:
        train, test = load_and_merge(
            "data/traveldata_train.csv",
            "data/surveydata_train.csv",
            "data/traveldata_test.csv",
            "data/surveydata_test.csv",
        )
    except FileNotFoundError as e:
        print("❌ Missing data file:", e)
        return

    train, test = basic_map_and_impute(train, test)

    if "Overall_Experience" not in train.columns:
        raise RuntimeError("Training file missing 'Overall_Experience' target")

    X = train.drop(["ID", "Overall_Experience"], axis=1)
    y = train["Overall_Experience"]
    X_test = test.drop("ID", axis=1)

    preprocessor = build_preprocessor(X)

    model_base, scoring, cv = choose_model_and_scoring(y)

    pipeline = Pipeline([("preproc", preprocessor), ("model", model_base)])

    param_dist = get_param_dist()

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        scoring=scoring,
        cv=cv,
        verbose=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )

    print("🔎 Running hyperparameter search (this may take time)...")
    search.fit(X, y)

    print("✅ Best params:", search.best_params_)
    print("✅ Best CV score:", search.best_score_)

    # Refit is True so best_estimator_ is refit on the full training set
    best_pipeline = search.best_estimator_

    # Save model artifact
    model_path = MODEL_DIR / "satisfaction_model.pkl"
    joblib.dump(best_pipeline, model_path)
    print(f"💾 Saved model to {model_path}")

    # Predict test set
    preds = best_pipeline.predict(X_test)

    # If regression, clip to reasonable range and round if original target was integer-like
    if isinstance(best_pipeline.named_steps["model"], XGBRegressor):
        preds = np.clip(preds, 0.0, 5.0)
        # If original target looked integer, round to nearest integer
        if pd.api.types.is_integer_dtype(y) or y.dropna().apply(float.is_integer).all():
            preds = np.rint(preds).astype(int)

    submission = pd.DataFrame({"ID": test["ID"], "Overall_Experience": preds})
    submission.to_csv("Submission_kebrahimpour.csv", index=False)
    print("📤 Submission written to Submission_kebrahimpour.csv")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
