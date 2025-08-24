import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    KFold,
    cross_val_score,
)
from sklearn.metrics import (
    mean_absolute_error,
    make_scorer,
)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge

# try optional LightGBM
try:
    import lightgbm as lgb

    LGB_AVAILABLE = True
except Exception:
    lgb = None
    LGB_AVAILABLE = False

# Config (added - fixes NameError)
RANDOM_STATE = 42
N_ITER_SEARCH = 24
CV_FOLDS = 5
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# try to import optuna, allow graceful fallback
try:
    import optuna
except Exception:
    optuna = None


def run_optuna_search(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Pipeline,
    n_trials: int,
    random_state: int,
    cv,
    scoring,
):
    """
    Run an Optuna study to optimize XGBoost hyperparameters.
    Returns (best_params, best_score).
    """
    if optuna is None:
        raise RuntimeError(
            "Optuna is not installed. Install with `pip install optuna` to use Optuna tuning."
        )

    # detect problem type from y
    is_regression = not (
        pd.api.types.is_integer_dtype(y) or pd.api.types.is_categorical_dtype(y)
    )
    n_classes = int(y.nunique()) if not is_regression else None

    def objective(trial: "optuna.trial.Trial") -> float:
        # sample hyperparameters
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

        if is_regression:
            model = XGBRegressor(
                random_state=random_state,
                n_jobs=1,
                verbosity=0,
                **params,
            )
        else:
            objective_name = (
                "binary:logistic"
                if (n_classes is not None and n_classes <= 2)
                else "multi:softprob"
            )
            model = XGBClassifier(
                objective=objective_name,
                use_label_encoder=False,
                random_state=random_state,
                n_jobs=1,
                verbosity=0,
                **params,
            )

        pipe = Pipeline([("preproc", preprocessor), ("model", model)])

        # use cross validation to evaluate
        scores = cross_val_score(pipe, X, y, scoring=scoring, cv=cv, n_jobs=1)
        mean_score = float(np.mean(scores))

        # report intermediate value for pruning support
        trial.report(mean_score, 0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return mean_score

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    return study.best_trial.params, study.best_value, study


def load_and_merge(
    train_travel_path, train_survey_path, test_travel_path, test_survey_path
):
    """Load and merge train/test travel + survey CSVs. Merge on ID if present else by index."""
    if not os.path.exists(train_travel_path) or not os.path.exists(train_survey_path):
        raise FileNotFoundError("Training files not found")
    if not os.path.exists(test_travel_path) or not os.path.exists(test_survey_path):
        raise FileNotFoundError("Test files not found")

    train_travel = pd.read_csv(train_travel_path)
    train_survey = pd.read_csv(train_survey_path)
    test_travel = pd.read_csv(test_travel_path)
    test_survey = pd.read_csv(test_survey_path)

    if "ID" in train_travel.columns and "ID" in train_survey.columns:
        train = pd.merge(train_travel, train_survey, on="ID")
    else:
        train = pd.merge(train_travel, train_survey, left_index=True, right_index=True)

    if "ID" in test_travel.columns and "ID" in test_survey.columns:
        test = pd.merge(test_travel, test_survey, on="ID")
    else:
        test = pd.merge(test_travel, test_survey, left_index=True, right_index=True)

    return train, test


def basic_map_and_impute(train, test):
    """Map textual ratings to numeric and perform simple imputations (medians for numeric)."""
    # Impute Arrival_Delay_in_Mins with training median if present
    for col in (
        "Arrival_Delay_in_Mins",
        "Departure_Delay_in_Mins",
        "Travel_Distance",
        "Age",
    ):
        if col in train.columns:
            med = pd.to_numeric(train[col], errors="coerce").median()
            train[col] = pd.to_numeric(train[col], errors="coerce").fillna(med)
            if col in test.columns:
                test[col] = pd.to_numeric(test[col], errors="coerce").fillna(med)

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
            med = train[c].median()
            train[c] = train[c].fillna(med)
            if c in test.columns:
                test[c] = test[c].map(satisfaction_mapping).astype(float).fillna(med)

    # Ensure categorical string columns in both frames are strings and missing replaced with "__MISSING__"
    for df in (train, test):
        for col in df.select_dtypes(include=["object", "category"]).columns:
            df[col] = df[col].fillna("__MISSING__").astype(str)

    return train, test


def build_preprocessor(X: pd.DataFrame):
    """Build ColumnTransformer: median impute + StandardScaler for numerics, impute+OneHot for categoricals."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ("ID", "Overall_Experience")]
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

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
    """Return (model_base, scoring, cv) depending on whether y is classification or regression."""
    # classification if integer-like or categorical
    if (
        pd.api.types.is_integer_dtype(y)
        or pd.api.types.is_categorical_dtype(y)
        or y.dtype == object
    ):
        n_classes = int(y.nunique())
        if n_classes <= 2:
            model = XGBClassifier(
                objective="binary:logistic",
                use_label_encoder=False,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        else:
            model = XGBClassifier(
                objective="multi:softprob",
                use_label_encoder=False,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        scoring = "accuracy"
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    else:
        model = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        scoring = make_scorer(mean_absolute_error, greater_is_better=False)
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return model, scoring, cv


def get_param_dist():
    """Parameter grid for RandomizedSearchCV (keys match pipeline step 'model')."""
    return {
        "model__n_estimators": [100, 300, 500, 800, 1200],
        "model__learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1, 0.2],
        "model__max_depth": [3, 4, 6, 8, 10],
        "model__subsample": [0.5, 0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.5, 0.7, 0.8, 1.0],
        "model__min_child_weight": [1, 3, 5, 7],
        "model__reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0, 3.0],
    }


def create_ensemble(is_regression: bool, xgb_params: dict | None, random_state: int):
    """Construct a stacking ensemble with XGB, RandomForest and (optionally) LightGBM.
    If xgb_params provided they will be applied to the XGB base learner.
    """
    estimators = []
    if is_regression:
        xgb_base = XGBRegressor(
            random_state=random_state, n_jobs=-1, verbosity=0, **(xgb_params or {})
        )
        rf_base = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        estimators.append(("xgb", xgb_base))
        estimators.append(("rf", rf_base))
        if LGB_AVAILABLE:
            estimators.append(
                ("lgb", lgb.LGBMRegressor(random_state=random_state, n_jobs=-1))
            )
        final_estimator = Ridge()
        ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            n_jobs=-1,
            passthrough=False,
        )
    else:
        xgb_base = XGBClassifier(
            use_label_encoder=False,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
            **(xgb_params or {}),
        )
        rf_base = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        estimators.append(("xgb", xgb_base))
        estimators.append(("rf", rf_base))
        if LGB_AVAILABLE:
            estimators.append(
                ("lgb", lgb.LGBMClassifier(random_state=random_state, n_jobs=-1))
            )
        final_estimator = LogisticRegression(max_iter=2000)
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            n_jobs=-1,
            passthrough=False,
        )
    return ensemble


def main():
    print("🚀 Starting optimized pipeline...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-optuna", action="store_true", help="Use Optuna for hyperparameter tuning"
    )
    parser.add_argument(
        "--use-ensemble",
        action="store_true",
        help="Create stacking ensemble of XGB + RF (+ LightGBM if available)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=40,
        help="Number of Optuna trials (if using Optuna)",
    )
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--n-iter",
        type=int,
        default=N_ITER_SEARCH,
        help="RandomizedSearchCV iterations (if not using Optuna)",
    )
    args = parser.parse_args()

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

    # If user requests ensemble, we'll build stacking models later.
    if args.use_optuna:
        if optuna is None:
            raise RuntimeError(
                "Optuna is not installed. Install it with: pip install optuna"
            )
        print("🔬 Running Optuna tuning for XGB (used inside ensemble or standalone)...")
        best_params, best_score, study = run_optuna_search(
            X, y, preprocessor, args.n_trials, args.random_state, cv, scoring
        )
        print("✅ Optuna best score:", best_score)
        print("✅ Optuna best params:", best_params)

        # Build final model(s)
        is_regression = isinstance(model_base, XGBRegressor)
        if args.use_ensemble:
            # Use best_params for XGB base and include RF / LGB defaults in stacking
            final_model = create_ensemble(
                is_regression=is_regression,
                xgb_params=best_params,
                random_state=args.random_state,
            )
            print(
                "🏋️ Fitting stacking ensemble (Optuna-tuned XGB inside) on full training data..."
            )
            # wrap ensemble in pipeline with preprocessor
            best_pipeline = Pipeline(
                [("preproc", preprocessor), ("model", final_model)]
            )
            best_pipeline.fit(X, y)
        else:
            # Standalone XGB with best params
            if is_regression:
                final_model = XGBRegressor(
                    random_state=args.random_state,
                    n_jobs=-1,
                    verbosity=0,
                    **best_params,
                )
            else:
                n_classes = int(y.nunique()) if not is_regression else None
                objective_name = (
                    "binary:logistic"
                    if (n_classes is not None and n_classes <= 2)
                    else "multi:softprob"
                )
                final_model = XGBClassifier(
                    objective=objective_name,
                    use_label_encoder=False,
                    random_state=args.random_state,
                    n_jobs=-1,
                    verbosity=0,
                    **best_params,
                )
            best_pipeline = Pipeline(
                [("preproc", preprocessor), ("model", final_model)]
            )
            best_pipeline.fit(X, y)

    else:
        # RandomizedSearch / direct fit path
        if args.use_ensemble:
            # Build ensemble with default hyperparameters (faster than searching whole ensemble space)
            is_regression = isinstance(model_base, XGBRegressor)
            ensemble_model = create_ensemble(
                is_regression=is_regression,
                xgb_params=None,
                random_state=args.random_state,
            )
            best_pipeline = Pipeline(
                [("preproc", preprocessor), ("model", ensemble_model)]
            )
            print(
                "🏋️ Fitting stacking ensemble (no hyperparam search) on full training data..."
            )
            best_pipeline.fit(X, y)
        else:
            print("🔎 Running RandomizedSearchCV tuning...")
            pipeline = Pipeline([("preproc", preprocessor), ("model", model_base)])
            param_dist = get_param_dist()
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_dist,
                n_iter=args.n_iter,
                scoring=scoring,
                cv=cv,
                verbose=2,
                random_state=args.random_state,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X, y)
            print("✅ Best params:", search.best_params_)
            print("✅ Best CV score:", search.best_score_)
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
