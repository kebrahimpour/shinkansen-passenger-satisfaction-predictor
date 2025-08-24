#!/usr/bin/env python3
"""Training script for the Shinkansen Passenger Satisfaction Predictor.

This script loads and merges travel and survey data, trains the
satisfaction predictor model, saves the fitted model, and optionally
evaluates it on test data.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from shinkansen_predictor import SatisfactionPredictor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure to install dependencies with: uv sync")
    sys.exit(1)


def load_and_merge_data(
    travel_path: str, survey_path: str, verbose: bool = False
) -> pd.DataFrame:
    """Load and merge travel and survey data from CSV files."""
    if verbose:
        print(f"📊 Loading travel data from {travel_path}...")
        print(f"📊 Loading survey data from {survey_path}...")

    if not os.path.exists(travel_path):
        raise FileNotFoundError(f"Travel data file not found: {travel_path}")
    if not os.path.exists(survey_path):
        raise FileNotFoundError(f"Survey data file not found: {survey_path}")

    travel_df = pd.read_csv(travel_path)
    survey_df = pd.read_csv(survey_path)

    if verbose:
        print(
            f"Merging {len(travel_df)} travel records with "
            f"{len(survey_df)} survey records."
        )

    # Merge on the index since the files are aligned by row count
    merged_df = pd.merge(travel_df, survey_df, left_index=True, right_index=True)

    if verbose:
        print(
            f"✅ Merged data has {len(merged_df)} records and "
            f"{len(merged_df.columns)} columns."
        )
        print("🔍 Columns in merged data:", merged_df.columns.tolist())
    return merged_df


def prepare_features_and_target(df: pd.DataFrame) -> tuple:
    """Extract features (X) and target (y) from the DataFrame."""
    # Use the correct column names from your CSV files
    required_features = [
        "Age",
        "Type_Travel",
        "Travel_Class",
        "Travel_Distance",
        "Seat_Comfort",
        "Catering",
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
        "Departure_Delay_in_Mins",
        "Arrival_Delay_in_Mins",
    ]
    # The target variable is 'Overall_Experience'
    target_column = "Overall_Experience"

    # Check for required feature columns
    missing_features = [col for col in required_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    # --- Data Cleaning Step ---
    # Identify columns that should be treated as categories
    categorical_cols = [
        "Type_Travel",
        "Travel_Class",
        "Seat_Comfort",
        "Catering",
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
    # Fill missing values and ensure consistent string type for categorical columns
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("missing").astype(str)

    # Identify numerical columns
    numerical_cols = [
        "Age",
        "Travel_Distance",
        "Departure_Delay_in_Mins",
        "Arrival_Delay_in_Mins",
    ]
    # Fill missing values in numerical columns with the column's median
    for col in numerical_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    # --- End of Data Cleaning Step ---

    X = df[required_features].to_dict(orient="records")

    y = None
    # Extract target variable only if it exists in the dataframe
    if target_column in df.columns:
        y = df[target_column].tolist()

    return X, y


def evaluate_model(predictor: SatisfactionPredictor, X_test: list, y_test: list):
    """Evaluate the model and print metrics."""
    print("\n🧪 Evaluating model on test data...")
    predictions = predictor.predict(X_test)
    # Simple Mean Absolute Error calculation
    mae = sum(abs(p - a) for p, a in zip(predictions, y_test)) / len(y_test)
    print(f"✅ Mean Absolute Error on test set: {mae:.4f}")


def train_model(
    X_train: list, y_train: list, verbose: bool = False
) -> SatisfactionPredictor:
    """Train a SatisfactionPredictor model."""
    if verbose:
        print("🤖 Initializing SatisfactionPredictor...")
    predictor = SatisfactionPredictor()
    if verbose:
        print(f"🏋️ Training model on {len(X_train)} samples...")
    predictor.fit(X_train, y_train)
    if verbose:
        print("✅ Model training completed successfully!")
    return predictor


def save_model(
    predictor: SatisfactionPredictor, model_path: str, verbose: bool = False
):
    """Save the trained model to disk."""
    if verbose:
        print(f"💾 Saving model to {model_path}...")
    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    predictor.save_model(model_path)
    if verbose:
        print("✅ Model saved successfully!")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Shinkansen passenger satisfaction predictor",
    )
    # Training data paths
    parser.add_argument("--travel-data-path", default="data/traveldata_train.csv")
    parser.add_argument("--survey-data-path", default="data/surveydata_train.csv")
    # Test data paths for evaluation
    parser.add_argument("--test-travel-data-path", default=None)
    parser.add_argument("--test-survey-data-path", default=None)
    # Model output path
    parser.add_argument("--model-path", default="models/model.pkl")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("🚅 Shinkansen Passenger Satisfaction Predictor - Training Script")
    print("=" * 65)

    try:
        # Load and prepare training data
        train_df = load_and_merge_data(
            args.travel_data_path, args.survey_data_path, args.verbose
        )
        X_train, y_train = prepare_features_and_target(train_df)

        # Train model
        predictor = train_model(X_train, y_train, args.verbose)

        # Save model
        save_model(predictor, args.model_path, args.verbose)

        # Optional: Evaluate on test data
        if args.test_travel_data_path and args.test_survey_data_path:
            test_df = load_and_merge_data(
                args.test_travel_data_path, args.test_survey_data_path, args.verbose
            )
            X_test, y_test = prepare_features_and_target(test_df)

            # Only evaluate if the test data includes the target column
            if y_test is not None:
                evaluate_model(predictor, X_test, y_test)
            else:
                print(
                    "\n🧪 Test data does not contain the target column "
                    "('Overall_Experience')."
                )
                print("Skipping model evaluation.")

        print("\n🎉 Training completed successfully!")
        print(f"💾 Model saved to: {args.model_path}")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
