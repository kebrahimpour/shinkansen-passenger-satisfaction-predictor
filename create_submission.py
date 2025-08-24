"""
This script performs the full machine learning pipeline to generate the
submission file for the Shinkansen Passenger Satisfaction competition.
"""

import pandas as pd
from xgboost import XGBClassifier

print("🚀 Starting the prediction pipeline...")

# --- 1. Data Loading and Merging ---
try:
    print("📂 Loading data files...")
    train_travel = pd.read_csv("data/traveldata_train.csv")
    train_survey = pd.read_csv("data/surveydata_train.csv")
    test_travel = pd.read_csv("data/traveldata_test.csv")
    test_survey = pd.read_csv("data/surveydata_test.csv")
except FileNotFoundError as e:
    print(f"❌ Error: Data file not found. {e}")
    print("Please ensure the data files are in the 'data/' directory.")
    exit()

train_df = pd.merge(train_travel, train_survey, on="ID")
test_df = pd.merge(test_travel, test_survey, on="ID")
print("✅ Data loaded and merged successfully.")

# --- 2. Data Cleaning and Preprocessing ---
print("🧹 Cleaning and preprocessing data...")

# Impute missing 'Arrival_Delay_in_Mins' with the median from the training data
median_arrival_delay = train_df["Arrival_Delay_in_Mins"].median()
train_df["Arrival_Delay_in_Mins"] = train_df["Arrival_Delay_in_Mins"].fillna(
    median_arrival_delay
)
test_df["Arrival_Delay_in_Mins"] = test_df["Arrival_Delay_in_Mins"].fillna(
    median_arrival_delay
)

# Define a mapping for text-based survey responses to numerical values
satisfaction_mapping = {
    "Extremely Poor": 0,
    "Poor": 1,
    "Needs Improvement": 2,
    "Acceptable": 3,
    "Good": 4,
    "Excellent": 5,
}

# List of survey columns that contain text ratings
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

for col in survey_cols:
    # Convert text ratings to numbers. Unmapped values become NaN.
    train_df[col] = train_df[col].map(satisfaction_mapping)
    test_df[col] = test_df[col].map(satisfaction_mapping)

    # Calculate median from the training data (works on numbers, ignores NaNs)
    col_median = train_df[col].median()

    # Fill NaNs (from unmapped or missing values) with the median
    train_df[col] = train_df[col].fillna(col_median)
    test_df[col] = test_df[col].fillna(col_median)

print("✅ Data cleaning complete.")

# --- 3. Feature Engineering ---
print("🔧 Engineering features...")
# Separate target variable from features
X = train_df.drop(["ID", "Overall_Experience"], axis=1)
y = train_df["Overall_Experience"]
X_test = test_df.drop("ID", axis=1)

# One-hot encode categorical features
categorical_cols = X.select_dtypes(include=["object"]).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Align columns between train and test sets to ensure they match
X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)
print("✅ Feature engineering complete.")

# --- 4. Model Training ---
print("🤖 Training the XGBoost model on the full training data...")
# Using hyperparameters that are a good starting point
xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    random_state=42,
)

# We train on the full dataset (X, y) to make the best possible predictions
xgb_model.fit(X, y, verbose=False)
print("✅ Model training complete.")

# --- 5. Prediction and Submission File Generation ---
print("📈 Making predictions on the test data...")
test_predictions = xgb_model.predict(X_test)

submission_df = pd.DataFrame(
    {"ID": test_df["ID"], "Overall_Experience": test_predictions}
)

submission_df.to_csv("Submission_kebrahimpour.csv", index=False)
print(
    "\n🎉 Success! The 'Submission_kebrahimpour.csv' file has been created "
    "in your project directory."
)
print("Submission file head:")
print(submission_df.head())
