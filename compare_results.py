import pandas as pd
import numpy as np

# --- Create a dummy "our_results.csv" for this example ---
# In your actual notebook, you will NOT need this part.
# You will just load your own file.
try:
    # Let's base our dummy file on the structure of the sample submission
    sample_df = pd.read_csv("Sample_Submission.csv")
    our_results = sample_df.copy()

    # Let's introduce some differences for the sake of comparison
    # Assuming the prediction column is the second column (index 1)
    prediction_col = our_results.columns[1]

    # Change about 20% of the values slightly
    num_changes = int(len(our_results) * 0.2)
    indices_to_change = np.random.choice(our_results.index, num_changes, replace=False)

    # If the column is numeric, add some noise. Otherwise, change the category.
    if pd.api.types.is_numeric_dtype(our_results[prediction_col]):
        our_results.loc[indices_to_change, prediction_col] += (
            np.random.randn(num_changes) * 0.1
        )
    else:  # If it's categorical, we'll just mark it as 'different'
        our_results.loc[indices_to_change, prediction_col] = "DIFFERENT_PREDICTION"

    our_results.to_csv("our_results.csv", index=False)
    print("Created a dummy 'our_results.csv' for demonstration.")
except Exception as e:
    print(f"Could not read Sample_Submission.csv to create a dummy file. Error: {e}")
    print("Please ensure you have uploaded both files.")
