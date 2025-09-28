import os

import joblib
import numpy as np
import pandas as pd


def load_and_prepare_test_data():
    """Load and prepare test data using the same preprocessing as training"""

    print("Loading new dataset...")

    # Load the test dataset
    corn = pd.read_csv("corn_add.csv", sep=",")

    # Step 1: Understanding the data
    print("The dataset size is:", corn.shape)
    print("The columns in the dataset are:", corn.columns.tolist())

    # Display dataset info
    corn.info()
    print("The number of null values in the dataset is:", corn.isna().sum().sum())

    # Step 2: Data preparation (EXACT same as training)
    print("Starting data preparation with same preprocessing as training...")

    # Select subset for analysis (same as training)
    corn_subset = corn[
        [
            "Education",
            "Gender",
            "Age bracket",
            "Household size",
            "Acreage",
            "Fertilizer amount",
            "Laborers",
            "Yield",
            "Main credit source",
            "Farm records",
            "Main advisory source",
            "Extension provider",
            "Advisory format",
            "Advisory language",
        ]
    ]

    # Column names standardization (same as training)
    corn_subset.columns = [name.lower() for name in corn_subset.columns]
    corn_subset.columns = [name.replace(" ", "_") for name in corn_subset.columns]

    # Handle missing acreage values (same as training)
    missing_land = corn_subset["acreage"].isna().sum()
    amount_ml = (missing_land / corn.shape[0]) * 100
    print(f"The percentage of registries with missing acreage values: {amount_ml:.2f}%")

    # Remove rows with missing acreage
    filter_missing = corn_subset["acreage"].isna()
    corn_subset = corn_subset[~filter_missing]

    # Handle missing education values (same as training)
    corn_subset.loc[corn_subset["education"].isna(), "education"] = "No educated"

    print("Main statistics for cleaned dataset:")
    print(corn_subset.describe(include="all"))

    # Step 3: Feature selection (EXACT same feature selection as training)
    print("The target variable is Yield")

    # The significant variables (MUST match training exactly)
    significant_var = [
        "education",
        "age_bracket",
        "household_size",
        "laborers",
        "main_advisory_source",
        "acreage",
        "fertilizer_amount",
        "yield",
    ]

    # Filter dataset to include significant variables
    corn_cleaned = corn_subset[significant_var]
    corn_cleaned.reset_index(drop=True, inplace=True)
    print("Final cleaned data shape:", corn_cleaned.shape)
    print(corn_cleaned.head())

    # Prepare dataset
    X = corn_cleaned.drop("yield", axis=1)
    y = corn_cleaned["yield"]

    # Convert to dictionaries
    X_dic = X.to_dict(orient="records")

    print("Loading training vectorizer for consistent feature encoding...")

    # Load the vectorizer from training to ensure consistent encoding
    if not os.path.exists("vectorizer.pkl"):
        raise FileNotFoundError(
            "vectorizer.pkl not found. This file should come from training."
        )

    dv = joblib.load("vectorizer.pkl")

    # Apply the same vectorizer (transform only, not fit_transform)
    print("Transforming test data using training vectorizer...")
    X_encoded_val = dv.transform(X_dic)

    # Save encoded validation features
    np.save("X_encoded_val.npy", X_encoded_val)
    print(f"Saved X_encoded_val with shape: {X_encoded_val.shape}")

    # Save validation target values
    np.save("target_val.npy", y.values)
    print(f"Saved target_val with shape: {y.values.shape}")

    # Save feature names for reference (should match training)
    feature_names = dv.get_feature_names_out()
    np.save("feature_names.npy", feature_names)
    print(f"Feature names saved: {len(feature_names)} features")

    # Verify feature consistency
    if os.path.exists("feature_names.npy"):
        try:
            training_features = np.load("feature_names.npy")
            if len(training_features) != len(feature_names):
                print(
                    f"WARNING: Feature count mismatch! Training: {
                        len(training_features)}, Test: {
                        len(feature_names)}")
            else:
                print("✓ Feature count matches training data")
        except Exception as e:
            print(f"Could not verify feature consistency: {e}")

    print("=== TEST DATA PREPARATION COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    load_and_prepare_test_data()
