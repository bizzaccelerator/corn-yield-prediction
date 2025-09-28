import os
import kaggle
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer


def load_and_prepare_base_data():
    """Load and prepare training data from Kaggle"""

    # Authenticate with Kaggle
    kaggle.api.authenticate()

    # Download dataset
    handle = "japondo/corn-farming-data"
    print("Files to download:", kaggle.api.dataset_list_files(handle).files)
    kaggle.api.dataset_download_files(handle, path=".", unzip=True)

    # Find the CSV file in the current directory
    csv_files = [f for f in os.listdir(".") if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV files found after download")

    # Use the first CSV file found
    csv_filename = csv_files[0]
    print(f"Loading CSV file: {csv_filename}")

    # Load the dataset
    corn = pd.read_csv(csv_filename, sep=",")

    # Step 1: Understanding the data
    print("The dataset size is:", corn.shape)
    print("The columns in the dataset are:", corn.columns.tolist())

    # Display info about the dataset
    corn.info()
    print("The number of null values in the dataset is:", corn.isna().sum().sum())

    # Step 2: Data preparation
    print("Starting data preparation...")

    # Select subset for analysis
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

    # Column names standardization
    corn_subset.columns = [name.lower() for name in corn_subset.columns]
    corn_subset.columns = [name.replace(" ", "_") for name in corn_subset.columns]

    # Handle missing acreage values
    missing_land = corn_subset["acreage"].isna().sum()
    amount_ml = (missing_land / corn.shape[0]) * 100
    print(f"The percentage of registries with missing acreage values: {amount_ml:.2f}%")

    # Remove rows with missing acreage
    filter_missing = corn_subset["acreage"].isna()
    corn_subset = corn_subset[~filter_missing]

    # Handle missing education values
    corn_subset.loc[corn_subset["education"].isna(), "education"] = "No educated"

    print("Main statistics for cleaned dataset:")
    print(corn_subset.describe(include="all"))

    # Step 3: Feature selection
    print("The target variable is Yield")

    # Significant variables identified during analysis
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

    # Filter dataset to include only significant variables
    corn_cleaned = corn_subset[significant_var]
    corn_cleaned.reset_index(drop=True, inplace=True)
    print("Final cleaned data shape:", corn_cleaned.shape)
    print(corn_cleaned.head())

    # Prepare dataset for modeling
    X = corn_cleaned.drop("yield", axis=1)
    y = corn_cleaned["yield"]

    # Convert to dictionaries for vectorization
    X_dic = X.to_dict(orient="records")

    # Initialize and fit vectorizer for one-hot encoding
    dv = DictVectorizer(sparse=False)
    X_encoded = dv.fit_transform(X_dic)

    # Save encoded features
    np.save("X_encoded.npy", X_encoded)
    print(f"Saved X_encoded with shape: {X_encoded.shape}")

    # Save target values
    np.save("y.npy", y.values)
    print(f"Saved y with shape: {y.values.shape}")

    # Save feature names for reference
    feature_names = dv.get_feature_names_out()
    np.save("feature_names.npy", feature_names)
    print(f"Saved feature names: {len(feature_names)} features")

    print("=== BASE DATA PREPARATION COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    load_and_prepare_base_data()
