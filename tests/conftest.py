import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root and scripts to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment with all required mock files."""

    # Create mock corn.csv with correct columns
    corn_data = {
        "Education": ["Secondary", "Primary", "Tertiary"],
        "Gender": ["Male", "Female", "Male"],
        "Age bracket": ["26-35", "36-45", "46-55"],
        "Household size": [5, 6, 4],
        "Acreage": [2.5, 3.0, 2.0],
        "Fertilizer amount": [100, 120, 80],
        "Laborers": [3, 4, 2],
        "Yield": [1500, 1600, 1400],
        "Main credit source": ["Bank", "Cooperative", "Self"],
        "Farm records": ["Yes", "No", "Yes"],
        "Main advisory source": ["Extension", "Radio", "TV"],
        "Extension provider": ["Government", "NGO", "Private"],
        "Advisory format": ["Group", "Individual", "Digital"],
        "Advisory language": ["English", "Local", "English"],
    }
    df = pd.DataFrame(corn_data)
    df.to_csv("corn.csv", index=False)

    # Create mock encoded training data
    X_encoded_train = np.random.rand(100, 10)
    np.save("X_encoded.npy", X_encoded_train)

    # Create mock target data
    y_train = np.random.rand(100)
    np.save("y.npy", y_train)

    # Create mock feature names
    feature_names = np.array(
        [
            "Education",
            "Gender",
            "Age_bracket",
            "Household_size",
            "Acreage",
            "Fertilizer_amount",
            "Laborers",
            "Main_credit_source",
            "Farm_records",
            "Main_advisory_source",
        ]
    )
    np.save("feature_names.npy", feature_names)

    # Create mock training CSVs
    n_train = 60
    X_train_data = pd.DataFrame(
        {
            "Education": np.random.choice([0, 1, 2], n_train),
            "Gender": np.random.choice([0, 1], n_train),
            "Age_bracket": np.random.choice([0, 1, 2], n_train),
            "Household_size": np.random.randint(3, 8, n_train),
            "Acreage": np.random.uniform(1.5, 4.0, n_train),
            "Fertilizer_amount": np.random.randint(50, 150, n_train),
            "Laborers": np.random.randint(1, 6, n_train),
        }
    )
    X_train_data.to_csv("X_train.csv", index=False)

    y_train_data = pd.DataFrame({"Yield": np.random.uniform(1000, 2000, n_train)})
    y_train_data.to_csv("y_train.csv", index=False)

    # Create validation CSVs
    n_val = 20
    X_val_data = pd.DataFrame(
        {
            "Education": np.random.choice([0, 1, 2], n_val),
            "Gender": np.random.choice([0, 1], n_val),
            "Age_bracket": np.random.choice([0, 1, 2], n_val),
            "Household_size": np.random.randint(3, 8, n_val),
            "Acreage": np.random.uniform(1.5, 4.0, n_val),
            "Fertilizer_amount": np.random.randint(50, 150, n_val),
            "Laborers": np.random.randint(1, 6, n_val),
        }
    )
    X_val_data.to_csv("X_val.csv", index=False)

    y_val_data = pd.DataFrame({"Yield": np.random.uniform(1000, 2000, n_val)})
    y_val_data.to_csv("y_val.csv", index=False)

    # 🔹 Save encoded validation sets (both npy and csv for compatibility)
    X_encoded_val = np.random.rand(n_val, 10)
    np.save("X_encoded_val.npy", X_encoded_val)
    pd.DataFrame(X_encoded_val).to_csv("X_encoded_val.csv", index=False)

    y_val = np.random.rand(n_val)
    np.save("y_val.npy", y_val)

    # Create test CSVs
    n_test = 15
    X_test_data = pd.DataFrame(
        {
            "Education": np.random.choice([0, 1, 2], n_test),
            "Gender": np.random.choice([0, 1], n_test),
            "Age_bracket": np.random.choice([0, 1, 2], n_test),
            "Household_size": np.random.randint(3, 8, n_test),
            "Acreage": np.random.uniform(1.5, 4.0, n_test),
            "Fertilizer_amount": np.random.randint(50, 150, n_test),
            "Laborers": np.random.randint(1, 6, n_test),
        }
    )
    X_test_data.to_csv("X_test.csv", index=False)
    y_test_data = pd.DataFrame({"Yield": np.random.uniform(1000, 2000, n_test)})
    y_test_data.to_csv("y_test.csv", index=False)

    # DictVectorizer
    from sklearn.feature_extraction import DictVectorizer

    dv = DictVectorizer()
    dv.fit([{"f0": 1, "f1": 2}])
    with open("dict_vectorizer", "wb") as f:
        pickle.dump(dv, f)

    # final_run_info.json with metrics (both lowercase and uppercase keys)
    final_run_info = {
        "run_id": "test_run_123456",
        "model_name": "test_model",
        "model_version": 1,
        "metrics": {
            "rmse": 0.5,
            "r2": 0.8,
            "mae": 0.3,
            "mse": 0.25,
            "RMSE": 0.5,
            "R2": 0.8,
        },
        "validation_metrics": {
            "rmse": 0.52,
            "r2": 0.78,
            "mae": 0.32,
            "mse": 0.27,
            "RMSE": 0.52,
            "R2": 0.78,
        },
        "parameters": {"n_estimators": 100, "max_depth": 5},
        "model_path": "models/test_model.pkl",
    }
    with open("final_run_info.json", "w") as f:
        json.dump(final_run_info, f)

    # model_info.json
    model_info = {
        "run_id": "test_run_123456",
        "model_name": "GradientBoostingRegressor",
        "model_version": 1,
        "metrics": {"rmse": 0.5, "r2": 0.8, "mae": 0.3, "RMSE": 0.5, "R2": 0.8},
        "parameters": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
        "feature_names": feature_names.tolist(),
    }
    with open("model_info.json", "w") as f:
        json.dump(model_info, f)

    # Kaggle credentials
    kaggle_dir = Path.home() / ".config" / "kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    with open(kaggle_dir / "kaggle.json", "w") as f:
        json.dump({"username": "test", "key": "test"}, f)
    os.chmod(kaggle_dir / "kaggle.json", 0o600)

    yield
