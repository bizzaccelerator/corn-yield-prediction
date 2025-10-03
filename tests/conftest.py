"""Pytest configuration and fixtures for test suite."""

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

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Dimensions
    n_train, n_val, n_test = 60, 20, 15
    n_features = 10

    # ===== ENCODED DATA (numpy arrays) =====
    X_encoded_train = np.random.rand(n_train, n_features)
    np.save("X_encoded.npy", X_encoded_train)
    np.save("X_encoded_train.npy", X_encoded_train)

    X_encoded_val = np.random.rand(n_val, n_features)
    np.save("X_encoded_val.npy", X_encoded_val)

    y_train_array = np.random.uniform(1000, 2000, n_train)
    np.save("y.npy", y_train_array)

    y_val_array = np.random.uniform(1000, 2000, n_val)
    np.save("target_val.npy", y_val_array)

    # ===== FEATURE NAMES =====
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

    # ===== CORN.CSV (original data with string values) =====
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

    # ===== CSV DATA (numeric only for sklearn models) =====
    X_train_df = pd.DataFrame(
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
    X_train_df.to_csv("X_train.csv", index=False)

    y_train_df = pd.DataFrame({"Yield": y_train_array})
    y_train_df.to_csv("y_train.csv", index=False)

    X_val_df = pd.DataFrame(
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
    X_val_df.to_csv("X_val.csv", index=False)

    y_val_df = pd.DataFrame({"Yield": np.random.uniform(1000, 2000, n_val)})
    y_val_df.to_csv("y_val.csv", index=False)

    X_test_df = pd.DataFrame(
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
    X_test_df.to_csv("X_test.csv", index=False)

    y_test_df = pd.DataFrame({"Yield": np.random.uniform(1000, 2000, n_test)})
    y_test_df.to_csv("y_test.csv", index=False)

    # ===== DICT VECTORIZER AND MODELS =====
    try:
        import joblib
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LinearRegression

        # DictVectorizer
        dv = DictVectorizer()
        dummy_data = [{"feature_" + str(i): i for i in range(5)} for _ in range(10)]
        dv.fit(dummy_data)
        with open("dict_vectorizer", "wb") as f:
            pickle.dump(dv, f)

        # Main model
        mock_model = GradientBoostingRegressor(n_estimators=10, random_state=42)
        mock_model.fit(X_encoded_train, y_train_array)
        joblib.dump(mock_model, "model.pkl")
        joblib.dump(mock_model, "models/test_model.pkl")

        # Production model
        prod_model = LinearRegression()
        prod_model.fit(X_encoded_train, y_train_array)
        joblib.dump(prod_model, "models/prod_model.pkl")

    except ImportError:
        print("Warning: sklearn not available, skipping model creation")

    # ===== JSON CONFIGS =====
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
            "MAE": 0.3,
            "MSE": 0.25,
        },
        "validation_metrics": {
            "rmse": 0.52,
            "r2": 0.78,
            "mae": 0.32,
            "mse": 0.27,
            "RMSE": 0.52,
            "R2": 0.78,
            "MAE": 0.32,
            "MSE": 0.27,
        },
        "rmse": 0.5,
        "r2": 0.8,
        "mae": 0.3,
        "mse": 0.25,
        "RMSE": 0.5,
        "R2": 0.8,
        "MAE": 0.3,
        "MSE": 0.25,
        "val_rmse": 0.52,
        "val_r2": 0.78,
        "Val_RMSE": 0.52,
        "Val_R2": 0.78,
        "Val_MAE": 0.32,
        "Val_MSE": 0.27,
        "parameters": {"n_estimators": 100, "max_depth": 5},
        "model_path": "models/test_model.pkl",
    }
    with open("final_run_info.json", "w") as f:
        json.dump(final_run_info, f, indent=2)

    model_info = {
        "run_id": "test_run_123456",
        "model_name": "GradientBoostingRegressor",
        "model_version": 1,
        "metrics": {"rmse": 0.5, "r2": 0.8, "mae": 0.3, "RMSE": 0.5, "R2": 0.8},
        "parameters": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
        "feature_names": feature_names.tolist(),
    }
    with open("model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    production_model_info = {
        "run_id": "prod_run_0001",
        "model_name": "test_model",
        "model_version": 0,
        "metrics": {
            "rmse": 0.6,
            "r2": 0.75,
            "RMSE": 0.6,
            "R2": 0.75,
            "mae": 0.35,
            "MAE": 0.35,
        },
        "validation_metrics": {"rmse": 0.62, "r2": 0.73, "RMSE": 0.62, "R2": 0.73},
        "RMSE": 0.6,
        "R2": 0.75,
        "parameters": {"n_estimators": 50, "max_depth": 3},
        "model_path": "models/prod_model.pkl",
    }
    with open("production_model_info.json", "w") as f:
        json.dump(production_model_info, f, indent=2)

    # Setup Kaggle credentials
    kaggle_dir = Path.home() / ".config" / "kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    if not kaggle_json.exists():
        with open(kaggle_json, "w") as f:
            json.dump({"username": "test", "key": "test"}, f)
        os.chmod(kaggle_json, 0o600)

    yield


@pytest.fixture
def sample_dataframe():
    """Provide a sample dataframe for testing."""
    data = {
        "Education": ["Secondary", "Primary", "Tertiary", "Secondary"],
        "Gender": ["Male", "Female", "Male", "Female"],
        "Age bracket": ["26-35", "36-45", "46-55", "26-35"],
        "Household size": [5, 6, 4, 5],
        "Acreage": [2.5, 3.0, 2.0, 2.8],
        "Fertilizer amount": [100, 120, 80, 110],
        "Laborers": [3, 4, 2, 3],
        "Yield": [1500, 1600, 1400, 1550],
        "Main credit source": ["Bank", "Cooperative", "Self", "Bank"],
        "Farm records": ["Yes", "No", "Yes", "Yes"],
        "Main advisory source": ["Extension", "Radio", "TV", "Extension"],
        "Extension provider": ["Government", "NGO", "Private", "Government"],
        "Advisory format": ["Group", "Individual", "Digital", "Group"],
        "Advisory language": ["English", "Local", "English", "English"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_features():
    """Provide sample feature data for model testing."""
    return [
        {
            "Education": "Secondary",
            "Gender": "Male",
            "Age bracket": "26-35",
            "Household size": 5,
            "Acreage": 2.5,
            "Fertilizer amount": 100,
            "Laborers": 3,
        }
    ]


@pytest.fixture
def mock_model():
    """Provide a mock model for testing."""

    class MockModel:
        def predict(self, X):
            if hasattr(X, "__len__"):
                return np.array([1500.0] * len(X))
            return np.array([1500.0])

        def fit(self, X, y):
            return self

        def score(self, X, y):
            return 0.85

    return MockModel()


@pytest.fixture
def temp_working_dir(tmp_path, monkeypatch):
    """Create a temporary working directory for tests that need file isolation"""
    monkeypatch.chdir(tmp_path)
    return tmp_path
