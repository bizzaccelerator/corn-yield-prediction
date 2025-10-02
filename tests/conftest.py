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

    # Create mock encoded training data for drift monitoring
    X_encoded_train = np.random.rand(100, 10)
    np.save("X_encoded.npy", X_encoded_train)

    # Create mock dict vectorizer
    class MockDictVectorizer:
        def transform(self, X):
            if isinstance(X, list):
                return np.random.rand(len(X), 10)
            return np.random.rand(1, 10)

        def get_feature_names_out(self):
            return [f"feature_{i}" for i in range(10)]

    with open("dict_vectorizer", "wb") as f:
        pickle.dump(MockDictVectorizer(), f)

    # Create mock final_run_info.json
    final_run_info = {
        "run_id": "test_run_123456",
        "model_name": "test_model",
        "model_version": 1,
        "metrics": {"rmse": 0.5, "r2": 0.8, "mae": 0.3},
        "parameters": {"n_estimators": 100, "max_depth": 5},
    }
    with open("final_run_info.json", "w") as f:
        json.dump(final_run_info, f)

    # Setup Kaggle credentials
    kaggle_dir = Path.home() / ".config" / "kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    if not kaggle_json.exists():
        with open(kaggle_json, "w") as f:
            json.dump({"username": "test", "key": "test"}, f)
        os.chmod(kaggle_json, 0o600)

    yield

    # Cleanup (optional)
    # You can add cleanup code here if needed


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
            return np.array([1500.0] * len(X))

        def fit(self, X, y):
            return self

        def score(self, X, y):
            return 0.85

    return MockModel()


@pytest.fixture
def temp_working_dir(tmp_path, monkeypatch):
    """Create a temporary working directory for tests that need file isolation."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
