import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment(tmp_path_factory):
    """
    Crea archivos y datos de prueba que los scripts esperan encontrar
    para que pytest pueda importar y ejecutar sin errores.
    """
    tmp_dir = tmp_path_factory.mktemp("test_data")
    os.chdir(tmp_dir)

    # ============================================================
    # 1. Modelo de prueba
    # ============================================================
    model = LinearRegression()
    X = np.random.rand(50, 10)
    y = np.random.rand(50)
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    with open("models/test_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # ============================================================
    # 2. Datos de entrenamiento y validación
    # ============================================================
    n_train, n_val, n_features = 100, 20, 10

    # Train
    X_encoded_train = pd.DataFrame(
        np.random.rand(n_train, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    X_encoded_train.to_csv("X_encoded_train.csv", index=False)
    np.save("X_encoded_train.npy", X_encoded_train.values)

    # Validation (CSV con columnas + npy con .values)
    X_encoded_val = pd.DataFrame(
        np.random.rand(n_val, n_features), columns=[f"f{i}" for i in range(n_features)]
    )
    X_encoded_val.to_csv("X_encoded_val.csv", index=False)
    np.save("X_encoded_val.npy", X_encoded_val.values)

    # ============================================================
    # 3. final_run_info.json con todos los formatos de métricas
    # ============================================================
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
        # fallback keys at root level (para scripts estrictos)
        "RMSE": 0.5,
        "R2": 0.8,
        "Val_RMSE": 0.52,
        "Val_R2": 0.78,
        "parameters": {"n_estimators": 100, "max_depth": 5},
        "model_path": "models/test_model.pkl",
    }

    with open("final_run_info.json", "w") as f:
        json.dump(final_run_info, f, indent=2)

    # ============================================================
    # 4. Producción ficticia (registro de modelo actual)
    # ============================================================
    prod_model_info = {
        "run_id": "prod_run_0001",
        "model_name": "test_model",
        "model_version": 0,
        "metrics": {"rmse": 0.6, "r2": 0.75},
        "parameters": {"n_estimators": 50, "max_depth": 3},
        "model_path": "models/prod_model.pkl",
    }
    with open("production_model_info.json", "w") as f:
        json.dump(prod_model_info, f, indent=2)

    with open("models/prod_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # ============================================================
    # 5. Fake data adicional (drift, test, etc.)
    # ============================================================
    pd.DataFrame(
        np.random.rand(10, n_features), columns=[f"f{i}" for i in range(n_features)]
    ).to_csv("drift_reference.csv", index=False)

    pd.DataFrame(
        np.random.rand(5, n_features), columns=[f"f{i}" for i in range(n_features)]
    ).to_csv("drift_current.csv", index=False)

    pd.DataFrame(
        np.random.rand(20, n_features), columns=[f"f{i}" for i in range(n_features)]
    ).to_csv("X_test.csv", index=False)

    pd.Series(np.random.rand(20)).to_csv("y_test.csv", index=False)

    yield
