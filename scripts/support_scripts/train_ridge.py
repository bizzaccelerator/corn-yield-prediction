# The first step involves importing the libraries required for the process:
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
import pickle
import json

# Model packages used
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge


t_model = os.getenv('MODEL_TYPE')
# Extracting the vectorizer 
with open("dict_vectorizer", 'rb') as f_in:
    dv = pickle.load(f_in)

### Step 4: Model identification ### - Let's try some models:

# Setup MLflow tracking URI
mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', "MLFLOW_TRACKING_URI")
model_name = os.getenv('MODEL_NAME')
mlflow.set_tracking_uri(mlflow_uri)

print(f"MLflow Tracking URI: {mlflow_uri}")

# The prepared datasets are:
X_train = pd.read_csv("X_train.csv", sep=",")
y_train = pd.read_csv("y_train.csv", sep=",")
X_val = pd.read_csv("X_val.csv", sep=",")
y_val = pd.read_csv("y_val.csv", sep=",")

# Handle target column if it exists
if 'target' in y_train.columns:
    y_train = y_train['target']
if 'target' in y_val.columns:
    y_val = y_val['target']

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# The evaluation of metrics for the model will be done using this formula:
def evaluate_model(y_test, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} Performance:")
    print(f"  RMSE: {rmse}")
    print(f"  R² Score: {r2}")
    return {'rmse': rmse, 'r2_score': r2}

# __2. Ridge Regression Model:__
# The model is trained as follows:

# Enable autologging for scikit-learn
mlflow.sklearn.autolog(log_input_examples=False, log_model_signatures=False)

# Start an MLflow run
with mlflow.start_run(run_name="Ridge-regression-corn-yield"):
    
    # Log dataset information
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_param("n_train_samples", X_train.shape[0])
    mlflow.log_param("n_val_samples", X_val.shape[0])
    
    ridge = Pipeline([
        ('scaler', StandardScaler()),   # Step 1: scale features
        ('ridge', Ridge())              # Step 2: Ridge regression
        ])
    ridge.fit(X_train, y_train)
    
    # The trained model is used to predict the values in the validation dataset:
    y_pred_val = ridge.predict(X_val)
    
    # The evaluation for the linear models are:
    metrics = evaluate_model(y_val, y_pred_val, "Ridge Regressor")
    
    # Log validation metrics
    mlflow.log_metric("val_rmse", metrics['rmse'])
    mlflow.log_metric("val_r2_score", metrics['r2_score'])
    
    # Save model locally for Kestra (in addition to MLflow logging)
    os.makedirs("model_artifacts", exist_ok=True)
    
    # Defining the model name:
    output_file = f"model_artifacts/{t_model}_model.bin"
    with open(output_file, 'wb') as f_out:
        pickle.dump((dv, ridge), f_out)
 
    # Log the model to MLflow
    mlflow.sklearn.log_model(
        sk_model=ridge,
        artifact_path="model",
        registered_model_name=model_name  # This registers the model
    )

    # Save metrics and run info
    run_info = {
        'mlflow_run_id': mlflow.active_run().info.run_id,
        'mlflow_tracking_uri': mlflow_uri,
        'validation_metrics': metrics,
        'model_type': 'RidgeRegressor',
        'model_uri': f"runs:/{mlflow.active_run().info.run_id}/model"
    }
    
    with open('model_artifacts/ridge_run_info.json', 'w') as f:
        json.dump(run_info, f, indent=2)
    
    print(f"Model training completed!")
    print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
    print(f"Model saved locally and logged to MLflow")
