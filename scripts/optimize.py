from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.sparse import hstack, csr_matrix
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os, pickle, json


########## THE FIRST PART IS LOADING THE DATA AND CONFIG

#Import the required information:
model_selection = os.getenv('MODEL_TYPE')
model_name = os.getenv('MODEL_NAME')
# Extracting the vectorizer 
with open("dict_vectorizer", 'rb') as f_in:
    dv = pickle.load(f_in)
    
# The prepared datasets are:
X_train = pd.read_csv("X_train.csv", sep=",")
y_train = pd.read_csv("y_train.csv", sep=",")
X_val = pd.read_csv("X_val.csv", sep=",")
y_val = pd.read_csv("y_val.csv", sep=",")
X_test = pd.read_csv("X_test.csv", sep=",")
y_test = pd.read_csv("y_test.csv", sep=",")

# Merge train & val

# Get the maximum number of features
max_features = max(X_train.shape[1], X_val.shape[1])

# Pad the smaller array with zeros
if X_train.shape[1] < max_features:
    padding = csr_matrix((X_train.shape[0], max_features - X_train.shape[1]))
    X_train = hstack([X_train, padding])
    
if X_val.shape[1] < max_features:
    padding = csr_matrix((X_val.shape[0], max_features - X_val.shape[1]))
    X_val = hstack([X_val, padding])

# Then finally merge
X_train_full = np.vstack((X_train, X_val))
y_train_full = pd.concat([y_train, y_val], ignore_index=True)

# Handle target column if it exists
if 'target' in y_train_full.columns:
    y_train = y_train_full['target']
if 'target' in y_test.columns:
    y_test = y_test['target']

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# Enable autologging for scikit-learn
mlflow.sklearn.autolog(log_input_examples=False, log_model_signatures=False)

# Setup MLflow tracking URI
mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', "MLFLOW_TRACKING_URI")
model_name = os.getenv('MODEL_NAME')
mlflow.set_tracking_uri(mlflow_uri)

print(f"MLflow Tracking URI: {mlflow_uri}")



###### THE EVALUATION FUCNTIONS USED ARE:

# The evaluation of metrics for the model will be done using this formula:
def evaluate_model(y_test, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} Performance:")
    print(f"  RMSE: {rmse}")
    print(f"  R² Score: {r2}")
    return {'rmse': rmse, 'r2_score': r2}


# The objective functions are: 
def objective_ridge(params):
    alpha = params['alpha']
    model = Ridge(alpha=alpha, random_state=42)    
    neg_mse = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
    return {'loss': -np.mean(neg_mse), 'status': STATUS_OK}




############ Evaluation and optimization of best models

if model_selection == 'linear':

    print("There is no need for optimization of this model. But Test will be considered.")

    # Start an MLflow run
    with mlflow.start_run(run_name="linear-regression-corn-yield"):

        # Log dataset information
        mlflow.log_param("n_features", X_train_full.shape[1])
        mlflow.log_param("n_train_samples", X_train_full.shape[0])
        mlflow.log_param("n_val_samples", X_test.shape[0])

        model_linear = LinearRegression()  
        model_linear.fit(X_train_full, y_train_full) 
        y_pred_linear = model_linear.predict(X_test)
        metrics_linear = evaluate_model(y_test, y_pred_linear, model_name)

        # Log validation metrics
        mlflow.log_metric("val_rmse", metrics_linear['rmse'])
        mlflow.log_metric("val_r2_score", metrics_linear['r2_score'])

        # Save model locally for Kestra (in addition to MLflow logging)
        os.makedirs("model_artifacts", exist_ok=True)
        
        # Defining the model name:
        output_file = f"model_artifacts/{model_name}_model.bin"
        with open(output_file, 'wb') as f_out:
            pickle.dump((dv, model_linear), f_out)
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model_linear,
            artifact_path="model",
            registered_model_name=model_name  # This registers the model
        )

        # Save metrics and run info
        run_info = {
            'mlflow_run_id': mlflow.active_run().info.run_id,
            'mlflow_tracking_uri': mlflow_uri,
            'validation_metrics': metrics_linear,
            'model_type': 'LinearRegression',
            'model_uri': f"runs:/{mlflow.active_run().info.run_id}/model"
        }
        
        with open('model_artifacts/optimized_run_info.json', 'w') as f:
            json.dump(run_info, f, indent=2)
        
        print(f"Model training completed!")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"Model saved locally and logged to MLflow")


elif model_selection == 'ridge':

    space_ridge = {
        'alpha': hp.loguniform('alpha', np.log(1e-4), np.log(1e3))
        }

    trials_ridge = Trials()
    best_ridge = fmin(fn=objective_ridge, space=space_ridge, algo=tpe.suggest, max_evals=20, trials=trials_ridge)

    model_ridge = Ridge(alpha=best_ridge['alpha'], random_state=42)

    model_ridge.fit(X_train_full, y_train_full)

    y_pred_ridge = model_ridge.predict(X_test)

    rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))






    

