from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.sparse import hstack, csr_matrix
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=alpha, random_state=42))
    ])
    neg_mse = cross_val_score(model, X_train_full, y_train_full, cv=5, scoring="neg_mean_squared_error")
    return {'loss': -np.mean(neg_mse), 'status': STATUS_OK}

def objective_lasso(params):
    alpha = params['alpha']
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Lasso(alpha=alpha, max_iter=5000, random_state=42))
    ])
    neg_mse = cross_val_score(model, X_train_full, y_train_full, cv=5, scoring="neg_mean_squared_error")
    return {'loss': -np.mean(neg_mse), 'status': STATUS_OK}

def objective_gbr(params):
    # Convert int params explicitly
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])
    params['min_samples_split'] = int(params['min_samples_split'])
    params['min_samples_leaf'] = int(params['min_samples_leaf'])
    model = GradientBoostingRegressor(**params, random_state=42)
    neg_mse = cross_val_score(model, X_train_full, y_train_full, cv=5, scoring="neg_mean_squared_error")
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
    
    # DISABLE autologging for hyperopt optimization to avoid conflicts
    mlflow.sklearn.autolog(disable=True)
    
    # Start an MLflow run
    with mlflow.start_run(run_name="Ridge-regression-corn-yield"):
        
        # Log dataset information
        mlflow.log_param("n_features", X_train_full.shape[1])
        mlflow.log_param("n_train_samples", X_train_full.shape[0])
        mlflow.log_param("n_val_samples", X_test.shape[0])
        
        print("Starting hyperparameter optimization...")
        
        # Optimization for parameter alpha
        space_ridge = {
            'alpha': hp.loguniform('alpha', np.log(1e-4), np.log(1e3))
        }
        
        trials_ridge = Trials()
        best_ridge = fmin(fn=objective_ridge, space=space_ridge, algo=tpe.suggest, max_evals=20, trials=trials_ridge)
        
        print(f"Optimization completed. Best alpha: {best_ridge['alpha']}")
        
        # Log the best parameters found
        mlflow.log_param("best_alpha", best_ridge['alpha'])
        mlflow.log_param("optimization_trials", len(trials_ridge.trials))
        mlflow.log_param("optimization_algorithm", "TPE")
        
        # Create and train the optimized model
        model_ridge = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=best_ridge['alpha'], random_state=42))
        ])
        model_ridge.fit(X_train_full, y_train_full)
        
        # Make predictions and evaluate
        y_pred_ridge = model_ridge.predict(X_test)
        metrics_ridge = evaluate_model(y_test, y_pred_ridge, model_name)
        
        # Log validation metrics manually (since autolog is disabled)
        mlflow.log_metric("val_rmse", metrics_ridge['rmse'])
        mlflow.log_metric("val_r2_score", metrics_ridge['r2_score'])
        
        # Log model parameters
        for param_name, param_value in model_ridge.get_params().items():
            mlflow.log_param(f"model_{param_name}", param_value)
        
        # Save model locally for Kestra
        os.makedirs("model_artifacts", exist_ok=True)
        
        output_file = f"model_artifacts/{model_name}_model.bin"
        with open(output_file, 'wb') as f_out:
            pickle.dump((dv, model_ridge), f_out)
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model_ridge,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        # Log hyperopt trials information
        try:
            # Log some statistics about the optimization
            losses = [trial['result']['loss'] for trial in trials_ridge.trials]
            mlflow.log_metric("best_cv_loss", min(losses))
            mlflow.log_metric("mean_cv_loss", np.mean(losses))
            mlflow.log_metric("std_cv_loss", np.std(losses))
        except Exception as e:
            print(f"Warning: Could not log hyperopt statistics: {e}")
        
        # Save metrics and run info
        run_info = {
            'mlflow_run_id': mlflow.active_run().info.run_id,
            'mlflow_tracking_uri': mlflow_uri,
            'validation_metrics': metrics_ridge,
            'model_type': 'Ridge',
            'model_uri': f"runs:/{mlflow.active_run().info.run_id}/model",
            'best_parameters': best_ridge,
            'optimization_info': {
                'algorithm': 'TPE',
                'max_evals': 20,
                'n_trials_completed': len(trials_ridge.trials)
            }
        }
        
        with open('model_artifacts/optimized_run_info.json', 'w') as f:
            json.dump(run_info, f, indent=2)
        
        print(f"Model training completed!")
        print(f"Best alpha: {best_ridge['alpha']}")
        print(f"Validation RMSE: {metrics_ridge['rmse']:.4f}")
        print(f"Validation R²: {metrics_ridge['r2_score']:.4f}")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"Model saved locally and logged to MLflow")


elif model_selection == 'lasso':

    # DISABLE autologging for hyperopt optimization to avoid conflicts
    mlflow.sklearn.autolog(disable=True)

    # Start an MLflow run
    with mlflow.start_run(run_name="lasso-regression-corn-yield"):

        # Log dataset information
        mlflow.log_param("n_features", X_train_full.shape[1])
        mlflow.log_param("n_train_samples", X_train_full.shape[0])
        mlflow.log_param("n_val_samples", X_test.shape[0])
        
        print("Starting hyperparameter optimization...")
        
        # Optimization for parameter alpha
        space_lasso = {
            'alpha': hp.loguniform('alpha', np.log(1e-4), np.log(1e3))
        }
        
        trials_lasso = Trials()
        best_lasso = fmin(fn=objective_lasso, space=space_lasso, algo=tpe.suggest, max_evals=20, trials=trials_lasso)
        
        print(f"Optimization completed. Best alpha: {best_lasso['alpha']}")
        
        # Log the best parameters found
        mlflow.log_param("best_alpha", best_lasso['alpha'])
        mlflow.log_param("optimization_trials", len(trials_lasso.trials))
        mlflow.log_param("optimization_algorithm", "TPE")
        
        # Create and train the optimized model
        model_lasso = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=best_lasso['alpha'], max_iter=5000, random_state=42))
        ])
        model_lasso.fit(X_train_full, y_train_full)
        
        # Make predictions and evaluate
        y_pred_lasso = model_lasso.predict(X_test)
        metrics_lasso = evaluate_model(y_test, y_pred_lasso, model_name)
        
        # Log validation metrics manually (since autolog is disabled)
        mlflow.log_metric("val_rmse", metrics_lasso['rmse'])
        mlflow.log_metric("val_r2_score", metrics_lasso['r2_score'])
        
        # Log model parameters
        for param_name, param_value in model_lasso.get_params().items():
            mlflow.log_param(f"model_{param_name}", param_value)
        
        # Save model locally for Kestra
        os.makedirs("model_artifacts", exist_ok=True)
        
        output_file = f"model_artifacts/{model_name}_model.bin"
        with open(output_file, 'wb') as f_out:
            pickle.dump((dv, model_lasso), f_out)
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model_lasso,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        # Log hyperopt trials information
        try:
            # Log some statistics about the optimization
            losses = [trial['result']['loss'] for trial in trials_lasso.trials]
            mlflow.log_metric("best_cv_loss", min(losses))
            mlflow.log_metric("mean_cv_loss", np.mean(losses))
            mlflow.log_metric("std_cv_loss", np.std(losses))
        except Exception as e:
            print(f"Warning: Could not log hyperopt statistics: {e}")
        
        # Save metrics and run info
        run_info = {
            'mlflow_run_id': mlflow.active_run().info.run_id,
            'mlflow_tracking_uri': mlflow_uri,
            'validation_metrics': metrics_lasso,
            'model_type': 'Lasso',
            'model_uri': f"runs:/{mlflow.active_run().info.run_id}/model",
            'best_parameters': best_lasso,
            'optimization_info': {
                'algorithm': 'TPE',
                'max_evals': 20,
                'n_trials_completed': len(trials_lasso.trials)
            }
        }
        
        with open('model_artifacts/optimized_run_info.json', 'w') as f:
            json.dump(run_info, f, indent=2)
        
        print(f"Model training completed!")
        print(f"Best alpha: {best_lasso['alpha']}")
        print(f"Validation RMSE: {metrics_lasso['rmse']:.4f}")
        print(f"Validation R²: {metrics_lasso['r2_score']:.4f}")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"Model saved locally and logged to MLflow")


elif model_selection == 'gbr':

    # DISABLE autologging for hyperopt optimization to avoid conflicts
    mlflow.sklearn.autolog(disable=True)

    # Start an MLflow run
    with mlflow.start_run(run_name="Gradient-Boosting-regression-corn-yield"):

        # Log dataset information
        mlflow.log_param("n_features", X_train_full.shape[1])
        mlflow.log_param("n_train_samples", X_train_full.shape[0])
        mlflow.log_param("n_val_samples", X_test.shape[0])
        
        print("Starting hyperparameter optimization...")
        
        # Optimization for parameter alpha
        space_gbr = {
            'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
            'max_depth': hp.quniform('max_depth', 2, 10, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1)
        }
        
        trials_gbr = Trials()
        best_gbr = fmin(fn=objective_gbr, space=space_gbr, algo=tpe.suggest, max_evals=20, trials=trials_gbr)
        
        # Convert some params to int for final training
        best_gbr['n_estimators'] = int(best_gbr['n_estimators'])
        best_gbr['max_depth'] = int(best_gbr['max_depth'])
        best_gbr['min_samples_split'] = int(best_gbr['min_samples_split'])
        best_gbr['min_samples_leaf'] = int(best_gbr['min_samples_leaf'])
        
        print(f"Optimization completed. Best n_estimators: {best_gbr['n_estimators']}")
        print(f"Optimization completed. Best max_depth: {best_gbr['max_depth']}")
        print(f"Optimization completed. Best learning_rate: {best_gbr['learning_rate']}")
        print(f"Optimization completed. Best min_samples_split: {best_gbr['min_samples_split']}")
        print(f"Optimization completed. Best min_samples_leaf: {best_gbr['min_samples_leaf']}")
        
        # Log the best parameters found
        mlflow.log_param("best_n_estimators", best_gbr['n_estimators'])
        mlflow.log_param("best_max_depth", best_gbr['max_depth'])
        mlflow.log_param("best_learning_rate", best_gbr['learning_rate'])
        mlflow.log_param("best_min_samples_split", best_gbr['min_samples_split'])
        mlflow.log_param("best_min_samples_leaf", best_gbr['min_samples_leaf'])
        mlflow.log_param("optimization_trials", len(trials_gbr.trials))
        mlflow.log_param("optimization_algorithm", "TPE")
        
        # Create and train the optimized model
        model_gbr = GradientBoostingRegressor(**best_gbr, random_state=42)
        model_gbr.fit(X_train_full, y_train_full)
        
        # Make predictions and evaluate
        y_pred_gbr = model_gbr.predict(X_test)
        metrics_gbr = evaluate_model(y_test, y_pred_gbr, model_name)
        
        # Log validation metrics manually (since autolog is disabled)
        mlflow.log_metric("val_rmse", metrics_gbr['rmse'])
        mlflow.log_metric("val_r2_score", metrics_gbr['r2_score'])
        
        # Log model parameters
        for param_name, param_value in model_gbr.get_params().items():
            mlflow.log_param(f"model_{param_name}", param_value)
        
        # Save model locally for Kestra
        os.makedirs("model_artifacts", exist_ok=True)
        
        output_file = f"model_artifacts/{model_name}_model.bin"
        with open(output_file, 'wb') as f_out:
            pickle.dump((dv, model_gbr), f_out)
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model_gbr,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        # Log hyperopt trials information
        try:
            # Log some statistics about the optimization
            losses = [trial['result']['loss'] for trial in trials_gbr.trials]
            mlflow.log_metric("best_cv_loss", min(losses))
            mlflow.log_metric("mean_cv_loss", np.mean(losses))
            mlflow.log_metric("std_cv_loss", np.std(losses))
        except Exception as e:
            print(f"Warning: Could not log hyperopt statistics: {e}")
        
        # Save metrics and run info
        run_info = {
            'mlflow_run_id': mlflow.active_run().info.run_id,
            'mlflow_tracking_uri': mlflow_uri,
            'validation_metrics': metrics_gbr,
            'model_type': 'GradientBoostingRegressor',
            'model_uri': f"runs:/{mlflow.active_run().info.run_id}/model",
            'best_parameters': best_gbr,
            'optimization_info': {
                'algorithm': 'TPE',
                'max_evals': 20,
                'n_trials_completed': len(trials_gbr.trials)
            }
        }
        
        with open('model_artifacts/optimized_run_info.json', 'w') as f:
            json.dump(run_info, f, indent=2)
        
        print(f"Model training completed!")
        print(f"Best Learning rate: {best_gbr['learning_rate']}")
        print(f"Validation RMSE: {metrics_gbr['rmse']:.4f}")
        print(f"Validation R²: {metrics_gbr['r2_score']:.4f}")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"Model saved locally and logged to MLflow")

