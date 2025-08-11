import os
import json
import pickle
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# --- Setup ---
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
client = MlflowClient()

model_name = os.getenv('MODEL_NAME')
candidate_run_info_path = os.getenv('CANDIDATE_RUN_INFO', 'final_run_info.json')
metric_to_compare = os.getenv('METRIC', 'rmse')  # can be 'rmse' or 'r2_score'
comparison_threshold = float(os.getenv('COMPARISON_THRESHOLD', 0.05))  # % improvement

print(f"Fetching production model for: {model_name}")
print(f"Candidate run info: {candidate_run_info_path}")

# --- Load candidate run info ---
with open(candidate_run_info_path, 'r') as f:
    candidate_info = json.load(f)

candidate_metrics = candidate_info['validation_metrics']
candidate_rmse = candidate_metrics.get('rmse')
candidate_r2 = candidate_metrics.get('r2_score')

print(f"Candidate model metrics -> RMSE: {candidate_rmse:.4f}, R²: {candidate_r2:.4f}")

# --- Get latest production model ---
production_versions = client.get_latest_versions(model_name, stages=["Production"])
if not production_versions:
    print("No production model found. This will be the first Production model.")
    promote_to_prod = True
    production_rmse, production_r2 = None, None
else:
    prod_version = production_versions[0]
    prod_tags = {}

    # Handle tags safely regardless of format
    if hasattr(prod_version, 'tags') and prod_version.tags:
        if isinstance(prod_version.tags, dict):
            prod_tags = prod_version.tags
        elif hasattr(prod_version.tags, '__iter__'):
            for tag in prod_version.tags:
                if hasattr(tag, 'key') and hasattr(tag, 'value'):
                    prod_tags[tag.key] = tag.value
                elif isinstance(tag, dict) and 'key' in tag:
                    prod_tags[tag['key']] = tag['value']

    production_rmse = float(prod_tags.get('validation_rmse', 'nan')) if 'validation_rmse' in prod_tags else None
    production_r2 = float(prod_tags.get('validation_r2', 'nan')) if 'validation_r2' in prod_tags else None

    print(f"Production model metrics -> RMSE: {production_rmse}, R²: {production_r2}")

    # --- Decision logic ---
    promote_to_prod = False
    if production_rmse is None or np.isnan(production_rmse):
        print("No RMSE found for production model — promoting candidate.")
        promote_to_prod = True
    else:
        improvement = (production_rmse - candidate_rmse) / production_rmse
        print(f"RMSE improvement over production: {improvement*100:.2f}%")
        if improvement > comparison_threshold:
            promote_to_prod = True
        else:
            print("Candidate model not significantly better than production.")

# --- Output decision ---
decision = {
    'promote_to_prod': promote_to_prod,
    'candidate_rmse': candidate_rmse,
    'candidate_r2': candidate_r2,
    'production_rmse': production_rmse,
    'production_r2': production_r2
}

with open('deployment_decision.json', 'w') as f:
    json.dump(decision, f, indent=2)

if promote_to_prod:
    print("✅ Candidate model will be promoted to Production.")
else:
    print("❌ Candidate model will NOT be promoted to Production.")
