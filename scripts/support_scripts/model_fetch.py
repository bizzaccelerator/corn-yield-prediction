import json
import os

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

# --- Setup ---
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
client = MlflowClient()

model_name = os.getenv("MODEL_NAME")
candidate_run_info_path = os.getenv("CANDIDATE_RUN_INFO", "final_run_info.json")
metric_to_compare = os.getenv("METRIC", "rmse")  # can be 'rmse' or 'r2_score'
comparison_threshold = float(os.getenv("COMPARISON_THRESHOLD", 0.05))  # % improvement
force_deployment = os.getenv("FORCE_DEPLOYMENT", "false").lower() == "true"

print(f"Fetching production model for: {model_name}")
print(f"Candidate run info: {candidate_run_info_path}")
print(f"Force deployment: {force_deployment}")

# --- Load candidate run info ---
with open(candidate_run_info_path, "r") as f:
    candidate_info = json.load(f)

candidate_metrics = candidate_info["validation_metrics"]
candidate_rmse = candidate_metrics.get("rmse")
candidate_r2 = candidate_metrics.get("r2_score")

print(f"Candidate model metrics -> RMSE: {candidate_rmse:.4f}, R2: {candidate_r2:.4f}")

# --- Get latest production model ---
production_versions = client.get_latest_versions(model_name, stages=["Production"])

should_deploy = False
production_rmse, production_r2 = None, None

if force_deployment:
    print("🔧 Force deployment enabled - will deploy regardless of performance")
    should_deploy = True
elif not production_versions:
    print("No production model found. This will be the first Production model.")
    should_deploy = True
    production_rmse, production_r2 = None, None
else:
    prod_version = production_versions[0]
    prod_tags = {}

    # Handle tags safely regardless of format
    if hasattr(prod_version, "tags") and prod_version.tags:
        if isinstance(prod_version.tags, dict):
            prod_tags = prod_version.tags
        elif hasattr(prod_version.tags, "__iter__"):
            for tag in prod_version.tags:
                if hasattr(tag, "key") and hasattr(tag, "value"):
                    prod_tags[tag.key] = tag.value
                elif isinstance(tag, dict) and "key" in tag:
                    prod_tags[tag["key"]] = tag["value"]

    production_rmse = (
        float(prod_tags.get("validation_rmse", "nan"))
        if "validation_rmse" in prod_tags
        else None
    )
    production_r2 = (
        float(prod_tags.get("validation_r2", "nan"))
        if "validation_r2" in prod_tags
        else None
    )

    print(f"Production model metrics -> RMSE: {production_rmse}, R2: {production_r2}")

    # --- Decision logic ---
    if production_rmse is None or np.isnan(production_rmse):
        print("No RMSE found for production model — promoting candidate.")
        should_deploy = True
    else:
        improvement = (production_rmse - candidate_rmse) / production_rmse
        print(f"RMSE improvement over production: {improvement * 100:.2f}%")
        if improvement > comparison_threshold:
            should_deploy = True
            print(
                f"Candidate model shows {
                    improvement *
                    100:.2f}% improvement (threshold: {
                    comparison_threshold *
                    100:.2f}%)")
        else:
            print(f"Candidate model improvement {
                improvement * 100:.2f}% is below threshold {comparison_threshold * 100:.2f}%")

# --- Output decision ---
decision = {
    "promote_to_prod": should_deploy,
    "should_deploy": should_deploy,  # Alternative key name for compatibility
    "candidate_rmse": candidate_rmse,
    "candidate_r2": candidate_r2,
    "production_rmse": production_rmse,
    "production_r2": production_r2,
    "force_deployment": force_deployment,
    "comparison_threshold": comparison_threshold,
    "improvement_percentage": (
        ((production_rmse - candidate_rmse) / production_rmse * 100)
        if production_rmse and not np.isnan(production_rmse)
        else None
    ),
}

with open("deployment_decision.json", "w") as f:
    json.dump(decision, f, indent=2)

if should_deploy:
    print("Candidate model will be promoted to Production and deployed.")
else:
    print("Candidate model will NOT be promoted to Production or deployed.")
