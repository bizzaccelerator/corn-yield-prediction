import json
import os
import pickle
import shutil

import joblib
import mlflow
import mlflow.tracking
from mlflow.artifacts import download_artifacts


def extract_production_model():
    """Extract current production model from MLflow registry"""

    # Create artifacts directory
    os.makedirs("mydir/artifacts", exist_ok=True)

    # Get environment variables
    mlflow_url = os.getenv("MLFLOW_URL")
    model_name = os.getenv("MODEL_NAME")

    if not mlflow_url or not model_name:
        raise ValueError("MLFLOW_URL and MODEL_NAME environment variables must be set")

    # Set MLflow tracking URI
    mlflow_tracking_uri = mlflow.set_tracking_uri(mlflow_url)

    # Get production model version
    client = mlflow.MlflowClient()

    try:
        # Get latest version in Production stage
        production_versions = client.get_latest_versions(
            model_name, stages=["Production"]
        )

        if not production_versions:
            raise Exception(f"No model found in Production stage for {model_name}")

        latest_prod_version = production_versions[0]
        model_version = latest_prod_version.version
        run_id = latest_prod_version.run_id

        print(f"MLflow URI: {mlflow_tracking_uri}")
        print(f"Run ID: {run_id}")
        print(f"Found production model: {model_name} version {model_version}")

        # Download the model
        print("Downloading model directory...")
        model_dir_path = f"runs:/{run_id}/model"
        local_model_dir = download_artifacts(model_dir_path)

        print(f"Downloaded model directory to: {local_model_dir}")
        print(f"Model directory contents: {os.listdir(local_model_dir)}")

        # Load MLflow model and save as simple pickle
        print("Loading and converting MLflow model...")
        model = mlflow.sklearn.load_model(model_dir_path)

        # Save the model
        model_output_path = "mydir/artifacts/model.pkl"
        joblib.dump(model, model_output_path)
        print(f"Model saved to: {model_output_path}")

        # Copy vectorizer from downloaded model directory
        vectorizer_source = os.path.join(local_model_dir, "vectorizer.pkl")
        vectorizer_output_path = "mydir/artifacts/vectorizer.pkl"

        vectorizer_found = False
        if os.path.exists(vectorizer_source):
            shutil.copy2(vectorizer_source, vectorizer_output_path)
            vectorizer_found = True
            print(
                f"Vectorizer copied from {vectorizer_source} to {vectorizer_output_path}"
            )
        else:
            print(f"Vectorizer not found at {vectorizer_source}")
            # List what's actually in the model directory
            print(f"Contents of {local_model_dir}: {os.listdir(local_model_dir)}")

        # Verify both files exist with sizes
        for artifact_name in ["model.pkl", "vectorizer.pkl"]:
            artifact_path = f"mydir/artifacts/{artifact_name}"
            if os.path.exists(artifact_path):
                size = os.path.getsize(artifact_path)
                print(f"{artifact_name}: {size} bytes")
            else:
                if artifact_name == "vectorizer.pkl":
                    print(f"Warning: {artifact_name} not found")
                else:
                    raise FileNotFoundError(f"{artifact_name} missing after download")

        print("=== ARTIFACT DOWNLOAD COMPLETED SUCCESSFULLY ===")

        # Save model info for next tasks
        model_info = {
            "model_name": model_name,
            "model_version": model_version,
            "model_uri": model_dir_path,
            "model_stage": "Production",
            "vectorizer_found": vectorizer_found,
        }

        with open("model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)

        # Show final directory contents
        print(f"Root directory contents: {os.listdir('.')}")
        print(f"Artifacts directory contents: {os.listdir('mydir/artifacts')}")

    except Exception as e:
        print(f"Error extracting model: {str(e)}")
        raise e


if __name__ == "__main__":
    extract_production_model()
