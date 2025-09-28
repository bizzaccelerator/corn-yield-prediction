from datetime import datetime
import os
import glob
import pandas as pd
import json
import numpy as np
import joblib
import requests

from evidently import Report, Dataset, DataDefinition, Regression
from evidently.ui.workspace import RemoteWorkspace, Workspace
from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import DashboardPanelPlot
from evidently.metrics import *
from evidently.presets import RegressionPreset


# Configuration for remote Evidently service
EVIDENTLY_SERVICE_URL = "https://evidently-ui-453290981886.us-central1.run.app"

# Configuration option to force report generation even if reports exist
FORCE_REPORT_GENERATION = (
    os.getenv("FORCE_REPORT_GENERATION", "false").lower() == "true"
)


def check_project_has_reports(ws, project_id) -> tuple[bool, int]:
    """
    Check if a project already has reports/runs
    Returns: (has_reports: bool, report_count: int)
    """
    try:
        # Convert UUID to string for file path operations
        project_id_str = str(project_id)

        print(f"Checking for reports in project: {project_id_str}")

        # Method 1: Check local workspace snapshots directory
        workspace_dir = os.path.join("workspace", project_id_str, "snapshots")
        print(f"Looking for snapshots in: {workspace_dir}")

        if os.path.exists(workspace_dir):
            snapshot_files = glob.glob(os.path.join(workspace_dir, "*.json"))
            snapshot_count = len(snapshot_files)

            if snapshot_count > 0:
                print(f"Found {snapshot_count} report snapshots in local workspace")
                return True, snapshot_count
            else:
                print("Snapshot directory exists but no report files found")
        else:
            print("No local snapshot directory found")

        # Method 2: Try to get runs from remote workspace
        try:
            # Try different methods to get runs from the workspace
            if hasattr(ws, "list_runs"):
                runs = ws.list_runs(project_id)
                if runs:
                    run_count = len(runs)
                    print(f"Found {run_count} runs via workspace.list_runs()")
                    return True, run_count

            # Try getting project first then runs
            project = ws.get_project(project_id)
            if hasattr(project, "list_runs"):
                runs = project.list_runs()
                if runs:
                    run_count = len(runs)
                    print(f"Found {run_count} runs via project.list_runs()")
                    return True, run_count

        except Exception as api_error:
            print(f"Could not check remote runs: {api_error}")

        # Method 3: Direct API call as fallback
        try:
            api_url = f"{EVIDENTLY_SERVICE_URL}/api/projects/{project_id_str}/reports"
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    report_count = len(data)
                elif isinstance(data, dict) and "items" in data:
                    report_count = len(data["items"])
                else:
                    report_count = 0

                if report_count > 0:
                    print(f"Found {report_count} reports via direct API call")
                    return True, report_count
        except Exception as api_error:
            print(f"Direct API call failed: {api_error}")

        print("No reports found by any method")
        return False, 0

    except Exception as e:
        print(f"Error checking project reports: {str(e)}")
        # If we can't check, assume no reports exist to be safe
        return False, 0


def get_or_create_project(ws, project_name, description):
    """Get existing project or create new one"""
    try:
        projects = ws.list_projects()
        for proj in projects:
            if proj.name == project_name:
                print(f"Found existing project: {project_name} (ID: {proj.id})")
                return proj, False
        print(f"Creating new project: {project_name}")
        new_project = ws.create_project(project_name)
        new_project.description = description
        new_project.save()
        return new_project, True
    except Exception as e:
        print(f"Error managing project: {e}")
        timestamped_name = f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        project = ws.create_project(timestamped_name)
        project.description = description
        project.save()
        return project, True


def setup_regression_dashboard(project):
    """Setup regression dashboard panels"""
    print("Setting up regression dashboard panels...")
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Model Performance Overview",
            subtitle="Corn yield prediction performance metrics",
            size="full",
            values=[
                PanelMetric(legend="MAE", metric="MAE"),
                PanelMetric(legend="RMSE", metric="RMSE"),
                PanelMetric(legend="R²", metric="R2Score"),
            ],
            plot_params={"plot_type": "bar", "is_stacked": False},
        ),
        tab="Model Performance",
    )

    # Add individual metric panels
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Mean Absolute Error",
            subtitle="MAE trend over time",
            size="half",
            values=[PanelMetric(legend="MAE", metric="MAE")],
            plot_params={"plot_type": "line"},
        ),
        tab="Model Performance",
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="R² Score",
            subtitle="Model explained variance",
            size="half",
            values=[PanelMetric(legend="R²", metric="R2Score")],
            plot_params={"plot_type": "line"},
        ),
        tab="Model Performance",
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="RSME Score",
            subtitle="Average difference between predicted values and actual values",
            size="half",
            values=[PanelMetric(legend="RMSE", metric="RMSE")],
            plot_params={"plot_type": "line"},
        ),
        tab="Model Performance",
    )

    project.save()


def create_skip_metadata(project_id, project_name, report_count, reason):
    """Create metadata when skipping report generation"""
    project_metadata = {
        "project_id": str(project_id),
        "project_name": project_name,
        "report_created": False,
        "skip_reason": reason,
        "existing_report_count": report_count,
        "evidently_service_url": EVIDENTLY_SERVICE_URL,
        "dashboard_url": f"{EVIDENTLY_SERVICE_URL}/projects/{project_id}",
        "checked_at": datetime.now().isoformat(),
        "status": "skipped",
    }
    with open("project_metadata.json", "w") as f:
        json.dump(project_metadata, f, indent=2)
    return project_metadata


### MAIN EXECUTION STARTS HERE ###
try:
    ws = RemoteWorkspace(EVIDENTLY_SERVICE_URL)
    project_name = "Corn Yield ML Monitoring"
    project_description = "Comprehensive ML monitoring for corn yield prediction model - performance and drift analysis"

    project, is_new_project = get_or_create_project(
        ws, project_name, project_description
    )

    # Check if project already has reports (skip if reports exist, unless forced or new project)
    if not is_new_project and not FORCE_REPORT_GENERATION:
        has_reports, report_count = check_project_has_reports(ws, project.id)

        if has_reports:
            print("\n" + "=" * 60)
            print("MONITORING TASK SKIPPED")
            print("=" * 60)
            print(f"Project '{project_name}' already has {report_count} reports.")
            print("Skipping report generation to avoid duplicates.")
            if FORCE_REPORT_GENERATION:
                print("To force report generation, set FORCE_REPORT_GENERATION=true")
            print(f"Dashboard URL: {EVIDENTLY_SERVICE_URL}/projects/{project.id}")
            print("=" * 60)

            create_skip_metadata(
                project.id,
                project_name,
                report_count,
                f"Project already has {report_count} existing reports",
            )

            # Exit successfully without creating new reports
            exit(0)
    elif FORCE_REPORT_GENERATION:
        print(
            "🔄 FORCE_REPORT_GENERATION is enabled - proceeding with report generation"
        )

    print("Proceeding with report generation...")

except Exception as e:
    print(f"Error during initial setup: {e}")
    print("Proceeding with local workspace fallback...")
    ws = Workspace.create("workspace")
    project = ws.create_project("Corn Yield ML Monitoring")
    project.description = "Local fallback project"
    is_new_project = True


### LOAD DATA ###
if os.path.exists("X_encoded_val.csv"):
    X_encoded_train = np.load("X_encoded.npy")
    target_train = np.load("y.npy")
    feature_names = np.load("feature_names.npy", allow_pickle=True)
    X_encoded_train = pd.DataFrame(X_encoded_train, columns=feature_names)
    target_train = pd.Series(target_train, name="target")
    X_encoded_val = pd.read_csv("X_encoded_val.csv", sep=";")
    target_val = pd.read_csv("target_val.csv", sep=";")
    print("Using both training and validation data")
else:
    X_encoded_train = np.load("X_encoded.npy")
    target_train = np.load("y.npy")
    feature_names = np.load("feature_names.npy", allow_pickle=True)
    X_encoded_train = pd.DataFrame(X_encoded_train, columns=feature_names)
    target_train = pd.Series(target_train, name="target")
    X_encoded_val, target_val = None, None
    print("Using training data only")


### LOAD MODEL ###
with open("model_info.json", "r") as f:
    model_info = json.load(f)
model = joblib.load("model.pkl")

### PREDICTIONS ###
train_predictions = model.predict(X_encoded_train.values)
val_predictions = (
    model.predict(X_encoded_val.values) if X_encoded_val is not None else None
)

### BUILD DATASETS ###
reference_data = X_encoded_train.copy()
reference_data["target"] = target_train.values
reference_data["prediction"] = train_predictions

current_data = None
if X_encoded_val is not None:
    current_data = X_encoded_val.copy()
    current_data["target"] = target_val.values
    current_data["prediction"] = val_predictions

schema = DataDefinition(
    regression=[Regression(target="target", prediction="prediction")],
    numerical_columns=list(X_encoded_train.columns),
)

eval_data_ref = Dataset.from_pandas(reference_data, data_definition=schema)
eval_data_cur = (
    Dataset.from_pandas(current_data, data_definition=schema)
    if current_data is not None
    else None
)

### CREATE REPORT ###
if is_new_project:
    setup_regression_dashboard(project)

report = Report(
    metrics=[
        MAE(column="prediction"),
        RMSE(column="prediction"),
        R2Score(column="prediction"),
        MeanError(column="prediction"),
    ]
)

if current_data is not None:
    report_result = report.run(current_data=eval_data_cur, reference_data=eval_data_ref)
    analysis_type = "with_validation"
    print("Running regression analysis report with reference and current data...")
else:
    report_result = report.run(current_data=eval_data_ref, reference_data=None)
    analysis_type = "training_only"
    print("Running regression analysis report using training data...")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
report_name = f"{timestamp}_regression_performance"

ws.add_run(project.id, report_result, include_data=False)
print(f"Report '{report_name}' added to workspace successfully")

project_metadata = {
    "project_id": str(project.id),
    "project_name": project.name,
    "report_name": report_name,
    "report_created": True,
    "report_type": "RegressionPerformance",
    "description": project.description,
    "evidently_service_url": EVIDENTLY_SERVICE_URL,
    "evidently_ui_url": f"{EVIDENTLY_SERVICE_URL}/",
    "dashboard_url": f"{EVIDENTLY_SERVICE_URL}/projects/{project.id}",
    "created_at": datetime.now().isoformat(),
    "is_new_project": is_new_project,
    "status": "success",
    "data_info": {
        "training_samples": len(X_encoded_train),
        "validation_samples": len(X_encoded_val) if X_encoded_val is not None else 0,
        "features_count": len(X_encoded_train.columns),
        "model_type": model_info.get("model_type", "Unknown"),
        "analysis_type": analysis_type,
    },
}

with open("project_metadata.json", "w") as f:
    json.dump(project_metadata, f, indent=2)

print("\n" + "=" * 60)
print("REGRESSION MONITORING RESULTS")
print("=" * 60)
print(f"Project: {project.name}")
print(f"Dashboard URL: {project_metadata['dashboard_url']}")
print(f"Analysis type: {analysis_type}")
print(f"Training samples: {len(X_encoded_train)}")
if X_encoded_val is not None:
    print(f"Validation samples: {len(X_encoded_val)}")
print(f"Features analyzed: {len(X_encoded_train.columns)}")
print("=" * 60)
