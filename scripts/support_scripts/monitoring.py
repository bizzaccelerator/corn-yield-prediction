from datetime import datetime
import os
import pandas as pd
import json
import numpy as np
import joblib
import requests

from evidently import Report, Dataset, DataDefinition
from evidently.ui.workspace import RemoteWorkspace
from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import DashboardPanelPlot
from evidently.metrics import *
from evidently.presets import DataDriftPreset, DataSummaryPreset
from sklearn.model_selection import train_test_split

# Configuration for remote Evidently service
EVIDENTLY_SERVICE_URL = "https://evidently-ui-453290981886.us-central1.run.app"

### LOAD THE DATA (same as before)
validation_files_exist = os.path.exists('X_encoded_val.csv')

if not validation_files_exist:
    X_encoded_train = np.load('X_encoded.npy')
    target_train = np.load('y.npy')
    feature_names = np.load('feature_names.npy', allow_pickle=True)

    X_encoded_train = pd.DataFrame(X_encoded_train, columns=feature_names)
    target_train = pd.Series(target_train, name='target')

    X_encoded_val = None
    target_val = None

    print("Using training data only (no validation data provided)")
else:
    X_encoded_train = np.load('X_encoded.npy')
    target_train = np.load('y.npy')
    feature_names = np.load('feature_names.npy', allow_pickle=True)

    X_encoded_train = pd.DataFrame(X_encoded_train, columns=feature_names)
    target_train = pd.Series(target_train, name='target')

    X_encoded_val = pd.read_csv('X_encoded_val.csv', sep=";")
    target_val = pd.read_csv('target_val.csv', sep=";")

    print("Using both training and validation data")

### LOAD THE MODEL AND VECTORIZER
with open('model_info.json', "r") as f:
    model_info = json.load(f)

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

### CREATE PREDICTIONS (same as before)
train_predictions = model.predict(X_encoded_train.values)

if X_encoded_val is not None:
    val_predictions = model.predict(X_encoded_val.values)
    print("Created predictions for both training and validation data")
else:
    X_ref, X_cur, y_ref, y_cur = train_test_split(
        X_encoded_train, target_train, test_size=0.3, random_state=42
    )

    ref_predictions = model.predict(X_ref.values)
    cur_predictions = model.predict(X_cur.values)

    X_encoded_train = X_ref
    target_train = y_ref
    train_predictions = ref_predictions

    X_encoded_val = X_cur
    target_val = y_cur
    val_predictions = cur_predictions

    print("Split training data into reference and current datasets for monitoring")

### CREATE THE REPORT DATASETS (same as before)
reference_data = X_encoded_train.copy()
reference_data['target'] = target_train.values
reference_data['prediction'] = train_predictions

current_data = X_encoded_val.copy()
current_data['target'] = target_val.values
current_data['prediction'] = val_predictions

vectorized_features = list(X_encoded_train.columns)

schema = DataDefinition(
    numerical_columns=vectorized_features + ["prediction"],
    categorical_columns=[]
)

eval_data_1 = Dataset.from_pandas(pd.DataFrame(reference_data), data_definition=schema)
eval_data_2 = Dataset.from_pandas(pd.DataFrame(current_data), data_definition=schema)

### CREATE REMOTE WORKSPACE AND PROJECT
try:
    # Connect to remote Evidently workspace
    ws = RemoteWorkspace(EVIDENTLY_SERVICE_URL)
    
    # Try to get existing project or create new one
    project_name = "Corn Yield prediction"
    try:
        # List existing projects to check if it exists
        projects = ws.list_projects()
        existing_project = None
        for proj in projects:
            if proj.name == project_name:
                existing_project = proj
                break
        
        if existing_project:
            project = existing_project
            print(f"Using existing project: {project.name}")
        else:
            project = ws.create_project(project_name)
            project.description = "Predict the yield of corn in Kenya"
            print(f"Created new project: {project.name}")
    except Exception as e:
        print(f"Error managing project: {e}")
        # Fallback: create project with timestamp
        project = ws.create_project(f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        project.description = "Predict the yield of corn in Kenya"

    # Create and run report
    report = Report(metrics=[
        StdValue(column="prediction"), 
        RowCount(),
        ValueDrift(column="prediction", method="ks"),
        QuantileValue(column="prediction")
    ])
    
    report_result = report.run(current_data=eval_data_2, reference_data=eval_data_1)
    
    # Add report to workspace
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_name = f'{timestamp}_corn_yield_report'
    
    ws.add_run(project.id, report_result, include_data=False)
    
    ### ADD DASHBOARD PANELS
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Main Dashboard",
            size="full",
            values=[],
            plot_params={"plot_type": "text"},
        ),
        tab="Monitoring",
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Standard Deviation",
            subtitle="Std deviation for prediction.",
            size="half",
            values=[PanelMetric(
                legend="Standard Deviation",
                metric="StdValue",
                metric_labels={"column": "prediction"}
            )],
            plot_params={"plot_type": "bar", "is_stacked": False},
        ),
        tab="Monitoring",
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Row count",
            subtitle="Total number of evaluations over time.",
            size="half",
            values=[PanelMetric(legend="Row count", metric="RowCount")],
            plot_params={"plot_type": "counter", "aggregation": "sum"},
        ),
        tab="Monitoring",
    )

    project.save()
    
    # Export project metadata for Kestra
    project_metadata = {
        "project_id": str(project.id),
        "project_name": project.name,
        "report_name": report_name,
        "description": project.description,
        "evidently_service_url": EVIDENTLY_SERVICE_URL,
        "evidently_ui_url": "https://evidently-ui-453290981886.us-central1.run.app/",
        "created_at": datetime.now().isoformat(),
        "dashboard_url": f"https://evidently-ui-453290981886.us-central1.run.app/projects/{project.id}"
    }
    
    with open("project_metadata.json", "w") as f:
        json.dump(project_metadata, f, indent=2)
    
    print("Monitoring report and dashboard created successfully on remote Evidently service!")
    print(f"Project ID: {project.id}")
    print(f"Dashboard URL: {project_metadata['dashboard_url']}")
    
except Exception as e:
    print(f"Error connecting to remote Evidently service: {e}")
    print("Falling back to local workspace creation...")
    
    # Fallback to original local workspace creation
    from evidently.ui.workspace import Workspace
    
    ws = Workspace.create("workspace")
    project = ws.create_project("Corn Yield prediction")
    project.description = "Predict the yield of corn in Kenya"

    report = Report(metrics=[StdValue(column="prediction"), RowCount()])
    report_result = report.run(current_data=eval_data_2, reference_data=eval_data_1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_name = f'{timestamp}_report'
    
    ws.add_run(project.id, report_result, include_data=False)
    
    project_metadata = {
        "project_id": str(project.id),
        "project_name": project.name,
        "report_name": report_name,
        "description": project.description,
        "workspace_path": ws.path,
        "created_at": datetime.now().isoformat(),
        "fallback_mode": True
    }
    
    with open("project_metadata.json", "w") as f:
        json.dump(project_metadata, f, indent=2)
    
    print("Created local workspace as fallback")