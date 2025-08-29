from datetime import datetime
import os
import pandas as pd
import json
import numpy as np
import joblib
import requests

from evidently import Report, Dataset, DataDefinition, Regression
from evidently.ui.workspace import RemoteWorkspace
from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import DashboardPanelPlot
from evidently.metrics import *
from evidently.presets import RegressionPreset
from sklearn.model_selection import train_test_split

# Configuration for remote Evidently service
EVIDENTLY_SERVICE_URL = "https://evidently-ui-453290981886.us-central1.run.app"

def get_or_create_project(ws, project_name, description):
    """Get existing project or create new one with better error handling"""
    try:
        # List existing projects
        projects = ws.list_projects()
        
        for proj in projects:
            if proj.name == project_name:
                print(f"Found existing project: {project_name} (ID: {proj.id})")
                return proj, False  # Found existing, no setup needed
        
        # Project doesn't exist, create it
        print(f"Creating new project: {project_name}")
        new_project = ws.create_project(project_name)
        new_project.description = description
        return new_project, True  # New project, setup needed
        
    except Exception as e:
        print(f"Error managing project: {e}")
        # Fallback: create with timestamp
        timestamped_name = f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        project = ws.create_project(timestamped_name)
        project.description = description
        return project, True

def setup_regression_dashboard(project):
    """Setup regression dashboard panels - only call for new projects"""
    print("Setting up regression dashboard panels...")
    
    # Regression Performance Overview
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Model Performance Overview",
            subtitle="Corn yield prediction performance metrics",
            size="full",
            values=[
                PanelMetric(legend="MAE", metric="MAE"),
                PanelMetric(legend="RMSE", metric="RMSE"),
                PanelMetric(legend="R²", metric="R2Score")
            ],
            plot_params={"plot_type": "bar", "is_stacked": False},
        ),
        tab="Model Performance",
    )

    # Individual metric panels
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
            title="Root Mean Squared Error",
            subtitle="RMSE trend over time",
            size="half",
            values=[PanelMetric(legend="RMSE", metric="RMSE")],
            plot_params={"plot_type": "line"},
        ),
        tab="Model Performance",
    )

### LOAD THE DATA
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

### CREATE PREDICTIONS
train_predictions = model.predict(X_encoded_train.values)

if X_encoded_val is not None:
    val_predictions = model.predict(X_encoded_val.values)
    print("Created predictions for both training and validation data")
else:
    val_predictions = None
    X_encoded_val = None
    target_val = None
    print("Using complete training data as reference only")

### CREATE THE REPORT DATASET FOR REGRESSION ANALYSIS
reference_data = X_encoded_train.copy()
reference_data['target'] = target_train.values
reference_data['prediction'] = train_predictions

if X_encoded_val is not None:
    current_data = X_encoded_val.copy()
    current_data['target'] = target_val.values
    current_data['prediction'] = val_predictions
    print("Using validation data as current dataset")
else:
    current_data = None
    print("Using only reference data for model performance evaluation")

vectorized_features = list(X_encoded_train.columns)

# Define schema for regression analysis
schema = DataDefinition(
    regression=[Regression(target="target", prediction="prediction")],
    numerical_columns=vectorized_features,
)

eval_data_1 = Dataset.from_pandas(pd.DataFrame(reference_data), data_definition=schema)
eval_data_2 = Dataset.from_pandas(pd.DataFrame(current_data), data_definition=schema) if current_data is not None else None

### CREATE REMOTE WORKSPACE AND PROJECT
try:
    ws = RemoteWorkspace(EVIDENTLY_SERVICE_URL)
    
    project_name = "Corn Yield ML Monitoring"  # Unified project name
    project_description = "Comprehensive ML monitoring for corn yield prediction model - performance and drift analysis"
    
    project, is_new_project = get_or_create_project(ws, project_name, project_description)
    
    if is_new_project:
        setup_regression_dashboard(project)
        project.save()
        print("New project created with regression dashboard")
    else:
        print("Using existing project - no dashboard changes needed")

    # Create regression report
    if current_data is not None:
        report = Report(metrics=[
            MAE(column="prediction"),
            RMSE(column="prediction"), 
            R2Score(column="prediction"),
            MeanError(column="prediction")
        ])
        print("Running regression analysis report with reference and current data...")
        report_result = report.run(current_data=eval_data_2, reference_data=eval_data_1)
        analysis_type = "with_validation"
    else:
        report = Report(metrics=[
            MAE(column="prediction"),
            RMSE(column="prediction"),
            R2Score(column="prediction"),
            MeanError(column="prediction")
        ])
        print("Running regression analysis report using training data...")
        report_result = report.run(current_data=eval_data_1, reference_data=None)
        analysis_type = "training_only"
    
    # Add report to workspace
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_name = f'{timestamp}_regression_performance'
    
    ws.add_run(project.id, report_result, include_data=False)
    print(f"Report '{report_name}' added to workspace successfully")
    
    # Export project metadata for Kestra
    project_metadata = {
        "project_id": str(project.id),
        "project_name": project.name,
        "report_name": report_name,
        "report_type": "RegressionPerformance",
        "description": project.description,
        "evidently_service_url": EVIDENTLY_SERVICE_URL,
        "evidently_ui_url": "https://evidently-ui-453290981886.us-central1.run.app/",
        "created_at": datetime.now().isoformat(),
        "dashboard_url": f"https://evidently-ui-453290981886.us-central1.run.app/projects/{project.id}",
        "is_new_project": is_new_project,
        "data_info": {
            "training_samples": len(X_encoded_train),
            "validation_samples": len(X_encoded_val) if X_encoded_val is not None else 0,
            "features_count": len(vectorized_features),
            "model_type": model_info.get("model_type", "Unknown"),
            "analysis_type": analysis_type
        }
    }
    
    with open("project_metadata.json", "w") as f:
        json.dump(project_metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("REGRESSION MONITORING RESULTS")
    print("="*60)
    print(f"Project: {project.name}")
    print(f"Dashboard URL: {project_metadata['dashboard_url']}")
    print(f"Analysis type: {analysis_type}")
    print(f"Training samples: {len(X_encoded_train)}")
    if X_encoded_val is not None:
        print(f"Validation samples: {len(X_encoded_val)}")
    print(f"Features analyzed: {len(vectorized_features)}")
    print("="*60)
    
except Exception as e:
    print(f"Error connecting to remote Evidently service: {e}")
    print("Falling back to local workspace creation...")
    
    # Fallback to local workspace creation
    from evidently.ui.workspace import Workspace
    
    ws = Workspace.create("workspace")
    project = ws.create_project("Corn Yield ML Monitoring")
    project.description = "Comprehensive ML monitoring for corn yield prediction model"

    if current_data is not None:
        report = Report(metrics=[
            MAE(column="prediction"),
            RMSE(column="prediction"),
            R2Score(column="prediction"), 
            MeanError(column="prediction")
        ])
        report_result = report.run(current_data=eval_data_2, reference_data=eval_data_1)
        analysis_type = "with_validation"
    else:
        report = Report(metrics=[
            MAE(column="prediction"),
            RMSE(column="prediction"),
            R2Score(column="prediction"),
            MeanError(column="prediction")
        ])
        report_result = report.run(current_data=eval_data_1, reference_data=eval_data_1)
        analysis_type = "training_only"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_name = f'{timestamp}_regression_report_local'
    
    ws.add_run(project.id, report_result, include_data=False)
    
    project_metadata = {
        "project_id": str(project.id),
        "project_name": project.name,
        "report_name": report_name,
        "report_type": "RegressionPerformance",
        "description": project.description,
        "workspace_path": ws.path,
        "created_at": datetime.now().isoformat(),
        "fallback_mode": True,
        "data_info": {
            "training_samples": len(X_encoded_train),
            "validation_samples": len(X_encoded_val) if X_encoded_val is not None else 0,
            "features_count": len(vectorized_features),
            "model_type": model_info.get("model_type", "Unknown"),
            "analysis_type": analysis_type
        }
    }
    
    with open("project_metadata.json", "w") as f:
        json.dump(project_metadata, f, indent=2)
    
    print("Created local workspace as fallback with regression analysis")