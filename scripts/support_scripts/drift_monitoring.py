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
from evidently.presets import DataDriftPreset, RegressionPreset
from sklearn.model_selection import train_test_split

# Configuration for remote Evidently service
EVIDENTLY_SERVICE_URL = "https://evidently-ui-453290981886.us-central1.run.app"

def dashboard_needs_drift_setup(project):
    """Check if dashboard already has drift panels configured"""
    try:
        # Check if the project has any panels in the "Data Drift" tab
        for tab_name, panels in project.dashboard.panels.items():
            if tab_name == "Data Drift" and len(panels) > 0:
                return False
        return True
    except Exception as e:
        print(f"Error checking drift dashboard setup: {e}")
        return True

def setup_drift_dashboard(project):
    """Setup dashboard panels for data drift analysis"""
    print("Setting up drift dashboard panels...")
    
    # Data Drift Overview panel
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Data Drift Overview",
            subtitle="Data drift detection summary",
            size="full",
            values=[PanelMetric(
                legend="Dataset Drift",
                metric="DatasetDriftMetric"
            )],
            plot_params={"plot_type": "bar", "is_stacked": False},
        ),
        tab="Data Drift",
    )

    # Number of Drifted Features panel
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Drifted Features Count",
            subtitle="Number of features with detected drift",
            size="half",
            values=[PanelMetric(
                legend="Drifted Features",
                metric="DatasetDriftMetric"
            )],
            plot_params={"plot_type": "line"},
        ),
        tab="Data Drift",
    )

    # Drift Score panel
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Drift Score",
            subtitle="Overall drift score over time",
            size="half",
            values=[PanelMetric(
                legend="Drift Score",
                metric="DatasetDriftMetric"
            )],
            plot_params={"plot_type": "line"},
        ),
        tab="Data Drift",
    )

### LOAD THE DATA (same as before)
validation_files_exist = os.path.exists('X_encoded_val.npy')

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

    X_encoded_val_array = np.load('X_encoded_val.npy')
    target_val_array = np.load('target_val.npy')

    # Convert to DataFrame with same column names as training data
    X_encoded_val = pd.DataFrame(X_encoded_val_array, columns=feature_names)
    target_val = pd.Series(target_val_array, name='target')

    print("Using both training and validation data")

print("Loading model...")
with open('model_info.json', "r") as f:
    model_info = json.load(f)

model = joblib.load('model.pkl')
# Note: vectorizer already loaded above

### CREATE PREDICTIONS FOR BOTH DATASETS
print("Creating predictions...")
train_predictions = model.predict(X_encoded_train.values)
val_predictions = model.predict(X_encoded_val.values)

### CREATE THE DATASETS FOR DRIFT ANALYSIS
print("Preparing datasets for drift analysis...")

# Reference data (training)
reference_data = X_encoded_train.copy()
reference_data['target'] = target_train.values
reference_data['prediction'] = train_predictions

# Current data (validation)
current_data = X_encoded_val.copy()
current_data['target'] = target_val.values.flatten() if hasattr(target_val.values, 'flatten') else target_val.values
current_data['prediction'] = val_predictions

print(f"Reference data shape: {reference_data.shape}")
print(f"Current data shape: {current_data.shape}")

# Get feature columns (excluding target and prediction)
feature_columns = [col for col in reference_data.columns if col not in ['target', 'prediction']]

# Define schema for both regression and drift analysis
schema = DataDefinition(
    regression=[Regression(target="target", prediction="prediction")],
    numerical_columns=feature_columns,
)

eval_data_reference = Dataset.from_pandas(reference_data, data_definition=schema)
eval_data_current = Dataset.from_pandas(current_data, data_definition=schema)

### CREATE REMOTE WORKSPACE AND PROJECT
try:
    # Connect to remote Evidently workspace
    ws = RemoteWorkspace(EVIDENTLY_SERVICE_URL)
    
    # Try to get existing project or create new one
    project_name = "Corn Yield Drift Analysis"
    drift_dashboard_was_setup = False
    
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
            print(f"Using existing drift project: {project.name}")
            
            # Only setup drift dashboard if it doesn't exist yet
            if dashboard_needs_drift_setup(project):
                setup_drift_dashboard(project)
                drift_dashboard_was_setup = True
                print("Drift dashboard panels added to existing project")
            else:
                print("Drift dashboard already configured, skipping panel setup")
        else:
            project = ws.create_project(project_name)
            project.description = "Data drift detection and regression performance analysis for corn yield prediction model"
            setup_drift_dashboard(project)
            drift_dashboard_was_setup = True
            print(f"Created new drift project: {project.name}")
    except Exception as e:
        print(f"Error managing drift project: {e}")
        # Fallback: create project with timestamp
        project = ws.create_project(f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        project.description = "Data drift detection and regression performance analysis for corn yield prediction model"
        setup_drift_dashboard(project)
        drift_dashboard_was_setup = True

    # Create comprehensive report with both drift and regression metrics
    print("Creating drift and regression analysis report...")
    report = Report(metrics=[
        # Data Drift Metrics
        DataDriftPreset(drift_share=0.7),
        # Regression Performance Metrics  
        MAE(column="prediction"),
        RMSE(column="prediction"),
        R2Score(column="prediction"),
        MeanError(column="prediction")
    ])
    
    print("Running combined drift and regression analysis...")
    report_result = report.run(
        current_data=eval_data_current, 
        reference_data=eval_data_reference
    )
    
    # Add report to workspace
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_name = f'{timestamp}_corn_drift_regression_report'
    
    ws.add_run(project.id, report_result, include_data=False)
    print(f"Report '{report_name}' added to workspace successfully")
    
    # Save project only if dashboard was modified
    if drift_dashboard_was_setup:
        project.save()
        print("Drift project saved with dashboard configuration")
    
    # Extract drift results for analysis
    drift_results = report_result.get_metric_result("DatasetDriftMetric")
    drift_detected = drift_results.drift_detected if hasattr(drift_results, 'drift_detected') else False
    drift_score = getattr(drift_results, 'drift_score', 0.0) if hasattr(drift_results, 'drift_score') else 0.0
    
    # Export comprehensive metadata
    drift_metadata = {
        "project_id": str(project.id),
        "project_name": project.name,
        "report_name": report_name,
        "report_type": "DriftAndRegression",
        "description": project.description,
        "evidently_service_url": EVIDENTLY_SERVICE_URL,
        "evidently_ui_url": "https://evidently-ui-453290981886.us-central1.run.app/",
        "created_at": datetime.now().isoformat(),
        "dashboard_url": f"https://evidently-ui-453290981886.us-central1.run.app/projects/{project.id}",
        "drift_dashboard_was_setup": drift_dashboard_was_setup,
        "drift_analysis": {
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "features_analyzed": len(feature_columns),
            "reference_samples": len(reference_data),
            "current_samples": len(current_data)
        },
        "data_info": {
            "reference_samples": len(reference_data),
            "current_samples": len(current_data),
            "features_count": len(feature_columns),
            "model_type": model_info.get("model_type", "Unknown"),
            "analysis_type": "drift_and_regression"
        }
    }
    
    with open("drift_metadata.json", "w") as f:
        json.dump(drift_metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("DRIFT ANALYSIS RESULTS")
    print("="*60)
    print(f"Drift Detected: {'YES' if drift_detected else 'NO'}")
    print(f"Drift Score: {drift_score:.4f}")
    print(f"Features Analyzed: {len(feature_columns)}")
    print(f"Reference Samples: {len(reference_data)}")
    print(f"Current Samples: {len(current_data)}")
    print(f"Dashboard URL: {drift_metadata['dashboard_url']}")
    print("="*60)
    
    if drift_detected:
        print("DATA DRIFT DETECTED - Review dashboard for detailed analysis")
    else:
        print("NO SIGNIFICANT DRIFT DETECTED")
    
except Exception as e:
    print(f"Error in drift analysis: {e}")
    print("Falling back to local workspace creation...")
    
    # Fallback to local workspace
    from evidently.ui.workspace import Workspace
    
    ws = Workspace.create("drift_workspace")
    project = ws.create_project("Corn Yield Drift Analysis")
    project.description = "Data drift detection and regression performance analysis for corn yield prediction model"

    report = Report(metrics=[
        DataDriftPreset(drift_share=0.7),
        MAE(column="prediction"),
        RMSE(column="prediction"),
        R2Score(column="prediction"),
        MeanError(column="prediction")
    ])
    
    report_result = report.run(
        current_data=eval_data_current, 
        reference_data=eval_data_reference
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_name = f'{timestamp}_drift_regression_report_local'
    
    ws.add_run(project.id, report_result, include_data=False)
    
    drift_metadata = {
        "project_id": str(project.id),
        "project_name": project.name,
        "report_name": report_name,
        "report_type": "DriftAndRegression",
        "description": project.description,
        "workspace_path": ws.path,
        "created_at": datetime.now().isoformat(),
        "fallback_mode": True,
        "data_info": {
            "reference_samples": len(reference_data),
            "current_samples": len(current_data),
            "features_count": len(feature_columns),
            "model_type": model_info.get("model_type", "Unknown"),
            "analysis_type": "drift_and_regression"
        }
    }
    
    with open("drift_metadata.json", "w") as f:
        json.dump(drift_metadata, f, indent=2)
    
    print("Created local workspace as fallback with drift analysis")