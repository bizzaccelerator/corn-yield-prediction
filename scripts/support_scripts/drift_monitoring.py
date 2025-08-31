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
from evidently.presets import DataDriftPreset, RegressionPreset, DataSummaryPreset
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

def setup_drift_dashboard(project):
    """Setup drift dashboard panels - only call for new projects"""
    print("Setting up drift dashboard panels...")
    
    # Data Drift Overview
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Data Drift Detection",
            subtitle="Dataset drift monitoring",
            size="full",
            values=[PanelMetric(legend="Dataset Drift", metric="DatasetDriftMetric")],
            plot_params={"plot_type": "bar", "is_stacked": False},
        ),
        tab="Data Drift",
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Drift Score Trend",
            subtitle="Drift score over time",
            size="half",
            values=[PanelMetric(legend="Drift Score", metric="DatasetDriftMetric")],
            plot_params={"plot_type": "line"},
        ),
        tab="Data Drift",
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Feature Drift Analysis",
            subtitle="Per-feature drift detection",
            size="half",
            values=[PanelMetric(legend="Feature Drift", metric="DatasetDriftMetric")],
            plot_params={"plot_type": "bar"},
        ),
        tab="Data Drift",
    )

### LOAD THE DATA
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

    # Fix: Load validation data as numpy arrays and convert to DataFrame
    X_encoded_val_array = np.load('X_encoded_val.npy')
    target_val_array = np.load('target_val.npy')
    
    # Debug: Check shapes
    print(f"Training feature names shape: {feature_names.shape}")
    print(f"Validation data shape: {X_encoded_val_array.shape}")
    print(f"Training data shape: {X_encoded_train.shape}")
    
    # Handle potential shape mismatch
    if X_encoded_val_array.shape[1] != len(feature_names):
        print(f"WARNING: Shape mismatch detected!")
        print(f"Validation data has {X_encoded_val_array.shape[1]} features")
        print(f"Training data has {len(feature_names)} features")
        
        # Use the minimum number of features to avoid errors
        min_features = min(X_encoded_val_array.shape[1], len(feature_names))
        X_encoded_val_array = X_encoded_val_array[:, :min_features]
        feature_names_subset = feature_names[:min_features]
        
        print(f"Using first {min_features} features for compatibility")
        X_encoded_val = pd.DataFrame(X_encoded_val_array, columns=feature_names_subset)
        
        # Also update training data to match
        X_encoded_train = X_encoded_train.iloc[:, :min_features]
        feature_names = feature_names_subset
        
        print("Updated both datasets to have matching features")
    else:
        # Convert to DataFrame with same column names as training data
        X_encoded_val = pd.DataFrame(X_encoded_val_array, columns=feature_names)
    
    target_val = pd.Series(target_val_array, name='target')

    print("Using both training and validation data")

print("Loading model...")
with open('model_info.json', "r") as f:
    model_info = json.load(f)

model = joblib.load('model.pkl')

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
current_data['target'] = target_val.values
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
    ws = RemoteWorkspace(EVIDENTLY_SERVICE_URL)
    
    project_name = "Corn Yield ML Monitoring"  # Same unified project name
    project_description = "Comprehensive ML monitoring for corn yield prediction model - performance and drift analysis"
    
    project, is_new_project = get_or_create_project(ws, project_name, project_description)
    
    if is_new_project:
        # This should rarely happen since monitoring.py usually runs first
        setup_drift_dashboard(project)
        project.save()
        print("New project created with drift dashboard")
    else:
        print("Using existing project - no dashboard changes needed")

    # Create comprehensive report with both drift and regression metrics
    print("Creating drift and regression analysis report...")
    report = Report(metrics=[
        # Data Drift Metrics
        DataDriftPreset(),
        # Regression Performance Metrics  
        MAE(column="prediction"),
        RMSE(column="prediction"),
        R2Score(column="prediction"),
        MeanError(column="prediction"),
        # Data summary info
        DataSummaryPreset(),
    ])
    
    print("Running combined drift and regression analysis...")
    report_result = report.run(
        current_data=eval_data_current, 
        reference_data=eval_data_reference
    )
    
    # Add report to workspace
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_name = f'{timestamp}_drift_analysis'
    
    ws.add_run(project.id, report_result, include_data=False)
    print(f"Report '{report_name}' added to workspace successfully")
    
    # Extract drift results for analysis
    drift_results = report_result.get_metric_result("DatasetDriftMetric")
    drift_detected = drift_results.drift_detected if hasattr(drift_results, 'drift_detected') else False
    drift_score = getattr(drift_results, 'drift_score', 0.0) if hasattr(drift_results, 'drift_score') else 0.0
    
    # Export comprehensive metadata
    drift_metadata = {
        "project_id": str(project.id),
        "project_name": project.name,
        "report_name": report_name,
        "report_type": "DataDriftAnalysis",
        "description": project.description,
        "evidently_service_url": EVIDENTLY_SERVICE_URL,
        "evidently_ui_url": "https://evidently-ui-453290981886.us-central1.run.app/",
        "created_at": datetime.now().isoformat(),
        "dashboard_url": f"https://evidently-ui-453290981886.us-central1.run.app/projects/{project.id}",
        "is_new_project": is_new_project,
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
    print(f"Project: {project.name}")
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
    project = ws.create_project("Corn Yield ML Monitoring")
    project.description = "Comprehensive ML monitoring for corn yield prediction model"

    report = Report(metrics=[
        DataDriftPreset(),
        MAE(column="prediction"),
        RMSE(column="prediction"),
        R2Score(column="prediction"),
        MeanError(column="prediction"),
        DataSummaryPreset(),
    ])
    
    report_result = report.run(
        current_data=eval_data_current, 
        reference_data=eval_data_reference
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_name = f'{timestamp}_drift_analysis_local'
    
    ws.add_run(project.id, report_result, include_data=False)
    
    drift_metadata = {
        "project_id": str(project.id),
        "project_name": project.name,
        "report_name": report_name,
        "report_type": "DataDriftAnalysis",
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