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
    # Keep complete training data as reference, no current data for model performance evaluation
    val_predictions = None
    X_encoded_val = None
    target_val = None
    print("Using complete training data as reference only - evaluating model performance on training data")

### CREATE THE REPORT DATASET FOR REGRESSION ANALYSIS
reference_data = X_encoded_train.copy()
reference_data['target'] = target_train.values
reference_data['prediction'] = train_predictions

if X_encoded_val is not None:
    # Use validation data as current if available
    current_data = X_encoded_val.copy()
    current_data['target'] = target_val.values
    current_data['prediction'] = val_predictions
    print("Using validation data as current dataset")
else:
    # No current data - only reference data for model performance evaluation
    current_data = None
    print("Using only reference data for model performance evaluation")

vectorized_features = list(X_encoded_train.columns)

# Define schema for regression analysis - Evidently will auto-detect "target" and "prediction" columns
schema = DataDefinition(
    regression=[Regression(target="target", prediction="prediction")],
    numerical_columns=vectorized_features,
    )

eval_data_1 = Dataset.from_pandas(pd.DataFrame(reference_data), data_definition=schema)
eval_data_2 = Dataset.from_pandas(pd.DataFrame(current_data), data_definition=schema) if current_data is not None else None

def dashboard_needs_setup(project):
    """Check if dashboard already has regression panels configured"""
    try:
        # Check if the project has any panels in the "Regression Analysis" tab
        for tab_name, panels in project.dashboard.panels.items():
            if tab_name == "Regression Analysis" and len(panels) > 0:
                return False
        return True
    except Exception as e:
        print(f"Error checking dashboard setup: {e}")
        return True

def setup_dashboard(project):
    """Setup dashboard panels for regression analysis"""
    print("Setting up dashboard panels...")
    
    ### ADD REGRESSION-SPECIFIC DASHBOARD PANELS
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Regression Performance Dashboard",
            subtitle="Model performance metrics for corn yield prediction",
            size="full",
            values=[],
            plot_params={"plot_type": "text"},
        ),
        tab="Regression Analysis",
    )

    # Mean Absolute Error panel
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Mean Absolute Error (MAE)",
            subtitle="MAE over time",
            size="half",
            values=[PanelMetric(
                legend="MAE",
                metric="MAE"
            )],
            plot_params={"plot_type": "bar", "is_stacked": False},
        ),
        tab="Regression Analysis",
    )

    # Mean Squared Error panel
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Root Mean Squared Error (RMSE)",
            subtitle="RMSE over time",
            size="half",
            values=[PanelMetric(
                legend="RMSE",
                metric="RMSE"
            )],
            plot_params={"plot_type": "bar", "is_stacked": False},
        ),
        tab="Regression Analysis",
    )

    # R-squared panel
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="R-squared Score",
            subtitle="R² coefficient of determination",
            size="half",
            values=[PanelMetric(
                legend="R²",
                metric="R2Score"
            )],
            plot_params={"plot_type": "bar", "is_stacked": False},
        ),
        tab="Regression Analysis",
    )

    # Mean Error panel
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Mean Error (ME)",
            subtitle="Mean error over time",
            size="half",
            values=[PanelMetric(
                legend="Mean Error",
                metric="MeanError"
            )],
            plot_params={"plot_type": "bar", "is_stacked": False},
        ),
        tab="Regression Analysis",
    )

### CREATE REMOTE WORKSPACE AND PROJECT
try:
    # Connect to remote Evidently workspace
    ws = RemoteWorkspace(EVIDENTLY_SERVICE_URL)
    
    # Try to get existing project or create new one
    project_name = "Corn Yield Regression Analysis"
    dashboard_was_setup = False
    
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
            
            # Only setup dashboard if it doesn't exist yet
            if dashboard_needs_setup(project):
                setup_dashboard(project)
                dashboard_was_setup = True
                print("Dashboard panels added to existing project")
            else:
                print("Dashboard already configured, skipping panel setup")
        else:
            project = ws.create_project(project_name)
            project.description = "Regression performance analysis for corn yield prediction model"
            setup_dashboard(project)
            dashboard_was_setup = True
            print(f"Created new project: {project.name}")
    except Exception as e:
        print(f"Error managing project: {e}")
        # Fallback: create project with timestamp
        project = ws.create_project(f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        project.description = "Regression performance analysis for corn yield prediction model"
        setup_dashboard(project)
        dashboard_was_setup = True

    # Create regression report using individual metrics instead of RegressionPreset
    if current_data is not None:
        # Use both reference and current data if validation data exists
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
        # Use training data as both reference and current for model performance evaluation
        report = Report(metrics=[
            MAE(column="prediction"),
            RMSE(column="prediction"),
            R2Score(column="prediction"),
            MeanError(column="prediction")
        ])
        print("Running regression analysis report using training data as both reference and current...")
        report_result = report.run(current_data=eval_data_1, reference_data=None)
        analysis_type = "training_only"
    
    # Add report to workspace
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_name = f'{timestamp}_corn_yield_regression_report_{analysis_type}'
    
    ws.add_run(project.id, report_result, include_data=False)
    print(f"Report '{report_name}' added to workspace successfully")
    
    # Save project only if dashboard was modified
    if dashboard_was_setup:
        project.save()
        print("Project saved with dashboard configuration")
    
    # Export project metadata for Kestra
    project_metadata = {
        "project_id": str(project.id),
        "project_name": project.name,
        "report_name": report_name,
        "report_type": "RegressionPreset",
        "description": project.description,
        "evidently_service_url": EVIDENTLY_SERVICE_URL,
        "evidently_ui_url": "https://evidently-ui-453290981886.us-central1.run.app/",
        "created_at": datetime.now().isoformat(),
        "dashboard_url": f"https://evidently-ui-453290981886.us-central1.run.app/projects/{project.id}",
        "dashboard_was_setup": dashboard_was_setup,
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
    
    print("Regression monitoring report and dashboard created successfully on remote Evidently service!")
    print(f"Project ID: {project.id}")
    print(f"Dashboard URL: {project_metadata['dashboard_url']}")
    print(f"Analysis type: {analysis_type}")
    print(f"Training samples analyzed: {len(X_encoded_train)}")
    if X_encoded_val is not None:
        print(f"Validation samples analyzed: {len(X_encoded_val)}")
    print(f"Features analyzed: {len(vectorized_features)}")
    
except Exception as e:
    print(f"Error connecting to remote Evidently service: {e}")
    print("Falling back to local workspace creation...")
    
    # Fallback to original local workspace creation
    from evidently.ui.workspace import Workspace
    
    ws = Workspace.create("workspace")
    project = ws.create_project("Corn Yield Regression Analysis")
    project.description = "Regression performance analysis for corn yield prediction model"

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
    report_name = f'{timestamp}_regression_report_{analysis_type}'
    
    ws.add_run(project.id, report_result, include_data=False)
    
    project_metadata = {
        "project_id": str(project.id),
        "project_name": project.name,
        "report_name": report_name,
        "report_type": "RegressionPreset",
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