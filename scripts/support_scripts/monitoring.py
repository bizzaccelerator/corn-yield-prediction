from datetime import datetime
import os
import pandas as pd
import json
import numpy as np
import joblib
import shutil

from evidently import Report, Dataset, DataDefinition
from evidently.ui.workspace import Workspace
from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import DashboardPanelPlot
from evidently.metrics import *
from evidently.presets import DataDriftPreset, DataSummaryPreset
from sklearn.model_selection import train_test_split

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

### CREATE THE REPORT DATASETS

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

### EXAMPLE METRIC REPORT

report_2 = Report(metrics=[
    ValueDrift(column="prediction", method="ks"),
    StdValue(column="prediction"),
    QuantileValue(column="prediction")
])

my_spec_eval = report_2.run(current_data=eval_data_2, reference_data=eval_data_1)

### CREATE WORKSPACE AND PROJECT

ws = Workspace.create("workspace")
project = ws.create_project("Corn Yield prediction")
project.description = "Predict the yield of corn in Kenya"

report = Report(metrics=[StdValue(column="prediction"), RowCount()])
report = report.run(current_data=eval_data_2, reference_data=eval_data_1)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
report_name = f'{timestamp}_report'
report.save_json(os.path.join(ws.path, str(project.id), f"{report_name}.json"))

ws.add_run(project.id, report, include_data=False)

### ADD DASHBOARD PANELS

project.dashboard.add_panel(
    DashboardPanelPlot(
        title="Main Dashboard",
        size="full",
        values=[],
        plot_params={"plot_type": "text"},
    ),
    tab="My tab",
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
    tab="My tab",
)

project.dashboard.add_panel(
    DashboardPanelPlot(
        title="Row count",
        subtitle="Total number of evaluations over time.",
        size="half",
        values=[PanelMetric(legend="Row count", metric="RowCount")],
        plot_params={"plot_type": "counter", "aggregation": "sum"},
    ),
    tab="My tab",
)

project.save()

### EXPORT PROJECT METADATA (hybrid: try copy, else fallback)

project_json_path = os.path.join(ws.path, str(project.id), "project.json")
if os.path.exists(project_json_path):
    shutil.copy(project_json_path, "project_metadata.json")
    print("Exported Evidently project metadata from project.json")
else:
    project_metadata = {
        "project_id": str(project.id),
        "project_name": project.name,
        "description": project.description,
        "workspace_path": ws.path,
        "created_at": datetime.now().isoformat()
    }
    with open("project_metadata.json", "w") as f:
        json.dump(project_metadata, f, indent=2)
    print("Generated project_metadata.json manually (project.json not found)")


# Exporting the workspace folder
workspace_dir = "workspace"
zip_path = "workspace.zip"

if os.path.exists(workspace_dir):
    shutil.make_archive("workspace", "zip", workspace_dir)
    print(f"Workspace compressed: {zip_path}")
else:
    print("No workspace directory found to compress.")

print("Monitoring report and dashboard created successfully!")
print(f"Workspace path: {ws.path}")
print(f"Project ID: {project.id}")
