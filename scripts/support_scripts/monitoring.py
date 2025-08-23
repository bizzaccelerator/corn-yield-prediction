from datetime import datetime
import os
import pandas as pd
import json
import numpy as np
import joblib

from evidently import Report
from evidently.ui.workspace import Workspace
from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import DashboardPanelPlot
from evidently.metrics import *

from evidently import Dataset
from evidently import DataDefinition
from evidently.presets import DataDriftPreset, DataSummaryPreset

### LOAD THE DATA

# Check if validation files exist (they are None in the input)
validation_files_exist = os.path.exists('X_encoded_val.csv')

if not validation_files_exist:       
    # Load training data from .npy files
    X_encoded_train = np.load('X_encoded.npy')
    target_train = np.load('y.npy')
    
    # Convert to DataFrame for evidently
    feature_names = np.load('feature_names.npy', allow_pickle=True)
    X_encoded_train = pd.DataFrame(X_encoded_train, columns=feature_names)
    target_train = pd.Series(target_train, name='target')

    X_encoded_val = None
    target_val = None
    
    print("Using training data only (no validation data provided)")
else:
    # Load training data from .npy files
    X_encoded_train = np.load('X_encoded.npy')
    target_train = np.load('y.npy')
    
    # Convert to DataFrame for evidently
    feature_names = np.load('feature_names.npy', allow_pickle=True)
    X_encoded_train = pd.DataFrame(X_encoded_train, columns=feature_names)
    target_train = pd.Series(target_train, name='target')

    # Load validation data from CSV files
    X_encoded_val = pd.read_csv('X_encoded_val.csv', sep=";")
    target_val = pd.read_csv('target_val.csv', sep=";")
    
    print("Using both training and validation data")

### LOAD THE MODEL AND VECTORIZER

# Load model info
with open('model_info.json', "r") as f:
    model_info = json.load(f)
      
# Load the model
model = joblib.load('model.pkl')

# Load the vectorizer
vectorizer = joblib.load('vectorizer.pkl')

### CREATE PREDICTIONS
# Use .values to convert DataFrame to numpy array for prediction (avoid feature name warning)
train_predictions = model.predict(X_encoded_train.values)

# Handle case when there's no validation data
if X_encoded_val is not None:
    val_predictions = model.predict(X_encoded_val.values)
    print("Created predictions for both training and validation data")
else:
    # Split training data to create reference and current datasets
    from sklearn.model_selection import train_test_split
    
    # Split the training data for monitoring purposes
    X_ref, X_cur, y_ref, y_cur = train_test_split(
        X_encoded_train, target_train, 
        test_size=0.3, random_state=42
    )
    
    # Use .values to avoid feature name warnings
    ref_predictions = model.predict(X_ref.values)
    cur_predictions = model.predict(X_cur.values)
    
    # Use split data as reference and current
    X_encoded_train = X_ref
    target_train = y_ref
    train_predictions = ref_predictions
    
    X_encoded_val = X_cur
    target_val = y_cur
    val_predictions = cur_predictions
    
    print("Split training data into reference and current datasets for monitoring")

### CREATE THE REPORT

# Step 1: Add predictions as a new column in a copy of data
reference_data = X_encoded_train.copy()
reference_data['target'] = target_train.values
reference_data['prediction'] = train_predictions

current_data = X_encoded_val.copy()  
current_data['target'] = target_val.values  
current_data['prediction'] = val_predictions

# Map the column types:

# Get all vectorized feature column names
vectorized_features = list(X_encoded_train.columns)

# Define the schema - all vectorized features are numerical
schema = DataDefinition(
    numerical_columns=vectorized_features + ["prediction"],  
    categorical_columns=[],
)

# Create Evidently Datasets to work with:
eval_data_1 = Dataset.from_pandas(
    pd.DataFrame(reference_data),
    data_definition=schema
)

eval_data_2 = Dataset.from_pandas(
    pd.DataFrame(current_data),
    data_definition=schema
)

# Create a report for specific metrics
report_2 = Report(metrics=[
    ValueDrift(column="prediction", method="ks"), # Method changed for exploratory purposes
    StdValue(column="prediction"), # Removed method parameter as it's not needed here
    QuantileValue(column="prediction") # Changed from "fare_amount" to "prediction"
])

# Evaluate the Drift for specific column
my_spec_eval = report_2.run(current_data=eval_data_2, reference_data=eval_data_1)

### CREATE THE DASHBOARD

# Create the workspace
ws = Workspace.create("workspace")
# Create a project 
project = ws.create_project("Corn Yield prediction")
project.description = "Predict the yield of corn in Kenya"

# Create SDK report
report = Report(metrics=[StdValue(column="prediction"), RowCount()])
report = report.run(current_data=eval_data_2, reference_data=eval_data_1)

# Save report to project
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
report_name = f'{timestamp}_report'
report.save_json(os.path.join(ws.path, str(project.id), f"{report_name}.json"))

# Adding the report to ws
ws.add_run(project.id, report, include_data=False)

# Creating dashboard
project.dashboard.add_panel(
    DashboardPanelPlot(
        title="Main Dashboard",
        size="full", 
        values=[], #leave empty
        plot_params={"plot_type": "text"},
    ),
    tab="My tab", #will create a Tab if there is no Tab with this name
)

project.dashboard.add_panel(
    DashboardPanelPlot(
        title="Standard Deviation",
        subtitle="Std deviation for prediction.",
        size="half",
        values=[
            PanelMetric(
                legend="Standard Deviation", 
                metric="StdValue",
                metric_labels={"column": "prediction"}
                )
            ],
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

print("Monitoring report and dashboard created successfully!")
print(f"Workspace path: {ws.path}")
print(f"Project ID: {project.id}")