import mlflow
import mlflow.sklearn
import os
import json
import pickle
from mlflow.tracking import MlflowClient
   
# Setup MLflow
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
client = MlflowClient()

# Loading the vectorizer 
with open("dict_vectorizer", 'rb') as f_in:
    dv = pickle.load(f_in)
      
model_name = os.getenv('MODEL_NAME')
target_stage = os.getenv('TARGET_STAGE') 
performance_threshold = float(os.getenv('THRESHOLD')) 
force_deployment = (os.getenv('FORCE_DEPLOYMENT')) == "true"
      
print(f"Fetching {target_stage} model: {model_name}")
print(f"Performance threshold: {performance_threshold}")
print(f"Force deployment: {force_deployment}")
      
try:
    # Get the latest model version from target stage
    versions = client.get_latest_versions(model_name, stages=[target_stage])
          
    if not versions:
        # Fallback to Staging if Production not found
        if target_stage == "Production":
            print("No Production model found, trying Staging...")
            versions = client.get_latest_versions(model_name, stages=["Staging"])
              
        if not versions:
            raise Exception(f"No model found in {target_stage} or Staging")
          
    model_version = versions[0]
    print(f"Found model version: {model_version.version}")
          
    # Get run details to check performance metrics
    run_id = model_version.run_id
    run = client.get_run(run_id)
          
    # Extract performance metrics from both run metrics AND model version tags
    run_metrics = run.data.metrics
    performance_score = None
    
    # First, try to get metrics from model version tags (preferred method)
    print("Checking model version tags for performance metrics...")
    try:
        # Handle model version tags safely
        model_tags = {}
        if hasattr(model_version, 'tags') and model_version.tags:
            if isinstance(model_version.tags, dict):
                model_tags = model_version.tags
            elif hasattr(model_version.tags, '__iter__'):
                # Handle tags as iterable objects
                for tag in model_version.tags:
                    if hasattr(tag, 'key') and hasattr(tag, 'value'):
                        model_tags[tag.key] = tag.value
                    elif isinstance(tag, dict) and 'key' in tag and 'value' in tag:
                        model_tags[tag['key']] = tag['value']
        
        print(f"Model version tags: {list(model_tags.keys())}")
        
        # Try to find performance metrics in model version tags
        for metric_name in ['validation_r2', 'validation_rmse']:
            if metric_name in model_tags:
                metric_value = float(model_tags[metric_name])
                if metric_name == 'validation_rmse':
                    # For RMSE, lower is better, so we convert it to a "higher is better" score
                    # You might want to adjust this conversion based on your domain
                    performance_score = 1.0 / (1.0 + metric_value) if metric_value > 0 else 0
                else:  # validation_r2
                    performance_score = metric_value
                print(f"Found performance metric '{metric_name}' in model tags: {metric_value}")
                print(f"Converted performance score: {performance_score}")
                break
                
    except Exception as e:
        print(f"Error reading model version tags: {e}")
    
    # If no metrics found in model version tags, try run metrics as fallback
    if performance_score is None:
        print("No metrics found in model version tags, checking run metrics...")
        for metric_name in ['validation_r2', 'accuracy', 'f1_score', 'auc', 'validation_rmse']:
            if metric_name in run_metrics:
                if metric_name == 'validation_rmse':
                    # For RMSE, lower is better, so we invert the logic
                    performance_score = 1.0 / (1.0 + run_metrics[metric_name])
                else:
                    performance_score = run_metrics[metric_name]
                print(f"Found performance metric '{metric_name}' in run metrics: {performance_score}")
                break
          
    # Decision logic for deployment
    should_deploy = force_deployment or (performance_score and performance_score >= performance_threshold)
          
    if not should_deploy:
        if performance_score:
            print(f"Model performance ({performance_score:.3f}) below threshold ({performance_threshold})")
        else:
            print("No performance metrics found and force_deployment is False")
              
        # Create a decision file to control flow execution
        decision = {
            'should_deploy': False,
            'reason': f"Performance below threshold" if performance_score else "No performance metrics found",
            'performance_score': performance_score,
            'threshold': performance_threshold
        }
              
        with open('deployment_decision.json', 'w') as f:
            json.dump(decision, f, indent=2)
              
        print("Deployment cancelled due to performance criteria")
        exit(0)  # Exit gracefully but don't fail the task
          
    print(f"Deployment approved - Performance: {performance_score:.3f} >= {performance_threshold}")
          
    # Load and save the model
    model_uri = f"models:/{model_name}/{model_version.version}"
    loaded_model = mlflow.sklearn.load_model(model_uri)
          
    # Save model as bin for the web service
    with open(f'{model_name}_model.bin', 'wb') as f:
        pickle.dump((dv,loaded_model), f)
          
    # Create comprehensive metadata file
    metadata = {
        'model_name': model_name,
        'model_version': model_version.version,
        'model_stage': model_version.current_stage,
        'model_uri': model_uri,
        'mlflow_run_id': model_version.run_id,
        'performance_score': performance_score,
        'performance_threshold': performance_threshold,
        'deployment_approved': True,
        'run_metrics': run_metrics,
        'model_version_tags': model_tags if 'model_tags' in locals() else {}
    }
          
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
          
    # Create deployment decision file
    decision = {
        'should_deploy': True,
        'performance_score': performance_score,
        'threshold': performance_threshold,
        'model_version': model_version.version
    }
          
    with open('deployment_decision.json', 'w') as f:
        json.dump(decision, f, indent=2)
          
    print(f"Model {model_name} v{model_version.version} ready for deployment")
          
except Exception as e:
    print(f"Failed to fetch model: {e}")
    # Create decision file indicating failure
    decision = {
        'should_deploy': False,
        'reason': f"Error: {str(e)}",
        'performance_score': None,
        'threshold': performance_threshold
    }
          
    with open('deployment_decision.json', 'w') as f:
        json.dump(decision, f, indent=2)
    raise