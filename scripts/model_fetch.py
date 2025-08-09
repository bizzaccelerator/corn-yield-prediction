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
improvement_threshold = float(os.getenv('IMPROVEMENT_THRESHOLD', '0.02'))  # Default 2% improvement needed
      
print(f"Fetching {target_stage} model: {model_name}")
print(f"Performance threshold: {performance_threshold}")
print(f"Improvement threshold: {improvement_threshold}")
print(f"Force deployment: {force_deployment}")

def extract_performance_metrics(model_version, run_metrics=None):
    """
    Extract performance metrics from model version tags or run metrics
    
    Returns:
        dict: Dictionary with 'r2_score', 'rmse', and 'performance_score'
    """
    performance_data = {'r2_score': None, 'rmse': None, 'performance_score': None}
    
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
    
    # Try to find metrics in model version tags first
    if 'validation_r2' in model_tags:
        performance_data['r2_score'] = float(model_tags['validation_r2'])
        performance_data['performance_score'] = performance_data['r2_score']
    
    if 'validation_rmse' in model_tags:
        performance_data['rmse'] = float(model_tags['validation_rmse'])
        # If no R2 score, use RMSE as performance score (convert to "higher is better")
        if performance_data['performance_score'] is None:
            performance_data['performance_score'] = 1.0 / (1.0 + performance_data['rmse']) if performance_data['rmse'] > 0 else 0
    
    # Fallback to run metrics if no model version tags found
    if performance_data['performance_score'] is None and run_metrics:
        for metric_name in ['validation_r2', 'accuracy', 'f1_score', 'auc']:
            if metric_name in run_metrics:
                performance_data['performance_score'] = run_metrics[metric_name]
                if metric_name == 'validation_r2':
                    performance_data['r2_score'] = run_metrics[metric_name]
                break
        
        # Check for RMSE in run metrics
        if 'validation_rmse' in run_metrics:
            performance_data['rmse'] = run_metrics['validation_rmse']
            if performance_data['performance_score'] is None:
                performance_data['performance_score'] = 1.0 / (1.0 + performance_data['rmse'])
    
    return performance_data, model_tags

def compare_models(candidate_metrics, production_metrics):
    """
    Compare candidate model with production model
    
    Returns:
        dict: Comparison results with improvement information
    """
    comparison = {
        'is_better': False,
        'improvement_pct': 0.0,
        'reason': 'No comparison possible',
        'metric_used': 'unknown'
    }
    
    if not production_metrics or production_metrics['performance_score'] is None:
        comparison['is_better'] = True  # No production model, so candidate wins
        comparison['reason'] = 'No production model to compare against'
        return comparison
    
    if candidate_metrics['performance_score'] is None:
        comparison['reason'] = 'Candidate model has no performance metrics'
        return comparison
    
    candidate_score = candidate_metrics['performance_score']
    production_score = production_metrics['performance_score']
    
    # Calculate improvement percentage
    improvement = (candidate_score - production_score) / abs(production_score) * 100 if production_score != 0 else 0
    comparison['improvement_pct'] = improvement
    
    # Determine if candidate is better
    comparison['is_better'] = candidate_score > production_score
    
    # Use more specific metrics if available
    if candidate_metrics['r2_score'] is not None and production_metrics['r2_score'] is not None:
        candidate_r2 = candidate_metrics['r2_score']
        production_r2 = production_metrics['r2_score']
        r2_improvement = (candidate_r2 - production_r2) / abs(production_r2) * 100 if production_r2 != 0 else 0
        
        comparison['metric_used'] = 'r2_score'
        comparison['improvement_pct'] = r2_improvement
        comparison['is_better'] = candidate_r2 > production_r2
        comparison['reason'] = f"R² Score: {candidate_r2:.4f} vs {production_r2:.4f} (improvement: {r2_improvement:.2f}%)"
    
    elif candidate_metrics['rmse'] is not None and production_metrics['rmse'] is not None:
        candidate_rmse = candidate_metrics['rmse']
        production_rmse = production_metrics['rmse']
        rmse_improvement = (production_rmse - candidate_rmse) / production_rmse * 100  # Lower RMSE is better
        
        comparison['metric_used'] = 'rmse'
        comparison['improvement_pct'] = rmse_improvement
        comparison['is_better'] = candidate_rmse < production_rmse
        comparison['reason'] = f"RMSE: {candidate_rmse:.4f} vs {production_rmse:.4f} (improvement: {rmse_improvement:.2f}%)"
    
    else:
        comparison['reason'] = f"Performance Score: {candidate_score:.4f} vs {production_score:.4f} (improvement: {improvement:.2f}%)"
    
    return comparison

try:
    # Get the candidate model (from target stage)
    candidate_versions = client.get_latest_versions(model_name, stages=[target_stage])
          
    if not candidate_versions:
        # Fallback to Staging if Production not found
        if target_stage == "Production":
            print("No Production model found, trying Staging...")
            candidate_versions = client.get_latest_versions(model_name, stages=["Staging"])
              
        if not candidate_versions:
            raise Exception(f"No model found in {target_stage} or Staging")
          
    candidate_version = candidate_versions[0]
    print(f"Found candidate model version: {candidate_version.version} in stage: {candidate_version.current_stage}")
    
    # Get candidate model performance
    candidate_run = client.get_run(candidate_version.run_id)
    candidate_metrics, candidate_tags = extract_performance_metrics(candidate_version, candidate_run.data.metrics)
    
    print(f"Candidate model performance: {candidate_metrics}")
    
    # Get current production model for comparison (if exists)
    production_metrics = {'performance_score': None, 'r2_score': None, 'rmse': None}
    production_version = None
    
    try:
        production_versions = client.get_latest_versions(model_name, stages=["Production"])
        if production_versions:
            production_version = production_versions[0]
            print(f"Found current production model version: {production_version.version}")
            
            production_run = client.get_run(production_version.run_id)
            production_metrics, production_tags = extract_performance_metrics(production_version, production_run.data.metrics)
            print(f"Production model performance: {production_metrics}")
        else:
            print("No current production model found - this will be the first deployment")
    except Exception as e:
        print(f"Could not fetch production model: {e}")
    
    # Performance evaluation
    print("\n" + "="*60)
    print("DEPLOYMENT DECISION ANALYSIS")
    print("="*60)
    
    # Check 1: Performance threshold
    meets_threshold = candidate_metrics['performance_score'] and candidate_metrics['performance_score'] >= performance_threshold
    print(f"Threshold Check: {'PASS' if meets_threshold else 'FAIL'}")
    if candidate_metrics['performance_score']:
        print(f"  Model Score: {candidate_metrics['performance_score']:.4f}")
        print(f"  Threshold:   {performance_threshold:.4f}")
    else:
        print(f"  No performance metrics found")
    
    # Check 2: Improvement over production
    comparison = compare_models(candidate_metrics, production_metrics)
    meets_improvement = comparison['is_better'] and comparison['improvement_pct'] >= improvement_threshold
    
    print(f"Improvement Check: {'PASS' if meets_improvement else 'FAIL'}")
    print(f"  {comparison['reason']}")
    if production_version:
        print(f"  Required improvement: {improvement_threshold:.1f}%")
        print(f"  Actual improvement:   {comparison['improvement_pct']:.2f}%")
    
    # Final deployment decision
    should_deploy = force_deployment or (meets_threshold and meets_improvement)
    
    print(f"\nFINAL DECISION: {'DEPLOY' if should_deploy else 'DO NOT DEPLOY'}")
    
    deployment_reason = []
    if force_deployment:
        deployment_reason.append("Force deployment enabled")
    else:
        if not meets_threshold:
            deployment_reason.append(f"Performance below threshold ({candidate_metrics['performance_score']:.3f} < {performance_threshold})")
        if not meets_improvement:
            deployment_reason.append(f"Insufficient improvement over production ({comparison['improvement_pct']:.2f}% < {improvement_threshold}%)")
        if meets_threshold and meets_improvement:
            deployment_reason.append("Meets both threshold and improvement criteria")
    
    print(f"REASON: {'; '.join(deployment_reason)}")
    print("="*60)
          
    if not should_deploy:
        # Create a decision file to control flow execution
        decision = {
            'should_deploy': False,
            'reason': '; '.join(deployment_reason),
            'candidate_performance': candidate_metrics,
            'production_performance': production_metrics,
            'threshold': performance_threshold,
            'improvement_threshold': improvement_threshold,
            'comparison': comparison
        }
              
        with open('deployment_decision.json', 'w') as f:
            json.dump(decision, f, indent=2)
              
        print("Deployment cancelled due to performance criteria")
        exit(0)  # Exit gracefully but don't fail the task
          
    print(f"Deployment approved!")
          
    # Load and save the model
    model_uri = f"models:/{model_name}/{candidate_version.version}"
    loaded_model = mlflow.sklearn.load_model(model_uri)
          
    # Save model as bin for the web service
    with open(f'{model_name}_model.bin', 'wb') as f:
        pickle.dump((dv, loaded_model), f)
          
    # Create comprehensive metadata file
    metadata = {
        'model_name': model_name,
        'model_version': candidate_version.version,
        'model_stage': candidate_version.current_stage,
        'model_uri': model_uri,
        'mlflow_run_id': candidate_version.run_id,
        'candidate_performance': candidate_metrics,
        'production_performance': production_metrics,
        'performance_threshold': performance_threshold,
        'improvement_threshold': improvement_threshold,
        'deployment_approved': True,
        'comparison_results': comparison,
        'candidate_run_metrics': candidate_run.data.metrics,
        'candidate_model_tags': candidate_tags,
        'production_version': production_version.version if production_version else None
    }
          
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
          
    # Create deployment decision file
    decision = {
        'should_deploy': True,
        'candidate_performance': candidate_metrics,
        'production_performance': production_metrics,
        'threshold': performance_threshold,
        'improvement_threshold': improvement_threshold,
        'model_version': candidate_version.version,
        'comparison': comparison,
        'reason': '; '.join(deployment_reason)
    }
          
    with open('deployment_decision.json', 'w') as f:
        json.dump(decision, f, indent=2)
          
    print(f"Model {model_name} v{candidate_version.version} ready for deployment")
          
except Exception as e:
    print(f"Failed to fetch model: {e}")
    # Create decision file indicating failure
    decision = {
        'should_deploy': False,
        'reason': f"Error: {str(e)}",
        'candidate_performance': None,
        'production_performance': None,
        'threshold': performance_threshold,
        'improvement_threshold': improvement_threshold
    }
          
    with open('deployment_decision.json', 'w') as f:
        json.dump(decision, f, indent=2)
    raise