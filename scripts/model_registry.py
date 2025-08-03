import mlflow
import json
import os
import pandas as pd
from mlflow.tracking import MlflowClient
      
# Setup MLflow
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
client = MlflowClient()
      
# Load run information
with open('run_info.json', 'r') as f:
    run_info = json.load(f)
      
model_name = os.getenv('MODEL_NAME')
run_id = run_info['mlflow_run_id']
val_rmse = run_info['validation_metrics']['rmse']
val_r2 = run_info['validation_metrics']['r2_score']
      
print(f"Managing model: {model_name}")
print(f"Run ID: {run_id}")
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Validation R²: {val_r2:.4f}")
      
try:
    # Get all versions of the model
    model_versions = client.search_model_versions(
        filter_string=f"name='{model_name}'"
    )
          
    if not model_versions:
        print(f"No versions found for model '{model_name}'")
        exit(1)
          
    # Find the current version (should be the one we just created)
    current_version = None
    for mv in model_versions:
        if mv.run_id == run_id:
            current_version = mv
            break
          
    if not current_version:
        print(f"Could not find model version for run {run_id}")
        exit(1)
          
    version_number = current_version.version
    print(f"Found model version: {version_number}")
    
    # Add validation metrics as tags to the model version
    try:
        client.set_model_version_tag(
            name=model_name,
            version=version_number,
            key="validation_rmse",
            value=str(val_rmse)
        )
        
        client.set_model_version_tag(
            name=model_name,
            version=version_number,
            key="validation_r2",
            value=str(val_r2)
        )
        print("Added validation metrics as tags")
    except Exception as e:
        print(f"Warning: Could not add validation metrics tags: {e}")
          
    # Model validation logic - promote if metrics are good
    rmse_threshold = 100.0  # Adjust based on your domain
    r2_threshold = 0.7      # Adjust based on your requirements
          
    if val_rmse < rmse_threshold and val_r2 > r2_threshold:
        print("Model meets quality criteria - promoting to Staging")
              
        # Transition to Staging
        client.transition_model_version_stage(
            name=model_name,
            version=version_number,
            stage="Staging",
            archive_existing_versions=False
        )
              
        # Add tags for the promotion
        try:
            client.set_model_version_tag(
                name=model_name,
                version=version_number,
                key="promotion_date",
                value=str(pd.Timestamp.now())
            )
                  
            client.set_model_version_tag(
                name=model_name,
                version=version_number,
                key="promoted_by",
                value="kestra_pipeline"
            )
        except Exception as e:
            print(f"Warning: Could not add promotion tags: {e}")
              
        print(f"Model version {version_number} promoted to Staging")
              
        # Check if we should promote to Production
        staging_versions = client.get_latest_versions(
            model_name, stages=["Staging"]
        )
              
        if len(staging_versions) == 1 and staging_versions[0].version == version_number:
            print("This is the only staging model - consider promoting to Production")
                  
            # Auto-promote to production if it's significantly better
            production_versions = client.get_latest_versions(
                model_name, stages=["Production"]
            )
                  
            should_promote_to_prod = True
                  
            if production_versions:
                # Compare with current production model
                prod_version = production_versions[0]
                print(f"Found existing production version: {prod_version.version}")
                      
                # Get production model metrics (FIXED - safe tag handling)
                try:
                    prod_tags = {}
                    if hasattr(prod_version, 'tags') and prod_version.tags:
                        if isinstance(prod_version.tags, dict):
                            prod_tags = prod_version.tags
                        elif hasattr(prod_version.tags, '__iter__'):
                            # Handle tags as iterable objects
                            for tag in prod_version.tags:
                                if hasattr(tag, 'key') and hasattr(tag, 'value'):
                                    prod_tags[tag.key] = tag.value
                                elif isinstance(tag, dict) and 'key' in tag and 'value' in tag:
                                    prod_tags[tag['key']] = tag['value']
                    
                    print(f"Production model tags: {list(prod_tags.keys())}")
                    
                    if "validation_rmse" in prod_tags:
                        prod_rmse = float(prod_tags["validation_rmse"])
                        improvement = (prod_rmse - val_rmse) / prod_rmse
                              
                        print(f"Production RMSE: {prod_rmse:.4f}")
                        print(f"New model RMSE: {val_rmse:.4f}")
                        print(f"Improvement: {improvement*100:.2f}%")
                              
                        # Only promote if improvement is significant (>5%)
                        should_promote_to_prod = improvement > 0.05
                    else:
                        print("No validation_rmse tag found in production model - promoting anyway")
                        
                except Exception as e:
                    print(f"Warning: Could not parse production model tags: {e}")
                    print("Promoting to production anyway")
                      
            if should_promote_to_prod:
                client.transition_model_version_stage(
                    name=model_name,
                    version=version_number,
                    stage="Production",
                    archive_existing_versions=True  # Archive old production
                )
                      
                try:
                    client.set_model_version_tag(
                        name=model_name,
                        version=version_number,
                        key="production_promotion_date",
                        value=str(pd.Timestamp.now())
                    )
                except Exception as e:
                    print(f"Warning: Could not add production promotion tag: {e}")
                      
                print(f"🏆 Model version {version_number} promoted to Production!")
            else:
                print("⏸️  Model kept in Staging - not enough improvement for Production")
              
    else:
        print("Model does not meet quality criteria")
        print(f" RMSE: {val_rmse:.4f} (threshold: {rmse_threshold})")
        print(f" R²: {val_r2:.4f} (threshold: {r2_threshold})")
              
        # Keep in None stage or move to Archived
        client.transition_model_version_stage(
            name=model_name,
            version=version_number,
            stage="Archived",
            archive_existing_versions=False
        )
              
        try:
            client.set_model_version_tag(
                name=model_name,
                version=version_number,
                key="archived_reason",
                value="did_not_meet_quality_criteria"
            )
        except Exception as e:
            print(f"Warning: Could not add archive tag: {e}")
              
        print(f"Model version {version_number} archived")
    
    # Print current model registry status (FIXED - safe tag handling)
    print("\nCurrent Model Registry Status:")
    all_versions = client.search_model_versions(f"name='{model_name}'")
    
    for stage in ["Production", "Staging", "Archived", "None"]:
        stage_versions = [mv for mv in all_versions if mv.current_stage == stage]
        if stage_versions:
            print(f"  {stage}:")
            for mv in stage_versions:
                try:
                    # Handle different tag formats safely
                    tags = {}
                    if hasattr(mv, 'tags') and mv.tags:
                        if isinstance(mv.tags, dict):
                            tags = mv.tags
                        elif hasattr(mv.tags, '__iter__'):
                            # Tags are iterable objects with key/value attributes
                            for tag in mv.tags:
                                if hasattr(tag, 'key') and hasattr(tag, 'value'):
                                    tags[tag.key] = tag.value
                                elif isinstance(tag, dict) and 'key' in tag and 'value' in tag:
                                    tags[tag['key']] = tag['value']
                        
                    rmse_tag = tags.get("validation_rmse", "N/A")
                    r2_tag = tags.get("validation_r2", "N/A")
                    print(f"    Version {mv.version} - RMSE: {rmse_tag}, R²: {r2_tag}")
                except Exception as e:
                    print(f"    Version {mv.version} - Error reading tags: {e}")
                    print(f"    Version {mv.version} - Basic info only")
      
except Exception as e:
    print(f"Model registry management failed: {e}")
    raise