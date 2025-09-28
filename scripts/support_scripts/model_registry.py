import json
import os

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

# Setup MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
client = MlflowClient()

# Load run information
with open("run_info.json", "r") as f:
    run_info = json.load(f)

model_name = os.getenv("MODEL_NAME")
run_id = run_info["mlflow_run_id"]
val_rmse = run_info["validation_metrics"]["rmse"]
val_r2 = run_info["validation_metrics"]["r2_score"]

print(f"Registering model: {model_name}")
print(f"Run ID: {run_id}")
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Validation R²: {val_r2:.4f}")

try:
    # First, register the model if it doesn't exist
    try:
        registered_model = client.get_registered_model(model_name)
        print(f"Model '{model_name}' already exists in registry")
    except mlflow.exceptions.RestException:
        print(f"Creating new registered model: {model_name}")
        client.create_registered_model(model_name)

    # Register this version
    model_version = client.create_model_version(
        name=model_name, source=f"runs:/{run_id}/model", run_id=run_id
    )

    version_number = model_version.version
    print(f"Registered model version: {version_number}")

    # Add validation metrics as tags to the model version
    try:
        client.set_model_version_tag(
            name=model_name,
            version=version_number,
            key="validation_rmse",
            value=str(val_rmse),
        )

        client.set_model_version_tag(
            name=model_name,
            version=version_number,
            key="validation_r2",
            value=str(val_r2),
        )

        client.set_model_version_tag(
            name=model_name,
            version=version_number,
            key="registered_by",
            value="kestra_pipeline",
        )

        client.set_model_version_tag(
            name=model_name,
            version=version_number,
            key="registration_date",
            value=str(pd.Timestamp.now()),
        )

        print("✅ Added validation metrics and metadata as tags")
    except Exception as e:
        print(f"⚠️  Warning: Could not add tags: {e}")

    # Model staging logic - promote through stages based on metrics
    rmse_threshold = 100.0  # Adjust based on your domain
    r2_threshold = 0.7  # Adjust based on your requirements

    print(f"🔍 Evaluating model quality...")
    print(f"   RMSE: {val_rmse:.4f} (threshold: {rmse_threshold})")
    print(f"   R²: {val_r2:.4f} (threshold: {r2_threshold})")

    if val_rmse < rmse_threshold and val_r2 > r2_threshold:
        print("✅ Model meets quality criteria - promoting to Staging")

        # First promote to Staging
        client.transition_model_version_stage(
            name=model_name,
            version=version_number,
            stage="Staging",
            archive_existing_versions=False,
        )

        # Add staging promotion tags
        try:
            client.set_model_version_tag(
                name=model_name,
                version=version_number,
                key="promotion_date",
                value=str(pd.Timestamp.now()),
            )

            client.set_model_version_tag(
                name=model_name,
                version=version_number,
                key="promoted_by",
                value="kestra_pipeline",
            )
        except Exception as e:
            print(f"⚠️  Warning: Could not add staging promotion tags: {e}")

        print(f"🎯 Model version {version_number} promoted to Staging")

        # Check if we should promote to Production
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])

        if len(staging_versions) == 1 and staging_versions[0].version == version_number:
            print("🔄 This is the only staging model - evaluating for Production...")

            # Check against current production model
            production_versions = client.get_latest_versions(
                model_name, stages=["Production"]
            )

            should_promote_to_prod = True

            if production_versions:
                # Compare with current production model
                prod_version = production_versions[0]
                print(f"📊 Found existing production version: {prod_version.version}")

                # Get production model metrics (safe tag handling)
                try:
                    prod_tags = {}
                    if hasattr(prod_version, "tags") and prod_version.tags:
                        if isinstance(prod_version.tags, dict):
                            prod_tags = prod_version.tags
                        elif hasattr(prod_version.tags, "__iter__"):
                            # Handle tags as iterable objects
                            for tag in prod_version.tags:
                                if hasattr(tag, "key") and hasattr(tag, "value"):
                                    prod_tags[tag.key] = tag.value
                                elif (
                                    isinstance(tag, dict)
                                    and "key" in tag
                                    and "value" in tag
                                ):
                                    prod_tags[tag["key"]] = tag["value"]

                    print(f"Production model tags available: {list(prod_tags.keys())}")

                    if "validation_rmse" in prod_tags:
                        prod_rmse = float(prod_tags["validation_rmse"])
                        improvement = (prod_rmse - val_rmse) / prod_rmse

                        print(f"📈 Production RMSE: {prod_rmse:.4f}")
                        print(f"📈 New model RMSE: {val_rmse:.4f}")
                        print(f"📊 Improvement: {improvement*100:.2f}%")

                        # Only promote if improvement is significant (>5%)
                        min_improvement = 0.05
                        should_promote_to_prod = improvement > min_improvement

                        if should_promote_to_prod:
                            print(
                                f"✅ Improvement {improvement*100:.2f}% > {min_improvement*100:.2f}% threshold"
                            )
                        else:
                            print(
                                f"❌ Improvement {improvement*100:.2f}% < {min_improvement*100:.2f}% threshold"
                            )
                    else:
                        print(
                            "⚠️  No validation_rmse tag found in production model - promoting anyway"
                        )

                except Exception as e:
                    print(f"⚠️  Warning: Could not parse production model tags: {e}")
                    print("🔄 Promoting to production anyway")
            else:
                print("🆕 No production model exists - this will be the first")

            if should_promote_to_prod:
                print(f"🚀 Promoting model version {version_number} to Production...")

                client.transition_model_version_stage(
                    name=model_name,
                    version=version_number,
                    stage="Production",
                    archive_existing_versions=True,  # Archive old production
                )

                # Add production promotion tags
                try:
                    client.set_model_version_tag(
                        name=model_name,
                        version=version_number,
                        key="production_promotion_date",
                        value=str(pd.Timestamp.now()),
                    )

                    client.set_model_version_tag(
                        name=model_name,
                        version=version_number,
                        key="promoted_by_pipeline",
                        value="true",
                    )
                except Exception as e:
                    print(f"⚠️  Warning: Could not add production promotion tags: {e}")

                print(
                    f"✅ Model version {version_number} successfully promoted to Production!"
                )
            else:
                print(
                    f"🎯 Model version {version_number} kept in Staging - insufficient improvement for Production"
                )
        else:
            print(
                f"🎯 Model version {version_number} promoted to Staging (other models also in staging)"
            )

    else:
        print("❌ Model does not meet quality criteria")
        print(f"   RMSE: {val_rmse:.4f} (required: < {rmse_threshold})")
        print(f"   R²: {val_r2:.4f} (required: > {r2_threshold})")

        # Keep in None stage or move to Archived
        client.transition_model_version_stage(
            name=model_name,
            version=version_number,
            stage="Archived",
            archive_existing_versions=False,
        )

        try:
            client.set_model_version_tag(
                name=model_name,
                version=version_number,
                key="archived_reason",
                value="did_not_meet_quality_criteria",
            )

            client.set_model_version_tag(
                name=model_name,
                version=version_number,
                key="archived_date",
                value=str(pd.Timestamp.now()),
            )
        except Exception as e:
            print(f"⚠️  Warning: Could not add archive tags: {e}")

        print(f"📦 Model version {version_number} archived")

    # Create model metadata for Docker deployment
    # Get the final stage of the model
    all_versions = client.search_model_versions(f"name='{model_name}'")
    final_versions = [mv for mv in all_versions if mv.version == version_number]
    final_stage = final_versions[0].current_stage if final_versions else "None"

    print(f"📋 Creating metadata for model in stage: {final_stage}")

    model_metadata = {
        "model_name": model_name,
        "model_version": version_number,
        "mlflow_run_id": run_id,
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
        "model_stage": final_stage,
        "validation_metrics": {"rmse": val_rmse, "r2_score": val_r2},
        "registration_timestamp": str(pd.Timestamp.now()),
        "model_uri": (
            f"models:/{model_name}/{final_stage}"
            if final_stage != "None"
            else f"models:/{model_name}/{version_number}"
        ),
    }

    with open("model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=2)

    print(f"📋 Created model_metadata.json for Docker deployment (stage: {final_stage})")

    # Print current model registry status
    print("\n📊 Current Model Registry Status:")
    all_versions = client.search_model_versions(f"name='{model_name}'")

    for stage in ["Production", "Staging", "Archived", "None"]:
        stage_versions = [mv for mv in all_versions if mv.current_stage == stage]
        if stage_versions:
            print(f"  {stage}:")
            for mv in stage_versions:
                try:
                    # Handle different tag formats safely
                    tags = {}
                    if hasattr(mv, "tags") and mv.tags:
                        if isinstance(mv.tags, dict):
                            tags = mv.tags
                        elif hasattr(mv.tags, "__iter__"):
                            for tag in mv.tags:
                                if hasattr(tag, "key") and hasattr(tag, "value"):
                                    tags[tag.key] = tag.value
                                elif (
                                    isinstance(tag, dict)
                                    and "key" in tag
                                    and "value" in tag
                                ):
                                    tags[tag["key"]] = tag["value"]

                    rmse_tag = tags.get("validation_rmse", "N/A")
                    r2_tag = tags.get("validation_r2", "N/A")
                    print(f"    Version {mv.version} - RMSE: {rmse_tag}, R²: {r2_tag}")
                except Exception as e:
                    print(f"    Version {mv.version} - Error reading tags: {e}")

except Exception as e:
    print(f"❌ Model registry registration failed: {e}")
    raise
