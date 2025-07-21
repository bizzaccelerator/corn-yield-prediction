output "artifact_bucket_url" {
  value = module.gcs.bucket_url
}

output "mlflow_url" {
  value = module.mlflow.mlflow_url
}

# output "evidently_url" {
#   value = module.monitoring.evidently_url
# }
