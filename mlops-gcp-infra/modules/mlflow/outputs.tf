output "mlflow_url" {
  value       = google_cloud_run_service.mlflow_server.status[0].url
  description = "URL of the MLflow server"
}

output "artifact_registry_url" {
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/mlops-repo"
  description = "Artifact Registry URL"
}