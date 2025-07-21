output "mlflow_url" {
  value = google_cloud_run_service.mlflow_service.status[0].url
}
