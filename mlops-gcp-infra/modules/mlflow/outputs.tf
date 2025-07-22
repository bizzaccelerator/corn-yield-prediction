output "mlflow_url" {
  value = google_cloud_run_service.mlflow_server.status[0].url
}
