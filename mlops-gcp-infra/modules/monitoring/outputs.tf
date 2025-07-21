output "evidently_url" {
  value = google_cloud_run_service.evidently_service.status[0].url
}