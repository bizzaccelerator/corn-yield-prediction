output "evidently_url" {
  value       = google_cloud_run_v2_service.evidently.uri
  description = "Public URL of Evidently UI"
}

output "evidently_bucket" {
  value       = google_storage_bucket.reports.name
  description = "GCS bucket for Evidently reports"
}

output "evidently_bucket_url" {
  value       = "https://storage.googleapis.com/${google_storage_bucket.reports.name}"
  description = "Public URL to access the GCS bucket contents"
}