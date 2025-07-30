output "bucket_url" {
  value = "gs://${google_storage_bucket.ml_bucket.name}"
}

output "mlflow_bucket_name" {
  description = "Name of the MLflow GCS bucket"
  value       = google_storage_bucket.ml_bucket.name
}


output "kestra_bucket_name" {
  description = "Name of the Kestra GCS bucket"
  value       = google_storage_bucket.kestra_bucket.name
}

output "kestra_bucket_url" {
  description = "URL of the Kestra GCS bucket"
  value       = google_storage_bucket.kestra_bucket.url
}

