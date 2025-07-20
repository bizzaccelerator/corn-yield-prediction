resource "google_storage_bucket" "ml_bucket" {
  name     = var.bucket_name
  location = var.region
  force_destroy = true
  uniform_bucket_level_access = true
}

resource "google_storage_bucket_object" "example" {
  name   = "README.md"
  bucket = google_storage_bucket.ml_bucket.name
  content = "This bucket is for ML artifacts and Evidently reports"
}
