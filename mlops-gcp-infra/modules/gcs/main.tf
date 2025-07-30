# CAMBIO: Versión simplificada que garantiza la creación del bucket
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "google_storage_bucket" "ml_bucket" {
  name                        = var.bucket_name
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true
}

# CAMBIO: Bucket de Kestra simplificado sin reglas de lifecycle complejas
resource "google_storage_bucket" "kestra_bucket" {
  name     = "${replace(var.project_id, "-", "")}-kestra-${random_id.bucket_suffix.hex}"
  location = var.region

  force_destroy = true

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
}

