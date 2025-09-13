locals {
  bucket_name = var.bucket_name_override != "" ? var.bucket_name_override : "${var.project_id}-evidently-reports"
  image_tag   = random_id.image_tag.hex
  image_name  = "${var.region}-docker.pkg.dev/${var.project_id}/${var.repo_name}/${var.service_name}:${local.image_tag}"
}

# Enable APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "storage.googleapis.com"
  ])
  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

# Artifact Registry
resource "google_artifact_registry_repository" "repo" {
  project       = var.project_id
  location      = var.region
  repository_id = var.repo_name
  description   = "Docker images for MLOps services"
  format        = "DOCKER"
}

# Reports bucket
resource "google_storage_bucket" "reports" {
  name                        = local.bucket_name
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true
}

# Public read access to the reports bucket
resource "google_storage_bucket_iam_member" "public_read" {
  bucket = google_storage_bucket.reports.name
  role   = "roles/storage.objectViewer"
  member = "allUsers"
}

# Service account
resource "google_service_account" "sa" {
  project      = var.project_id
  account_id   = "${var.service_name}-sa"
  display_name = "Evidently UI Service Account"
}

# IAM for Artifact Registry and GCS
resource "google_project_iam_member" "ar_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.sa.email}"
}

resource "google_storage_bucket_iam_member" "bucket_rw" {
  bucket = google_storage_bucket.reports.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.sa.email}"
}

# Unique tag trigger
resource "random_id" "image_tag" {
  byte_length = 4
}

# Build & push Docker image using Cloud Build
resource "null_resource" "build_image" {
  triggers = {
    image_name = local.image_name
    dockerfile = filesha256("${path.module}/evidently-ui/Dockerfile")
    cloudbuild = filesha256("${path.module}/evidently-ui/cloudbuild.yaml")
  }

  provisioner "local-exec" {
    working_dir = "${path.module}/evidently-ui"

    # Force cross-platform shell execution
    interpreter = ["bash", "-c"]

    command = <<EOT
      gcloud builds submit \
        --config=cloudbuild.yaml \
        --project ${var.project_id} \
        --substitutions=_IMAGE_NAME=${local.image_name} \
        .
    EOT
  }

  depends_on = [
    google_artifact_registry_repository.repo,
    google_project_service.apis
  ]
}

# Deploy Cloud Run with GCS volume
resource "google_cloud_run_v2_service" "evidently" {
  name                 = var.service_name
  location             = var.region
  ingress              = "INGRESS_TRAFFIC_ALL"
  deletion_protection  = false

  template {
    service_account = google_service_account.sa.email

    containers {
      image = local.image_name

      ports {
        container_port = 8080
      }

      volume_mounts {
        name       = "reports"
        mount_path = "/workspace"
      }

      env {
        name  = "EVIDENTLY_WORKSPACE"
        value = "/workspace"
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }
    }

    volumes {
      name = "reports"
      gcs {
        bucket    = google_storage_bucket.reports.name
        read_only = false
      }
    }

    timeout = "3600s"
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    null_resource.build_image,
    google_storage_bucket_iam_member.bucket_rw,
    google_project_iam_member.ar_reader
  ]
}

# Public access
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  location = var.region
  name     = google_cloud_run_v2_service.evidently.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}