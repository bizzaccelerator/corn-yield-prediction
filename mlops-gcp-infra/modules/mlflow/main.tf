provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_storage_bucket" "mlflow_bucket" {
  name                        = var.bucket_name
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [name]
  }
}

resource "google_sql_database_instance" "mlflow_instance" {
  name             = var.db_instance_name
  database_version = "POSTGRES_13"
  region           = var.region

  deletion_protection = false

  settings {
    tier = "db-f1-micro"
    ip_configuration {
      ipv4_enabled = true
      authorized_networks {
        name  = "allow-all"
        value = "0.0.0.0/0"
      }
    }
  }
}

resource "google_sql_user" "mlflow_user" {
  name     = var.db_user
  instance = google_sql_database_instance.mlflow_instance.name
  password = var.db_password
}

resource "google_sql_database" "mlflow_db" {
  name     = var.db_name
  instance = google_sql_database_instance.mlflow_instance.name
}

resource "google_cloud_run_service" "mlflow_service" {
  name     = "mlflow-server"
  location = var.region

  template {
    spec {
      containers {
        image = "ghcr.io/mlflow/mlflow:latest"

        env {
          name  = "BACKEND_STORE_URI"
          value = "postgresql://${var.db_user}:${var.db_password}@${google_sql_database_instance.mlflow_instance.ip_address[0].ip_address}/${var.db_name}"
        }

        env {
          name  = "ARTIFACT_ROOT"
          value = "gs://${google_storage_bucket.mlflow_bucket.name}/artifacts"
        }

        ports {
          container_port = 5000
        }
      }
    }

    metadata {
      annotations = {
        "run.googleapis.com/client-name" = "terraform"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  autogenerate_revision_name = true
}

resource "google_cloud_run_service_iam_member" "invoker" {
  location = google_cloud_run_service.mlflow_service.location
  service  = google_cloud_run_service.mlflow_service.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_project_service" "cloudbuild" {
  service = "cloudbuild.googleapis.com"
}

resource "google_project_service" "artifactregistry" {
  service = "artifactregistry.googleapis.com"
}

resource "google_cloudbuild_trigger" "mlflow_server_trigger" {
  name     = "mlflow-server-trigger"
  location = "global"

  github {
    owner = var.github_owner
    name  = var.github_repo
    push {
      branch = "^main$"
    }
  }

  filename = "mlops-gcp-infra/modules/mlflow/mlflow-server/cloudbuild.yaml"
  included_files = ["mlops-gcp-infra/modules/mlflow/mlflow-server/**"]

  substitutions = {
    _PROJECT_ID   = var.project_id
    _REGION       = var.region
    _SERVICE_NAME = var.mlflow_service_name
  }
}
