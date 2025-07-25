# Habilitar APIs necesarias
resource "google_project_service" "required_apis" {
  for_each = toset([
    "cloudresourcemanager.googleapis.com",
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "sqladmin.googleapis.com"
  ])

  project = var.project_id
  service = each.value
}

# Artifact Registry
resource "google_artifact_registry_repository" "mlops_repo" {
  location      = var.region
  repository_id = "mlops-repo"
  description   = "Repository for MLOps containers"
  format        = "DOCKER"

  depends_on = [google_project_service.required_apis]
}

# Cloud SQL (PostgreSQL)
resource "google_sql_database_instance" "mlflow_db" {
  name                = var.db_instance_name
  database_version    = "POSTGRES_14"
  region              = var.region
  deletion_protection = false

  settings {
    tier = "db-f1-micro"

    database_flags {
      name  = "cloudsql.iam_authentication"
      value = "on"
    }
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_sql_database" "mlflow_database" {
  name     = var.db_name
  instance = google_sql_database_instance.mlflow_db.name
}

resource "google_sql_user" "mlflow_user" {
  name     = var.db_user
  instance = google_sql_database_instance.mlflow_db.name
  password = var.db_password
}

# Service Account
resource "google_service_account" "mlflow_sa" {
  account_id   = "${var.mlflow_service_name}-sa"
  display_name = "MLflow Service Account"
}

# IAM para el Service Account
resource "google_project_iam_member" "mlflow_permissions" {
  for_each = toset([
    "roles/storage.admin",
    "roles/cloudsql.client",
    "roles/artifactregistry.reader"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.mlflow_sa.email}"
}

# Build inicial (null_resource para ejecutar una vez)
resource "null_resource" "initial_build" {
  provisioner "local-exec" {
    working_dir = "${path.module}/mlflow-server"
    command     = "gcloud builds submit . --config=cloudbuild.yml --substitutions=_IMAGE_TAG=latest,_REGION=us-central1,_REPO_NAME=mlops-repo --project=corn-yield-prediction-kenia"
  }

  depends_on = [
    google_artifact_registry_repository.mlops_repo,
    google_project_service.required_apis
  ]

  triggers = {
    dockerfile_hash = filemd5("${path.module}/mlflow-server/Dockerfile")
    entrypoint_hash = filemd5("${path.module}/mlflow-server/entrypoint.sh")
  }
}



# Cloud Run Service
resource "google_cloud_run_service" "mlflow_server" {
  name     = var.mlflow_service_name
  location = var.region

  template {
    spec {
      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/mlops-repo/mlflow-server:latest"

        ports {
          container_port = 8080
        }

        env {
          name  = "MLFLOW_BACKEND_URI"
          value = "postgresql://${var.db_user}:${var.db_password}@/${var.db_name}?host=/cloudsql/${var.project_id}:${var.region}:${var.db_instance_name}"
        }

        env {
          name  = "MLFLOW_ARTIFACT_ROOT"
          value = "gs://${var.bucket_name}/mlflow-artifacts"
        }

        resources {
          limits = {
            cpu    = "1000m"
            memory = "1Gi"
          }
        }
      }

      service_account_name = google_service_account.mlflow_sa.email
    }

    metadata {
      annotations = {
        "run.googleapis.com/cloudsql-instances" = "${var.project_id}:${var.region}:${var.db_instance_name}"
        "autoscaling.knative.dev/maxScale"      = "10"
        "run.googleapis.com/execution-environment" = "gen2"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [
    google_project_service.required_apis,
    google_sql_database_instance.mlflow_db,
    google_artifact_registry_repository.mlops_repo
  ]
}

# IAM para acceso público
resource "google_cloud_run_service_iam_member" "public_access" {
  service  = google_cloud_run_service.mlflow_server.name
  location = google_cloud_run_service.mlflow_server.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# IAM binding condicional para acceso público
resource "google_cloud_run_service_iam_binding" "mlflow_public_access" {
  count = var.allow_public_access ? 1 : 0
  
  service  = google_cloud_run_service.mlflow_server.name
  location = google_cloud_run_service.mlflow_server.location
  role     = "roles/run.invoker"
  
  members = [
    "allUsers"
  ]
}