# Habilitar APIs necesarias
resource "google_project_service" "required_apis" {
  for_each = toset([
    "cloudresourcemanager.googleapis.com",
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "sqladmin.googleapis.com",
    "iam.googleapis.com"
  ])

  project = var.project_id
  service = each.value

  disable_dependent_services = false
  disable_on_destroy         = false
  
  timeouts {
    create = "10m"
    update = "10m"
    delete = "10m"
  }
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

# PASO 1: SOLO Build y Push de la imagen (SIN DEPLOY)
resource "null_resource" "mlflow_image_build" {
  # Triggers: reconstruye solo si cambian los archivos fuente
  triggers = {
    dockerfile_hash    = filemd5("${path.module}/mlflow-server/Dockerfile")
    entrypoint_hash    = filemd5("${path.module}/mlflow-server/entrypoint.sh")
    requirements_hash  = filemd5("${path.module}/mlflow-server/requirements.txt")
    cloudbuild_hash    = filemd5("${path.module}/mlflow-server/cloudbuild.yml")
    # Para forzar manualmente un rebuild, puedes descomentar esto:
    # force_rebuild = timestamp()
  }

  provisioner "local-exec" {
    working_dir = "${path.module}/mlflow-server"

    #Usa Bash explícitamente como intérprete
    interpreter = ["bash", "-c"]

    command = <<-EOF
      echo "Iniciando build de imagen MLflow..."
      echo "Directorio de trabajo: $(pwd)"
      echo "Archivos disponibles:"
      ls -la

      # Verificar archivos necesarios
      if [[ ! -f "Dockerfile" ]]; then
        echo "Error: Dockerfile no encontrado en $(pwd)"
        exit 1
      fi

      if [[ ! -f "cloudbuild.yml" ]]; then
        echo "Error: cloudbuild.yml no encontrado en $(pwd)"
        exit 1
      fi

      if [[ ! -f "entrypoint.sh" ]]; then
        echo "Error: entrypoint.sh no encontrado en $(pwd)"
        exit 1
      fi

      echo "Todos los archivos necesarios están presentes"

      # Ejecutar Cloud Build
      echo "🚀 Ejecutando Cloud Build..."
      gcloud builds submit . \
        --config=cloudbuild.yml \
        --substitutions=_IMAGE_TAG=latest,_REGION=${var.region},_REPO_NAME=mlops-repo \
        --project=${var.project_id} \
        --timeout=600s

      BUILD_EXIT_CODE=$?

      if [[ $BUILD_EXIT_CODE -eq 0 ]]; then
        echo "Build completado exitosamente"

        # Verificar que la imagen esté disponible
        echo "Verificando imagen en Artifact Registry..."
        gcloud container images describe ${var.region}-docker.pkg.dev/${var.project_id}/mlops-repo/mlflow-server:latest \
          --project=${var.project_id} \
          --format="value(digest)" > /dev/null

        if [[ $? -eq 0 ]]; then
          echo "Imagen verificada y lista para deployment"
        else
          echo "Advertencia: No se pudo verificar la imagen inmediatamente"
        fi
      else
        echo "Error en el build de la imagen (Exit code: $BUILD_EXIT_CODE)"
        exit $BUILD_EXIT_CODE
      fi
    EOF
  }

  depends_on = [
    google_artifact_registry_repository.mlops_repo,
    google_project_service.required_apis
  ]
}


# PASO 2: Pequeña espera para asegurar que la imagen esté completamente disponible
resource "time_sleep" "wait_for_image" {
  depends_on = [null_resource.mlflow_image_build]
  
  create_duration = "30s"
}

# PASO 3: SOLO Deploy del servicio Cloud Run (usando imagen ya construida)
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
    google_artifact_registry_repository.mlops_repo,
    time_sleep.wait_for_image  # Espera a que la imagen esté completamente lista
  ]
}

# IAM para acceso público (usando solo IAM binding para evitar duplicación)
resource "google_cloud_run_service_iam_binding" "mlflow_public_access" {
  count = var.allow_public_access ? 1 : 0
  
  service  = google_cloud_run_service.mlflow_server.name
  location = google_cloud_run_service.mlflow_server.location
  role     = "roles/run.invoker"
  
  members = [
    "allUsers"
  ]
  
  depends_on = [
    google_project_service.required_apis,
    google_cloud_run_service.mlflow_server
  ]
}