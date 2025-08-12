# Enable required Google Cloud APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "containerregistry.googleapis.com",
    "storage.googleapis.com",
    "iam.googleapis.com"
  ])

  project = var.project_id
  service = each.value
  
  disable_on_destroy = false
}

# Build and push the Docker image
resource "null_resource" "build_evidently_image" {
  triggers = {
    dockerfile_hash = filemd5("${path.module}/evidently-app/Dockerfile")
    main_py_hash    = filemd5("${path.module}/evidently-app/main.py")
    requirements_hash = filemd5("${path.module}/evidently-app/requirements.txt")
    timestamp = timestamp()
  }

  provisioner "local-exec" {
    command = "gcloud builds submit --config=${path.module}/evidently-app/cloudbuild.yml ."
    working_dir = "${path.root}"
  }

  depends_on = [google_project_service.required_apis]
}

# Service account for the Cloud Run service
resource "google_service_account" "evidently_sa" {
  account_id   = "evidently-monitoring-sa"
  display_name = "Evidently Monitoring Service Account"
  
  depends_on = [google_project_service.required_apis]
}

# IAM binding for Cloud Storage
resource "google_project_iam_member" "evidently_storage" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.evidently_sa.email}"
}

# Cloud Run service for Evidently
resource "google_cloud_run_service" "evidently_service" {
  name     = "evidently-monitoring"
  location = var.region
  
  template {
    spec {
      containers {
        image = var.image
        
        ports {
          container_port = 8080
        }
        
        resources {
          limits = {
            memory = "1Gi"
            cpu    = "1000m"
          }
        }
      }
      
      timeout_seconds = 300
      service_account_name = google_service_account.evidently_sa.email
    }

    metadata {
      annotations = {
        "autoscaling.knative.dev/maxScale" = "3"
        "autoscaling.knative.dev/minScale" = "0"
        "run.googleapis.com/execution-environment" = "gen1"
      }
    }
  }

  autogenerate_revision_name = true

  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [
    null_resource.build_evidently_image,
    google_project_service.required_apis
  ]
}

# Allow public access to the service
resource "google_cloud_run_service_iam_member" "all_users" {
  service  = google_cloud_run_service.evidently_service.name
  location = google_cloud_run_service.evidently_service.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}