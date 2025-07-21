resource "google_cloud_run_service" "evidently_service" {
  name     = "evidently-monitoring"
  location = var.region

  template {
    spec {
      containers {
        image = var.image
        resources {
          limits = {
            memory = "512Mi"
            cpu    = "1"
          }
        }
      }
      timeout_seconds = 600  # Extend timeout
    }

    metadata {
      annotations = {
        "autoscaling.knative.dev/maxScale" = "1"
      }
    }
  }

  autogenerate_revision_name = true

  traffic {
    percent         = 100
    latest_revision = true
  }
}

resource "google_cloud_run_service_iam_member" "all_users" {
  service    = google_cloud_run_service.evidently_service.name
  location   = google_cloud_run_service.evidently_service.location
  role       = "roles/run.invoker"
  member     = "allUsers"
}