# modules/kestra/main.tf

# Data sources
data "google_compute_default_service_account" "default" {}

# Cloud SQL instance for Kestra
resource "google_sql_database_instance" "kestra_db" {
  name             = "${replace(var.project_name, "_", "-")}-kestra-db"
  database_version = "POSTGRES_15"
  region          = var.region

  settings {
    tier = "db-f1-micro"

    backup_configuration {
      enabled                        = true
      start_time                    = "23:00"
      point_in_time_recovery_enabled = true
    }

    ip_configuration {
      ipv4_enabled    = true
      authorized_networks {
        name  = "allow-all"
        value = "0.0.0.0/0"
      }
    }

    database_flags {
      name  = "log_checkpoints"
      value = "on"
    }
  }

  deletion_protection = false
}

resource "google_sql_database" "kestra_database" {
  name     = "kestra"
  instance = google_sql_database_instance.kestra_db.name
}

resource "google_sql_user" "kestra_user" {
  name     = "kestra"
  instance = google_sql_database_instance.kestra_db.name
  password = var.db_password
}

# Service Account for Kestra VM
resource "google_service_account" "kestra_sa" {
  account_id   = "kestra-service-account"
  display_name = "Kestra Service Account"
  description  = "Service account for Kestra VM"
}

# IAM bindings for the service account
resource "google_project_iam_member" "kestra_storage_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.kestra_sa.email}"
}

resource "google_project_iam_member" "kestra_cloudsql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.kestra_sa.email}"
}

resource "google_project_iam_member" "kestra_compute_viewer" {
  project = var.project_id
  role    = "roles/compute.viewer"
  member  = "serviceAccount:${google_service_account.kestra_sa.email}"
}

# Firewall rule to allow HTTP traffic to Kestra
resource "google_compute_firewall" "kestra_firewall" {
  name    = "${replace(var.project_name, "_", "-")}-kestra-firewall"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8080", "22", "443"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["kestra-server"]
}

# Static IP for Kestra VM
resource "google_compute_address" "kestra_static_ip" {
  name   = "${replace(var.project_name, "_", "-")}-kestra-ip"
  region = var.region
}

# Startup script for Kestra VM
locals {
  startup_script = templatefile("${path.module}/startup-script.sh", {
    db_host     = google_sql_database_instance.kestra_db.ip_address.0.ip_address
    db_name     = google_sql_database.kestra_database.name
    db_user     = google_sql_user.kestra_user.name
    db_password = var.db_password
    gcs_bucket  = var.gcs_bucket_name
    project_id  = var.project_id
  })
}

# Kestra VM instance
resource "google_compute_instance" "kestra_vm" {
  name         = "${replace(var.project_name, "_", "-")}-kestra-vm"
  machine_type = var.machine_type
  zone         = var.zone

  tags = ["kestra-server"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 30
      type  = "pd-standard"
    }
  }

  network_interface {
    network = "default"
    access_config {
      nat_ip = google_compute_address.kestra_static_ip.address
    }
  }

  service_account {
    email  = google_service_account.kestra_sa.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    startup-script = local.startup_script
  }

  # Allow recreation when startup script changes
  metadata_startup_script = local.startup_script

  depends_on = [
    google_sql_database_instance.kestra_db,
    google_sql_database.kestra_database,
    google_sql_user.kestra_user
  ]
}