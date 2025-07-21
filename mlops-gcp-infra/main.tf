terraform {
  required_version = ">= 1.0"
  backend "gcs" {}  # Can change from "local" to "gcs" (for google) or "s3" (for aws), if you would like to preserve your tf-state online
  required_providers {
    google = {
      source  = "hashicorp/google"
    }
  }
}

provider "google" {
  credentials = file(var.credentials)
  project     = var.project_id
  region      = var.region
}

module "gcs" {
  source      = "./modules/gcs"
  project_id  = var.project_id
  region      = var.region
  bucket_name = var.bucket_name
}

module "mlflow" {
  source           = "./modules/mlflow"
  project_id       = var.project_id
  region           = var.region
  bucket_name      = var.bucket_name
  db_instance_name = var.db_instance_name
  db_user          = var.db_user
  db_password      = var.db_password
  db_name          = var.db_name
  github_owner     = var.github_owner
  github_repo      = var.github_repo
}

# module "monitoring" {
#   source = "./modules/monitoring"
#   region = var.region
#   image  = var.evidently_image
# }
