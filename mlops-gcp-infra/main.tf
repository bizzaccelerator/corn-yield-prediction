terraform {
  required_version = ">= 1.0"
  backend "gcs" {} # Can change from "local" to "gcs" (for google) or "s3" (for aws), if you would like to preserve your tf-state online
  required_providers {
    google = {
      source = "hashicorp/google"
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
  source              = "./modules/mlflow"
  project_id          = var.project_id
  region              = var.region
  bucket_name         = var.bucket_name
  db_instance_name    = var.db_instance_name
  db_user             = var.db_user
  db_password         = var.db_password
  db_name             = var.db_name
  github_owner        = var.github_owner
  github_repo         = var.github_repo
  mlflow_service_name = var.mlflow_service_name
  allow_public_access = var.allow_public_access
}

module "kestra" {
  source = "./modules/kestra"

  project_id      = var.project_id
  project_name    = replace(var.project_id, "-", "_")
  region          = var.region
  zone            = var.zone
  db_password     = var.kestra_db_password
  gcs_bucket_name = module.gcs.kestra_bucket_name
}

module "monitoring" {
  source      = "./modules/monitoring"
  project_id  = var.project_id
  region      = var.region
  service_name = "evidently-ui"
  repo_name   = "apps"
}
