provider "google" {
  project = var.project_id
  region  = var.region
}

module "storage" {
  source      = "./modules/gcs"
  bucket_name = var.bucket_name
  region      = var.region
}
