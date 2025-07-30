variable "credentials" {}
variable "project_id" {}
variable "region" {}
variable "bucket_name" {}

variable "db_instance_name" {}
variable "db_user" {}
variable "db_password" {}
variable "db_name" {}

variable "evidently_image" {
  description = "Container image for Evidently Cloud Run"
}

variable "github_owner" {
  description = "GitHub organization or username"
  type        = string
}

variable "github_repo" {
  description = "GitHub repository name"
  type        = string
}

variable "mlflow_service_name" {
  description = "The name of the MLflow Cloud Run service"
  type        = string
}

variable "allow_public_access" {
  description = "Allow public access to MLflow server"
  type        = bool
  default     = true
}

# Kestra-specific variables
variable "kestra_db_password" {
  description = "Password for Kestra database user"
  type        = string
  sensitive   = true
  default     = "kestra_secure_password_2024"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}