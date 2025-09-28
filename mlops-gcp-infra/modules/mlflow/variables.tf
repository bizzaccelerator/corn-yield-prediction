variable "project_id" {}
variable "region" {}
variable "bucket_name" {}
variable "db_instance_name" {}
variable "db_user" {}
variable "db_password" {}
variable "db_name" {}

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
