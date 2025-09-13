variable "project_id" {
  type        = string
  description = "GCP Project ID"
}

variable "region" {
  type        = string
  description = "Region for Cloud Run and Artifact Registry"
  default     = "us-central1"
}

variable "service_name" {
  type        = string
  description = "Cloud Run service name"
  default     = "evidently-ui"
}

variable "repo_name" {
  type        = string
  description = "Artifact Registry repository name"
  default     = "apps"
}

variable "bucket_name_override" {
  type        = string
  description = "Optional bucket name override"
  default     = ""
}