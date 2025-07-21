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
