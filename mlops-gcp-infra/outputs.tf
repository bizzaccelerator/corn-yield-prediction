output "artifact_bucket_url" {
  value = module.gcs.bucket_url
}

# output "mlflow_url" {
#   value = module.mlflow.mlflow_url
# }

# output "evidently_url" {
#   value = module.monitoring.evidently_url
# }

# KESTRA
output "kestra_public_ip" {
  description = "Public IP address of Kestra server"
  value       = module.kestra.kestra_public_ip
}

output "kestra_url" {
  description = "URL to access Kestra UI"
  value       = module.kestra.kestra_url
}

output "kestra_database_instance" {
  description = "Kestra Cloud SQL instance name"
  value       = module.kestra.database_instance_name
}

output "kestra_database_connection_name" {
  description = "Kestra Cloud SQL connection name"
  value       = module.kestra.database_connection_name
}