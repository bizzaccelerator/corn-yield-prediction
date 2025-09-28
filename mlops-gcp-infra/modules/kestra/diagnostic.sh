#!/bin/bash

echo "=== DIAGNÓSTICO COMPLETO ==="

echo "1. Estado actual de Terraform:"
terraform state list

echo -e "\n2. Verificando módulo GCS en el estado:"
terraform state list | grep "module.gcs"

echo -e "\n3. Verificando si existe el recurso random_id:"
terraform state show module.gcs.random_id.bucket_suffix 2>/dev/null || echo "❌ random_id no encontrado"

echo -e "\n4. Verificando si existe el bucket de Kestra:"
terraform state show module.gcs.google_storage_bucket.kestra_bucket 2>/dev/null || echo "❌ kestra_bucket no encontrado"

echo -e "\n5. Outputs del módulo GCS:"
terraform output -json | jq '.["module.gcs"]' 2>/dev/null || echo "❌ No hay outputs del módulo GCS"

echo -e "\n6. Verificando conectividad en la VM (si está accesible):"
EXTERNAL_IP=$(terraform output -raw kestra_public_ip 2>/dev/null)
if [ ! -z "$EXTERNAL_IP" ]; then
    echo "IP externa: $EXTERNAL_IP"
    echo "Probando conexión al puerto 8080..."
    nc -zv $EXTERNAL_IP 8080 2>/dev/null && echo "✅ Puerto 8080 accesible" || echo "❌ Puerto 8080 no accesible"
else
    echo "❌ No se pudo obtener la IP externa"
fi
