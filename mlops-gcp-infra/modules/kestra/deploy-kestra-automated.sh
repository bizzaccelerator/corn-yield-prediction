#!/bin/bash

set -e

echo "🚀 DESPLIEGUE AUTOMATIZADO DE KESTRA EN GCP"
echo "=============================================="

# CAMBIO: Función para logging con timestamps
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# CAMBIO: Función para verificar comandos requeridos
check_prerequisites() {
    log "Verificando prerequisitos..."

    if ! command -v terraform &> /dev/null; then
        echo "❌ ERROR: Terraform no está instalado"
        exit 1
    fi

    if ! command -v gcloud &> /dev/null; then
        echo "❌ ERROR: Google Cloud CLI no está instalado"
        exit 1
    fi

    log "✅ Prerequisitos verificados"
}

# CAMBIO: Función para verificar configuración de Terraform
validate_terraform_config() {
    log "Validando configuración de Terraform..."

    if [ ! -f "main.tf" ]; then
        echo "❌ ERROR: main.tf no encontrado en el directorio actual"
        exit 1
    fi

    if [ ! -f "variables.tfvars" ]; then
        echo "❌ ERROR: variables.tfvars no encontrado"
        exit 1
    fi

    log "✅ Archivos de configuración encontrados"
}

# CAMBIO: Función para limpiar estado previo si es necesario
cleanup_if_needed() {
    if [ -f "terraform.tfstate" ] || [ -d ".terraform" ]; then
        log "⚠️  Estado previo de Terraform detectado"
        read -p "¿Desea limpiar el estado previo? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log "🧹 Limpiando estado previo..."
            terraform destroy -auto-approve -var-file="variables.tfvars" || true
            rm -rf .terraform terraform.tfstate* .terraform.lock.hcl || true
        fi
    fi
}

# CAMBIO: Función principal de despliegue
deploy_kestra() {
    log "1️⃣  Inicializando Terraform..."
    terraform init

    log "2️⃣  Validando configuración..."
    terraform validate

    log "3️⃣  Planificando despliegue..."
    terraform plan -var-file="variables.tfvars" -out=tfplan

    log "4️⃣  Aplicando módulo GCS..."
    terraform apply -target=module.gcs -var-file="variables.tfvars" -auto-approve

    log "5️⃣  Verificando creación del bucket de Kestra..."
    BUCKET_NAME=$(terraform output -raw kestra_bucket_name 2>/dev/null || echo "")
    if [ -z "$BUCKET_NAME" ]; then
        echo "❌ ERROR: No se pudo obtener el nombre del bucket de Kestra"
        echo "Outputs disponibles:"
        terraform output
        exit 1
    fi
    log "✅ Bucket de Kestra creado: $BUCKET_NAME"

    log "6️⃣  Aplicando módulo Kestra (Cloud SQL + VM)..."
    terraform apply -target=module.kestra -var-file="variables.tfvars" -auto-approve

    log "7️⃣  Aplicando configuración completa..."
    terraform apply -var-file="variables.tfvars" -auto-approve

    log "8️⃣  Obteniendo información de despliegue..."
    KESTRA_IP=$(terraform output -raw kestra_public_ip)
    KESTRA_URL=$(terraform output -raw kestra_url)

    log "✅ DESPLIEGUE COMPLETADO"
    echo ""
    echo "📋 INFORMACIÓN DEL DESPLIEGUE:"
    echo "   • IP Pública: $KESTRA_IP"
    echo "   • URL de Acceso: $KESTRA_URL"
    echo "   • Bucket GCS: $BUCKET_NAME"
    echo ""
}

# CAMBIO: Función para verificar que Kestra esté funcionando
verify_kestra_deployment() {
    log "🔍 Verificando despliegue de Kestra..."

    KESTRA_IP=$(terraform output -raw kestra_public_ip)
    KESTRA_URL="http://$KESTRA_IP:8080"

    log "Esperando que Kestra esté listo (esto puede tomar varios minutos)..."

    max_attempts=30
    attempt=1

    while [ $attempt -le $max_attempts ]; do
        log "Intento $attempt/$max_attempts: Verificando $KESTRA_URL/health"

        if curl -f -s --connect-timeout 10 --max-time 15 "$KESTRA_URL/health" >/dev/null 2>&1; then
            log "✅ SUCCESS: Kestra está funcionando correctamente!"
            echo ""
            echo "🎉 KESTRA DESPLEGADO Y FUNCIONANDO"
            echo "   Accede a: $KESTRA_URL"
            echo ""
            return 0
        else
            log "⏳ Kestra aún no está listo, esperando 30 segundos..."
            sleep 30
        fi
        ((attempt++))
    done

    log "⚠️  WARNING: No se pudo verificar que Kestra esté funcionando"
    log "Esto puede ser normal si Kestra aún se está iniciando"
    echo ""
    echo "🔧 INFORMACIÓN PARA DEBUGGING:"
    echo "   • Conéctate a la VM: gcloud compute ssh $(terraform output -raw vm_instance_name) --zone=\$(terraform output -raw zone || echo 'us-central1-a')"
    echo "   • Revisa logs: sudo tail -f /var/log/kestra-startup.log"
    echo "   • Revisa contenedor: cd /opt/kestra && sudo docker compose logs -f"
    echo ""
}

# CAMBIO: Función principal
main() {
    echo "Iniciando despliegue automatizado..."

    check_prerequisites
    validate_terraform_config
    cleanup_if_needed

    log "🏁 Comenzando despliegue..."
    deploy_kestra

    log "🔍 Verificando funcionamiento..."
    verify_kestra_deployment

    log "🎯 Proceso completado"
}

# CAMBIO: Manejo de errores
trap 'echo "❌ ERROR: El script falló en la línea $LINENO. Revisa los logs de Terraform."; exit 1' ERR

# Ejecutar función principal
main "$@"
