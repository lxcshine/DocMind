#!/usr/bin/env bash
# =====================================================================
# DocMind Kubernetes Deploy Script
# =====================================================================
# Usage:
#   ./deploy.sh [command] [environment]
#
# Commands:
#   deploy    Deploy to cluster (default)
#   teardown  Remove all resources
#   status    Show resource status
#   logs      Tail backend logs
#   backup    Backup MySQL data
#
# Environments:
#   dev       Use base overlay (default)
#   prod      Use production overlay
#
# Examples:
#   ./deploy.sh deploy dev
#   ./deploy.sh deploy prod
#   ./deploy.sh status
#   ./deploy.sh logs
#   ./deploy.sh backup
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="docmind"

# Select overlay
ENV="${2:-dev}"
if [ "$ENV" = "prod" ]; then
    OVERLAY="${SCRIPT_DIR}/overlays/production"
else
    OVERLAY="${SCRIPT_DIR}/base"
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# Pre-flight checks
preflight() {
    info "Running pre-flight checks..."

    command -v kubectl >/dev/null 2>&1 || error "kubectl not found in PATH"
    kubectl cluster-info >/dev/null 2>&1 || error "Cannot connect to Kubernetes cluster"

    # Check if secrets have been customized
    if grep -q "REPLACE_ME" "${SCRIPT_DIR}/base/secret.yaml" 2>/dev/null; then
        warn "secret.yaml still contains placeholder values!"
        warn "Please edit k8s/base/secret.yaml and replace all REPLACE_ME_ values."
        read -rp "Continue anyway? (y/N) " confirm
        [[ "$confirm" =~ ^[Yy]$ ]] || exit 1
    fi

    info "Pre-flight checks passed"
}

deploy() {
    info "Deploying DocMind to namespace '${NAMESPACE}' (env: ${ENV})..."
    preflight

    kubectl apply -k "$OVERLAY"

    info "Waiting for MySQL to be ready..."
    kubectl rollout status statefulset/mysql -n "$NAMESPACE" --timeout=300s

    info "Waiting for Backend to be ready..."
    kubectl rollout status deployment/backend -n "$NAMESPACE" --timeout=600s

    info "Waiting for Frontend to be ready..."
    kubectl rollout status deployment/frontend -n "$NAMESPACE" --timeout=120s

    info "Deployment complete!"
    echo ""
    info "Services:"
    kubectl get svc -n "$NAMESPACE"
    echo ""
    info "Pods:"
    kubectl get pods -n "$NAMESPACE" -o wide
    echo ""
    info "Access the application via the Ingress endpoint."
    info "Run './deploy.sh status' for detailed status."
}

teardown() {
    warn "This will remove ALL DocMind resources from namespace '${NAMESPACE}'"
    read -rp "Are you sure? (y/N) " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || exit 0

    info "Removing DocMind resources..."
    kubectl delete -k "$OVERLAY" --ignore-not-found=true

    # Optionally delete PVCs (data loss!)
    read -rp "Delete persistent volumes too? This will cause DATA LOSS! (y/N) " confirm_pvc
    if [[ "$confirm_pvc" =~ ^[Yy]$ ]]; then
        kubectl delete pvc -n "$NAMESPACE" --all
        info "PVCs deleted"
    fi

    info "Teardown complete"
}

status() {
    info "DocMind Status (namespace: ${NAMESPACE})"
    echo ""
    echo "=== Pods ==="
    kubectl get pods -n "$NAMESPACE" -o wide
    echo ""
    echo "=== Services ==="
    kubectl get svc -n "$NAMESPACE"
    echo ""
    echo "=== PVCs ==="
    kubectl get pvc -n "$NAMESPACE"
    echo ""
    echo "=== HPA ==="
    kubectl get hpa -n "$NAMESPACE"
    echo ""
    echo "=== Ingress ==="
    kubectl get ingress -n "$NAMESPACE"
    echo ""
    echo "=== Recent Events ==="
    kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10
}

logs() {
    local component="${LOGS_COMPONENT:-backend}"
    info "Tailing logs for ${component}..."
    kubectl logs -f "deployment/${component}" -n "$NAMESPACE" --tail=100
}

backup() {
    local ts=$(date +%Y%m%d_%H%M%S)
    local backup_file="docmind_mysql_backup_${ts}.sql"

    info "Creating MySQL backup..."
    kubectl exec mysql-0 -n "$NAMESPACE" -- \
        mysqldump -u root -p"$(kubectl get secret docmind-secrets -n "$NAMESPACE" -o jsonpath='{.data.MYSQL_ROOT_PASSWORD}' | base64 -d)" \
        --single-transaction --routines --triggers docmind > "$backup_file"

    info "Backup saved to: ${backup_file}"
    info "Size: $(du -h "$backup_file" | cut -f1)"
}

# Main
COMMAND="${1:-deploy}"
case "$COMMAND" in
    deploy)    deploy ;;
    teardown)  teardown ;;
    status)    status ;;
    logs)
        LOGS_COMPONENT="${3:-backend}"
        ENV="${2:-dev}"
        if [ "$ENV" = "prod" ]; then
            OVERLAY="${SCRIPT_DIR}/overlays/production"
        fi
        logs
        ;;
    backup)    backup ;;
    *)
        echo "Usage: $0 {deploy|teardown|status|logs|backup} [dev|prod] [component]"
        echo ""
        echo "Examples:"
        echo "  $0 deploy dev       Deploy to dev environment"
        echo "  $0 deploy prod      Deploy to production"
        echo "  $0 logs dev backend Tail backend logs in dev"
        echo "  $0 status           Show cluster status"
        echo "  $0 backup           Backup MySQL data"
        exit 1
        ;;
esac
