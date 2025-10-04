#!/bin/bash
#
# Multi-Model Worker Entrypoint Script
#
# Provides comprehensive initialization, health checks, and lifecycle management
# for the RunPod serverless deployment environment.
#
# Environment Variables:
#   RUNPOD_AI_API_KEY - RunPod API key for authentication
#   RUNPOD_WEBHOOK_SECRET - Secret for webhook validation
#   MODELS_DIR - Directory containing AI models (default: /runpod-volume/models)
#   VALIDATION_MODE - Model validation level: basic|strict (default: basic)
#   HEALTH_CHECK_INTERVAL - Health check interval in seconds (default: 30)
#   LOG_LEVEL - Logging level: DEBUG|INFO|WARNING|ERROR (default: INFO)
#   STARTUP_TIMEOUT - Maximum startup time in seconds (default: 300)

set -euo pipefail

# =============================================================================
# Configuration and Defaults
# =============================================================================

export MODELS_DIR="${MODELS_DIR:-/runpod-volume/models}"
export VALIDATION_MODE="${VALIDATION_MODE:-basic}"
export HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-30}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
export PYTHONPATH="/runpod-volume:${PYTHONPATH:-}"

# Logging configuration
readonly LOG_FILE="/tmp/worker-startup.log"
readonly PID_FILE="/tmp/worker.pid"
readonly HEALTH_FILE="/tmp/health.status"

# =============================================================================
# Logging Functions
# =============================================================================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"

    # Also log to stdout for RunPod
    echo "[$timestamp] [$level] $message" >&2
}

log_info() {
    log "INFO" "$@"
}

log_warn() {
    log "WARN" "$@"
}

log_error() {
    log "ERROR" "$@"
}

log_debug() {
    if [[ "$LOG_LEVEL" == "DEBUG" ]]; then
        log "DEBUG" "$@"
    fi
}

# =============================================================================
# System Health Checks
# =============================================================================

check_system_resources() {
    log_info "Checking system resources..."

    # Check available disk space
    local available_space=$(df /runpod-volume 2>/dev/null | awk 'NR==2 {print $4}' || echo "0")
    local required_space=$((50 * 1024 * 1024))  # 50GB in KB

    if [[ "$available_space" -lt "$required_space" ]]; then
        log_error "Insufficient disk space. Available: ${available_space}KB, Required: ${required_space}KB"
        return 1
    fi

    log_info "Disk space check passed: $(($available_space / 1024 / 1024))GB available"

    # Check memory
    local available_memory=$(free -m | awk 'NR==2{print $7}' || echo "0")
    local required_memory=4000  # 4GB minimum

    if [[ "$available_memory" -lt "$required_memory" ]]; then
        log_warn "Low available memory: ${available_memory}MB (recommended: ${required_memory}MB+)"
    else
        log_info "Memory check passed: ${available_memory}MB available"
    fi

    # Check GPU availability
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi >/dev/null 2>&1; then
            local gpu_info=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || echo "Unknown")
            log_info "GPU detected: $gpu_info"
            export CUDA_AVAILABLE=true
        else
            log_warn "nvidia-smi available but GPU not accessible"
            export CUDA_AVAILABLE=false
        fi
    else
        log_info "No GPU detected (CPU-only mode)"
        export CUDA_AVAILABLE=false
    fi

    return 0
}

check_python_environment() {
    log_info "Checking Python environment..."

    # Verify Python version
    local python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' || echo "unknown")
    log_info "Python version: $python_version"

    # Check critical Python packages
    local required_packages=("torch" "transformers" "diffusers" "safetensors" "huggingface_hub")

    for package in "${required_packages[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            local version=$(python3 -c "import $package; print(getattr($package, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
            log_info "Package $package: $version"
        else
            log_error "Required package not found: $package"
            return 1
        fi
    done

    # Check PyTorch CUDA availability
    if [[ "$CUDA_AVAILABLE" == "true" ]]; then
        if python3 -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available())" 2>/dev/null; then
            log_info "PyTorch CUDA support verified"
        else
            log_warn "PyTorch CUDA support not available"
        fi
    fi

    return 0
}

# =============================================================================
# Model Validation
# =============================================================================

validate_models() {
    log_info "Starting model validation (mode: $VALIDATION_MODE)..."

    # Ensure models directory exists
    if [[ ! -d "$MODELS_DIR" ]]; then
        log_error "Models directory not found: $MODELS_DIR"
        return 1
    fi

    # Check if any models are present
    local model_count=$(find "$MODELS_DIR" -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" | wc -l)
    if [[ "$model_count" -eq 0 ]]; then
        log_warn "No model files found in $MODELS_DIR"
        log_info "Attempting to download models..."

        # Try to download models if script is available
        if [[ -f "/runpod-volume/scripts/download_models.py" ]]; then
            log_info "Running model download script..."
            if python3 /runpod-volume/scripts/download_models.py --cache-dir="$MODELS_DIR" --validation-mode="$VALIDATION_MODE"; then
                log_info "Model download completed successfully"
            else
                log_error "Model download failed"
                return 1
            fi
        else
            log_error "Model download script not found and no models available"
            return 1
        fi
    else
        log_info "Found $model_count model files"
    fi

    # Run validation script if available
    if [[ -f "/runpod-volume/scripts/validate_models.py" ]]; then
        log_info "Running model validation script..."
        if python3 /runpod-volume/scripts/validate_models.py --models-dir="$MODELS_DIR" --mode="$VALIDATION_MODE"; then
            log_info "Model validation passed"
        else
            log_error "Model validation failed"
            return 1
        fi
    else
        log_warn "Model validation script not found, skipping detailed validation"
    fi

    return 0
}

# =============================================================================
# RunPod Integration
# =============================================================================

check_runpod_environment() {
    log_info "Checking RunPod environment..."

    # Check for RunPod environment variables
    if [[ -z "${RUNPOD_AI_API_KEY:-}" ]]; then
        log_warn "RUNPOD_AI_API_KEY not set"
    else
        log_info "RunPod API key configured"
    fi

    if [[ -z "${RUNPOD_WEBHOOK_SECRET:-}" ]]; then
        log_warn "RUNPOD_WEBHOOK_SECRET not set"
    else
        log_info "RunPod webhook secret configured"
    fi

    # Check RunPod SDK
    if python3 -c "import runpod" 2>/dev/null; then
        local runpod_version=$(python3 -c "import runpod; print(runpod.__version__)" 2>/dev/null || echo "unknown")
        log_info "RunPod SDK version: $runpod_version"
    else
        log_error "RunPod SDK not available"
        return 1
    fi

    return 0
}

# =============================================================================
# Health Monitoring
# =============================================================================

create_health_status() {
    local status="$1"
    local message="$2"
    local timestamp=$(date -Iseconds)

    cat > "$HEALTH_FILE" <<EOF
{
    "status": "$status",
    "message": "$message",
    "timestamp": "$timestamp",
    "uptime_seconds": $(cat /proc/uptime | cut -d' ' -f1 | cut -d'.' -f1),
    "models_dir": "$MODELS_DIR",
    "cuda_available": ${CUDA_AVAILABLE:-false}
}
EOF
}

background_health_monitor() {
    while true; do
        sleep "$HEALTH_CHECK_INTERVAL"

        # Basic health checks
        if [[ -f "$PID_FILE" ]] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
            create_health_status "healthy" "Worker process running normally"
        else
            create_health_status "unhealthy" "Worker process not responding"
            log_error "Worker process health check failed"
        fi
    done &

    local health_monitor_pid=$!
    echo "$health_monitor_pid" > /tmp/health_monitor.pid
    log_info "Health monitor started (PID: $health_monitor_pid)"
}

# =============================================================================
# Signal Handling
# =============================================================================

cleanup() {
    log_info "Received shutdown signal, cleaning up..."

    # Kill health monitor
    if [[ -f /tmp/health_monitor.pid ]]; then
        local health_pid=$(cat /tmp/health_monitor.pid)
        if kill -0 "$health_pid" 2>/dev/null; then
            kill "$health_pid" 2>/dev/null || true
        fi
        rm -f /tmp/health_monitor.pid
    fi

    # Kill worker process
    if [[ -f "$PID_FILE" ]]; then
        local worker_pid=$(cat "$PID_FILE")
        if kill -0 "$worker_pid" 2>/dev/null; then
            log_info "Sending TERM signal to worker process (PID: $worker_pid)"
            kill -TERM "$worker_pid" 2>/dev/null || true

            # Wait for graceful shutdown
            local timeout=10
            while kill -0 "$worker_pid" 2>/dev/null && [[ $timeout -gt 0 ]]; do
                sleep 1
                ((timeout--))
            done

            # Force kill if still running
            if kill -0 "$worker_pid" 2>/dev/null; then
                log_warn "Force killing worker process"
                kill -KILL "$worker_pid" 2>/dev/null || true
            fi
        fi
        rm -f "$PID_FILE"
    fi

    create_health_status "stopped" "Worker shutdown completed"
    log_info "Cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT SIGQUIT

# =============================================================================
# Main Initialization
# =============================================================================

main() {
    log_info "=== Multi-Model Worker Startup ==="
    log_info "Version: $(date '+%Y%m%d')"
    log_info "Models directory: $MODELS_DIR"
    log_info "Validation mode: $VALIDATION_MODE"
    log_info "Log level: $LOG_LEVEL"

    # Create initial health status
    create_health_status "starting" "Worker initialization in progress"

    # Run startup checks with timeout
    local startup_start=$(date +%s)

    if ! timeout "$STARTUP_TIMEOUT" bash -c '
        check_system_resources &&
        check_python_environment &&
        check_runpod_environment &&
        validate_models
    '; then
        log_error "Startup checks failed or timed out after ${STARTUP_TIMEOUT}s"
        create_health_status "failed" "Startup validation failed"
        exit 1
    fi

    local startup_duration=$(($(date +%s) - startup_start))
    log_info "Startup checks completed in ${startup_duration}s"

    # Start background health monitor
    background_health_monitor

    # Update health status
    create_health_status "ready" "Worker initialized and ready to serve requests"

    # Start the main worker process
    log_info "Starting RunPod worker..."

    # Export environment for the worker
    export PYTHONUNBUFFERED=1
    export PYTHONPATH="/runpod-volume:${PYTHONPATH:-}"

    # Start worker and save PID
    if [[ -f "/runpod-volume/handler.py" ]]; then
        python3 -u /runpod-volume/handler.py &
        local worker_pid=$!
        echo "$worker_pid" > "$PID_FILE"
        log_info "Worker process started (PID: $worker_pid)"

        # Wait for worker process
        wait "$worker_pid"
        local exit_code=$?

        log_info "Worker process exited with code: $exit_code"
        create_health_status "stopped" "Worker process terminated"

        # Clean up
        rm -f "$PID_FILE"

        exit $exit_code
    else
        log_error "Worker handler not found: /runpod-volume/handler.py"
        create_health_status "failed" "Worker handler not found"
        exit 1
    fi
}

# =============================================================================
# Health Check Endpoint (for external monitoring)
# =============================================================================

if [[ "${1:-}" == "health" ]]; then
    if [[ -f "$HEALTH_FILE" ]]; then
        cat "$HEALTH_FILE"
        exit 0
    else
        echo '{"status": "unknown", "message": "Health status not available"}'
        exit 1
    fi
fi

# =============================================================================
# Script Entry Point
# =============================================================================

# Ensure we're running as the expected user
if [[ "$(id -u)" -eq 0 ]] && [[ "${ALLOW_ROOT:-}" != "true" ]]; then
    log_error "Running as root is not recommended. Set ALLOW_ROOT=true to override."
    exit 1
fi

# Create required directories
mkdir -p "$(dirname "$LOG_FILE")" "$MODELS_DIR" /tmp

# Initialize log file
echo "=== Worker Startup Log - $(date -Iseconds) ===" > "$LOG_FILE"

# Run main function
main "$@"