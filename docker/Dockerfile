# Multi-Modal Inference Worker Dockerfile
#
# Multi-stage build for optimized production deployment
# Supports FLUX.1, ControlNet, AnimateDiff, LTX-Video, Inpainting, and Camera Control
# Platform: linux/amd64 (RunPod compatible)

#################################
# Base Stage - System Dependencies
#################################
FROM --platform=linux/amd64 python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies required for ML libraries
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create necessary directories
RUN mkdir -p /runpod-volume/models /runpod-volume/cache /app/logs

# Set environment variables
ENV PYTHONPATH=/app/src
ENV MODEL_CACHE_DIR=/runpod-volume/models
ENV HF_HOME=/runpod-volume/cache/hf
ENV TORCH_HOME=/runpod-volume/cache/torch
ENV RUNPOD_WORKER=true
ENV PYTHONUNBUFFERED=1

# Copy and install Python dependencies
COPY docker/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

#################################
# Models Stage - Download and Cache Models
#################################
FROM base as models

# Copy model download script
COPY scripts/ ./scripts/

# Download and validate models (respecting 40GB limit)
RUN python scripts/download_models.py \
    --target-size=40GB \
    --cache-dir=/runpod-volume/models \
    --validation-mode=strict \
    --progress-bar

# Verify model integrity and create manifest
RUN python scripts/validate_models.py --models-dir=/runpod-volume/models

#################################
# Development Stage - Include Development Tools
#################################
FROM models as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest>=7.4.0 \
    pytest-cov>=4.1.0 \
    black>=23.9.0 \
    flake8>=6.0.0 \
    ipython>=8.0.0 \
    jupyter>=1.0.0

# Copy all source code including tests
COPY . /app/

# Expose additional ports for development
EXPOSE 8000 8888

# Development command with hot reload
CMD ["python", "-u", "src/main.py"]

#################################
# Runtime Stage - Production Optimized
#################################
FROM base as runtime

# Copy pre-downloaded models from models stage
COPY --from=models /runpod-volume/models /runpod-volume/models
COPY --from=models /runpod-volume/cache /runpod-volume/cache

# Copy only necessary application files
COPY src/ ./src/
COPY docker/entrypoint.sh ./entrypoint.sh

# Make entrypoint executable
RUN chmod +x ./entrypoint.sh

# Create non-root user for security
RUN groupadd -r worker && useradd -r -g worker worker
RUN chown -R worker:worker /app /runpod-volume
USER worker

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose application port
EXPOSE 8000

# Use entrypoint script for proper initialization
ENTRYPOINT ["./entrypoint.sh"]
CMD ["python", "-u", "src/main.py"]

#################################
# Production Stage - Final Optimized Image
#################################
FROM runtime as production

# Additional production optimizations
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV CUDA_LAUNCH_BLOCKING=0
ENV OMP_NUM_THREADS=4

# Remove any remaining development artifacts
RUN find /runpod-volume -name "*.pyc" -delete && \
    find /runpod-volume -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Production label
LABEL maintainer="Media Labs Team" \
      version="1.0.0" \
      description="Multi-Modal Inference Worker for RunPod" \
      stage="production"