# Multi-Modal AI Inference Worker Makefile
# Standalone repository build and development automation

.PHONY: help setup clean test build push deploy dev lint format coverage validate

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
RESET := \033[0m

# Configuration
PYTHON := python3.11
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest

# Docker configuration
IMAGE_NAME := ghcr.io/mikeblakeway/multi-modal-worker
VERSION ?= $(shell git rev-parse --short HEAD)
DOCKERFILE := docker/Dockerfile

help: ## Display this help message
	@echo "$(BLUE)Multi-Modal AI Inference Worker$(RESET)"
	@echo "$(YELLOW)Available commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Set up development environment
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	@$(PYTHON) -m venv $(VENV)
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements-dev.txt
	@echo "$(GREEN)✅ Development environment ready$(RESET)"
	@echo "$(YELLOW)Activate with: source $(VENV)/bin/activate$(RESET)"

clean: ## Clean up temporary files and caches
	@echo "$(YELLOW)Cleaning up...$(RESET)"
	@rm -rf $(VENV)
	@rm -rf __pycache__/ */__pycache__/ */*/__pycache__/
	@rm -rf .pytest_cache/ htmlcov/ .coverage coverage.xml
	@rm -rf build/ dist/ *.egg-info/
	@rm -rf logs/ tmp/ temp/ test_output/ debug_output/
	@echo "$(GREEN)✅ Cleanup complete$(RESET)"

test: ## Run all tests
	@echo "$(BLUE)Running test suite...$(RESET)"
	@$(PYTHON_VENV) run_tests.py
	@echo "$(GREEN)✅ Tests completed$(RESET)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(RESET)"
	@$(PYTEST) tests/unit/ -v
	@echo "$(GREEN)✅ Unit tests completed$(RESET)"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	@$(PYTEST) tests/integration/ -v
	@echo "$(GREEN)✅ Integration tests completed$(RESET)"

coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	@$(PYTHON_VENV) run_tests.py --coverage
	@echo "$(GREEN)✅ Coverage report generated in htmlcov/$(RESET)"

validate: ## Validate framework and models
	@echo "$(BLUE)Validating framework...$(RESET)"
	@$(PYTHON_VENV) validate_framework.py
	@echo "$(GREEN)✅ Framework validation complete$(RESET)"

lint: ## Run code linting
	@echo "$(BLUE)Running linting...$(RESET)"
	@$(VENV)/bin/flake8 src/ tests/ --max-line-length=100
	@$(VENV)/bin/mypy src/ --ignore-missing-imports
	@echo "$(GREEN)✅ Linting complete$(RESET)"

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(RESET)"
	@$(VENV)/bin/black src/ tests/ scripts/ --line-length=100
	@echo "$(GREEN)✅ Code formatted$(RESET)"

build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(RESET)"
	@echo "$(YELLOW)Image: $(IMAGE_NAME):$(VERSION)$(RESET)"
	@docker build \
		--platform linux/amd64 \
		--build-arg VERSION=$(VERSION) \
		--build-arg BUILD_DATE="$(shell date -u +'%Y-%m-%dT%H:%M:%SZ')" \
		--tag $(IMAGE_NAME):$(VERSION) \
		--tag $(IMAGE_NAME):latest \
		--file $(DOCKERFILE) \
		.
	@echo "$(GREEN)✅ Docker image built: $(IMAGE_NAME):$(VERSION)$(RESET)"

push: build ## Push Docker image to registry
	@echo "$(BLUE)Pushing Docker image...$(RESET)"
	@docker push $(IMAGE_NAME):$(VERSION)
	@docker push $(IMAGE_NAME):latest
	@echo "$(GREEN)✅ Docker image pushed$(RESET)"

dev: ## Start development environment
	@echo "$(BLUE)Starting development environment...$(RESET)"
	@$(PYTHON_VENV) -m src.main
	@echo "$(GREEN)✅ Development server stopped$(RESET)"

dev-docker: ## Run development Docker container
	@echo "$(BLUE)Starting development Docker container...$(RESET)"
	@docker run -it --rm \
		--gpus all \
		-p 8000:8000 \
		-v $(PWD)/models:/workspace/models \
		-e LOG_LEVEL=DEBUG \
		$(IMAGE_NAME):latest
	@echo "$(GREEN)✅ Development container stopped$(RESET)"

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(RESET)"
	@$(PYTEST) tests/performance/ -v --benchmark-only
	@echo "$(GREEN)✅ Benchmarks complete$(RESET)"

validate-models: ## Validate downloaded models
	@echo "$(BLUE)Validating models...$(RESET)"
	@$(PYTHON_VENV) scripts/validate_models.py --models-dir=./models --mode=basic
	@echo "$(GREEN)✅ Model validation complete$(RESET)"

security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(RESET)"
	@$(VENV)/bin/bandit -r src/ -ll
	@$(VENV)/bin/safety check
	@echo "$(GREEN)✅ Security checks complete$(RESET)"

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(RESET)"
	@$(VENV)/bin/sphinx-build -b html docs/ docs/_build/html
	@echo "$(GREEN)✅ Documentation generated in docs/_build/html$(RESET)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	@$(PIP) install -r requirements-dev.txt
	@echo "$(GREEN)✅ Development dependencies installed$(RESET)"

check: lint test ## Run all checks (lint + test)
	@echo "$(GREEN)✅ All checks passed$(RESET)"

deploy-runpod: ## Deploy to RunPod (requires runpodctl)
	@echo "$(BLUE)Deploying to RunPod...$(RESET)"
	@runpodctl create template \
		--name "Multi-Modal AI Worker" \
		--image $(IMAGE_NAME):latest \
		--gpu-type "RTX 4090" \
		--container-disk-size 50 \
		--env MODELS_DIR=/runpod-volume/models \
		--env VALIDATION_MODE=basic \
		--env LOG_LEVEL=INFO
	@echo "$(GREEN)✅ Deployed to RunPod$(RESET)"

version: ## Display version information
	@echo "$(BLUE)Multi-Modal AI Inference Worker$(RESET)"
	@echo "$(YELLOW)Version: $(VERSION)$(RESET)"
	@echo "$(YELLOW)Image: $(IMAGE_NAME):$(VERSION)$(RESET)"
	@echo "$(YELLOW)Python: $(shell $(PYTHON) --version)$(RESET)"
	@echo "$(YELLOW)Git: $(shell git rev-parse HEAD)$(RESET)"

.DEFAULT_GOAL := help