# Eloquium TTS — convenience wrapper.
# Auto-detects an NVIDIA GPU and builds the matching torch (CPU vs CUDA),
# mirroring voicebox's justfile detection but at the Docker layer.

# CUDA index used when a GPU is detected. Override: make up CUDA_INDEX=.../cu121
CUDA_INDEX ?= https://download.pytorch.org/whl/cu124
CPU_INDEX  := https://download.pytorch.org/whl/cpu

# Pick the torch index based on whether nvidia-smi is present and working.
TORCH_INDEX := $(shell if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; \
	then echo "$(CUDA_INDEX)"; else echo "$(CPU_INDEX)"; fi)

.PHONY: up build down logs ps health restart

build: ## Build the image with the auto-detected torch index
	@echo ">> torch index: $(TORCH_INDEX)"
	docker compose build --build-arg TORCH_INDEX=$(TORCH_INDEX)

up: build ## Build (if needed) and start the service detached
	docker compose up -d
	@echo ">> started. logs: make logs   health: make health"

down: ## Stop and remove the container
	docker compose down

restart: ## Restart the running service
	docker compose restart

logs: ## Follow the service logs
	docker compose logs -f tts

ps: ## Show container status
	docker compose ps

health: ## Curl the health endpoint
	@curl -fsS localhost:8000/health && echo
