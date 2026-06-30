# Eloquium TTS — convenience wrapper.
#
# From scratch on a new machine:
#     git clone <repo> && cd eloquium
#     make gpu-setup     # only if the machine has an NVIDIA GPU (installs toolkit, needs sudo)
#     make run           # build + start + wait until healthy (CPU or GPU, auto-detected)
#     make gen TEXT="Привет, мир" LNG=ru     # synthesize -> prints path to a .wav
#
# Everything auto-detects the GPU: the right torch wheels (CPU vs CUDA) AND the
# docker-compose GPU override are selected based on whether nvidia-smi works.

# ---- torch index (build arg) -------------------------------------------------
CUDA_INDEX ?= https://download.pytorch.org/whl/cu124
CPU_INDEX  := https://download.pytorch.org/whl/cpu

# ---- GPU auto-detection ------------------------------------------------------
# USE_GPU is non-empty when an NVIDIA GPU is usable on this host.
USE_GPU := $(shell if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then echo 1; fi)

ifeq ($(USE_GPU),1)
  TORCH_INDEX   := $(CUDA_INDEX)
  COMPOSE_FILES := -f docker-compose.yml -f docker-compose.gpu.yml
else
  TORCH_INDEX   := $(CPU_INDEX)
  COMPOSE_FILES := -f docker-compose.yml
endif

DC := docker compose $(COMPOSE_FILES)

# ---- output / generation params ----------------------------------------------
OUT_DIR ?= out
LNG     ?= en
URL     ?= http://localhost:8000

.PHONY: run up build down logs ps health restart gen gpu-setup gpu-check help

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

run: build ## From scratch: build, start, and wait until the service is healthy
	@echo ">> GPU: $(if $(USE_GPU),detected (CUDA build),not detected (CPU build))"
	@if [ -n "$(USE_GPU)" ] && ! docker info 2>/dev/null | grep -qi 'nvidia'; then \
		echo ">> WARNING: GPU found but Docker can't see it. If startup fails, run: make gpu-setup"; \
	fi
	$(DC) up -d
	@echo ">> waiting for /health (first run downloads model weights — can take several minutes)..."
	@for i in $$(seq 1 120); do \
		if curl -fsS $(URL)/health >/dev/null 2>&1; then \
			echo ">> ready: $$(curl -s $(URL)/health)"; \
			echo ">> generate audio:  make gen TEXT=\"Привет, мир\" LNG=ru"; \
			exit 0; \
		fi; \
		sleep 5; \
	done; \
	echo ">> service did not become healthy in time — check: make logs"; exit 1

build: ## Build the image with the auto-detected torch index
	@echo ">> torch index: $(TORCH_INDEX)"
	$(DC) build --build-arg TORCH_INDEX=$(TORCH_INDEX)

up: build ## Build (if needed) and start the service detached
	$(DC) up -d
	@echo ">> started. logs: make logs   health: make health"

down: ## Stop and remove the container
	$(DC) down

restart: ## Restart the running service
	$(DC) restart

logs: ## Follow the service logs
	$(DC) logs -f tts

ps: ## Show container status
	$(DC) ps

health: ## Curl the health endpoint
	@curl -fsS $(URL)/health && echo

gen: ## Synthesize speech: make gen TEXT="..." [LNG=ru] -> prints path to a .wav
	@test -n "$(TEXT)" || { echo "usage: make gen TEXT=\"your text\" [LNG=ru]"; exit 1; }
	@mkdir -p $(OUT_DIR)
	@f="$(OUT_DIR)/tts_$$(date +%Y%m%d_%H%M%S).wav"; \
	payload=$$(printf '%s' '$(TEXT)' | LANG=C.UTF-8 python3 -c 'import json,sys; print(json.dumps({"text": sys.stdin.read(), "language": "$(LNG)"}))'); \
	code=$$(curl -s -o "$$f" -w '%{http_code}' -X POST $(URL)/tts \
		-H 'Content-Type: application/json' -d "$$payload"); \
	if [ "$$code" = "200" ]; then \
		echo "$$(cd "$$(dirname "$$f")" && pwd)/$$(basename "$$f")"; \
	else \
		echo ">> FAILED (HTTP $$code):"; cat "$$f"; echo; rm -f "$$f"; exit 1; \
	fi

gpu-check: ## Verify Docker can see the NVIDIA GPU
	docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L

gpu-setup: ## Install the NVIDIA Container Toolkit on the host (needs sudo, one-time)
	@echo ">> installing NVIDIA Container Toolkit (you will be prompted for sudo)"
	curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
		| sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
	curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
		| sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
		| sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
	sudo apt-get update
	sudo apt-get install -y nvidia-container-toolkit
	sudo nvidia-ctk runtime configure --runtime=docker
	sudo systemctl restart docker
	@echo ">> done. verify with: make gpu-check"
