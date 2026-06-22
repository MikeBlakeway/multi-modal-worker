# Multi-Modal AI Inference Worker

A Python serverless worker for RunPod that handles six distinct AI inference modalities from a single endpoint — text-to-image, image-to-video, text-to-video, ControlNet guided generation, inpainting, and camera control. A `modality` field in the request determines routing; everything else is unified: one container, one model management system, one response schema.

Pre-built image: `ghcr.io/mikeblakeway/multi-modal-worker:latest`

---

## Supported modalities

| Modality | Model | GPU memory | Use case |
|---|---|---|---|
| `text-to-image` | FLUX.1 Schnell (fp8) | ~12GB | High-quality image generation |
| `image-to-video` | AnimateDiff | ~10GB | Animate a static image into video |
| `text-to-video` | LTX-Video 2B (distilled) | ~14GB | Generate video directly from a prompt |
| `controlnet` | ControlNet (Canny / Depth) | ~10GB | Structurally guided image generation |
| `inpainting` | SDXL Inpainting | ~12GB | Fill or replace masked image regions |
| `camera-control` | CameraCtrl | ~8GB | Apply camera movement to video content |

---

## Architecture

### Request routing

```
RunPod event
    └─ handler()
           ├─ health_check?  → MultiModalHandler.health_check()
           ├─ system_status? → MultiModalHandler.get_system_status()
           └─ inference      → MultiModalHandler.process_request()
                                      ├─ RequestValidator
                                      ├─ ModalityDetector
                                      ├─ FluxHandler
                                      ├─ AnimateDiffHandler
                                      ├─ LTXVideoHandler
                                      ├─ ControlNetHandler
                                      ├─ InpaintingHandler
                                      ├─ CameraControlHandler
                                      └─ ResponseFormatter
```

Global singletons (`ModelManager`, `MultiModalHandler`) are initialised once on first request and reused across the serverless invocation lifecycle.

### Model management

Models are managed through a three-layer system:

**`ModelManager`** — registers model classes (not instances), loads them on demand when first requested, and evicts them when memory pressure requires it. Models are evicted by usage recency and priority score; high-priority models (e.g. shared VAE, tokenizers) are evicted last.

**`MemoryMonitor`** — tracks real-time system and GPU memory usage via `psutil` and `nvidia-smi`. Before loading a model, `can_load_model(mb)` is called to check available headroom. Stats include system and GPU used/total/available.

**`BaseModel`** — abstract base class for all model implementations. Each handler subclass implements `load()`, `unload()`, `infer(inputs)`, `validate_inputs()`, and `get_memory_usage()`. Base class tracks `is_loaded`, `load_time`, `last_used`, `memory_usage_mb`, and `priority`.

```python
# Example: request a model — loaded on demand, evicted by manager when needed
model = manager.get_model("flux-schnell")
result = model.infer({"prompt": "...", "steps": 4})
```

---

## API

### Request format (RunPod)

All requests follow the standard RunPod serverless format: `{ "input": { ... } }`.

```json
{
  "input": {
    "modality": "text-to-image",
    "prompt": "A mountain lake at dawn",
    "steps": 4,
    "guidance_scale": 1.0,
    "width": 1024,
    "height": 1024,
    "seed": 42
  }
}
```

**Common fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `modality` | string | required | One of the six supported modalities |
| `prompt` | string | — | Required for text-based modalities |
| `image_url` | string | — | Required for image-based modalities |
| `steps` | integer | 4 | Inference steps (1–50) |
| `guidance_scale` | float | 1.0 | CFG scale (0.0–20.0) |
| `width` | integer | 1024 | Output width (64–2048) |
| `height` | integer | 1024 | Output height (64–2048) |
| `seed` | integer | — | Optional, for reproducibility |
| `num_frames` | integer | — | For video modalities (8–32) |
| `fps` | integer | — | For video modalities (8–30) |
| `mask_url` | string | — | Required for `inpainting` |
| `control_image_url` | string | — | Required for `controlnet` |

### Special inputs

```json
{ "input": { "health_check": true } }
{ "input": { "system_status": true } }
```

`system_status` returns loaded model count, GPU memory usage, and supported modalities. Useful for verifying worker state before submitting long jobs.

### Response format

```json
{
  "output": {
    "modality": "text-to-image",
    "result_type": "image",
    "result_url": "https://...",
    "metadata": {
      "inference_time": 9.2,
      "model_used": "flux.1-schnell",
      "parameters": { "steps": 4, "width": 1024, "height": 1024 }
    }
  }
}
```

---

## Docker build

Four-stage multi-stage build in `docker/Dockerfile`:

| Stage | Inherits from | Purpose |
|---|---|---|
| `base` | `python:3.11-slim` | System libraries (OpenCV, ffmpeg, libGL) + Python dependencies |
| `models` | `base` | Download and validate model weights to `/runpod-volume/models` |
| `development` | `models` | Adds dev tools (pytest, black, flake8, Jupyter); exposes port 8888 |
| `runtime` → `production` | `base` | Copies pre-downloaded weights from `models` stage; non-root user; HEALTHCHECK; CUDA tuning |

The production stage runs as a non-root `worker` user and sets `TORCH_CUDNN_V8_API_ENABLED=1` and `OMP_NUM_THREADS=4`. HuggingFace and Torch caches are placed on the network volume (`HF_HOME=/runpod-volume/cache/hf`) to persist across cold starts.

---

## Startup sequence

`docker/entrypoint.sh` runs the following before handing off to `src/main.py`:

1. **System resources** — checks disk space (50GB minimum on `/runpod-volume`) and available RAM
2. **GPU detection** — runs `nvidia-smi`, sets `CUDA_AVAILABLE`, logs GPU name and memory
3. **Python environment** — verifies `torch`, `transformers`, `diffusers`, `safetensors`, `huggingface_hub` are importable and logs their versions
4. **RunPod SDK** — checks `runpod` is importable and logs its version
5. **Model validation** — counts `.safetensors`/`.bin`/`.pt` files in `MODELS_DIR`; runs `validate_models.py` if present; triggers `download_models.py` if directory is empty
6. **Background health monitor** — spawns a background process that writes a JSON health file every `HEALTH_CHECK_INTERVAL` seconds, tracking worker PID liveness
7. **Signal handling** — `SIGTERM`/`SIGINT`/`SIGQUIT` trigger graceful shutdown: health monitor killed, worker process sent SIGTERM, forced SIGKILL after 10s if still running

The startup sequence has a configurable timeout (`STARTUP_TIMEOUT`, default 300s). The `health` argument (`./entrypoint.sh health`) reads and prints the current JSON health file — used by the Docker `HEALTHCHECK`.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MODELS_DIR` | `/runpod-volume/models` | Model weight directory |
| `VALIDATION_MODE` | `basic` | Model validation level (`basic` or `strict`) |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `HEALTH_CHECK_INTERVAL` | `30` | Health file update interval (seconds) |
| `STARTUP_TIMEOUT` | `300` | Maximum allowed startup time (seconds) |
| `GPU_MEMORY_FRACTION` | `0.9` | GPU memory utilisation ceiling |
| `DEBUG_MODE` | `false` | Enables verbose debug logging in `src/main.py` |
| `ALLOW_ROOT` | — | Set to `true` to allow running as root (not recommended) |

---

## Performance (RTX 4090)

| Modality | Time | Resolution / Frames |
|---|---|---|
| Text-to-image (FLUX.1 Schnell) | 8–15s | 1024×1024 |
| Image-to-video (AnimateDiff) | 15–25s | 16 frames |
| Text-to-video (LTX-Video 2B) | 20–35s | 49 frames |
| ControlNet | 10–18s | 1024×1024 |

**Minimum GPU:** 16GB VRAM · **Recommended:** RTX 4090 (24GB) or A100 (40GB)

---

## RunPod deployment

**Pull the pre-built image:**

```bash
docker pull ghcr.io/mikeblakeway/multi-modal-worker:latest
```

**RunPod template settings:**

- Image: `ghcr.io/mikeblakeway/multi-modal-worker:latest`
- GPU: RTX 4090 (24GB) or A100 (40GB)
- Container disk: 20GB
- Network Volume: 100GB, mounted at `/runpod-volume`
- FlashBoot: enabled (for faster cold starts)

**Required environment variables:**

```env
MODELS_DIR=/runpod-volume/models
VALIDATION_MODE=basic
LOG_LEVEL=INFO
```

---

## Local development

```bash
git clone https://github.com/MikeBlakeway/multi-modal-worker.git
cd multi-modal-worker
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

**Run all tests:**

```bash
python run_tests.py
```

**Run by category:**

```bash
python -m pytest tests/unit/ -v -m unit
python -m pytest tests/integration/ -v -m integration
python -m pytest tests/ -v -m "not gpu and not model"   # skip GPU-dependent tests
```

**Validate the model management framework independently:**

```bash
python validate_framework.py
```

Tests `ModelManager`, `MemoryMonitor`, and `BaseModel` without requiring actual model weights. Exercises model registration, on-demand loading, inference, eviction, and memory checking.

**Build Docker image locally:**

```bash
docker build -t multi-modal-worker -f docker/Dockerfile --target runtime .
```

---

## Testing

pytest is configured in `pytest.ini` with coverage (`--cov=src`) and HTML report output (`htmlcov/`). Test markers:

| Marker | Meaning |
|---|---|
| `unit` | No external dependencies, fast |
| `integration` | May require models or services |
| `gpu` | Requires CUDA GPU |
| `model` | Requires actual model weights |
| `video` | Involves video processing |
| `slow` | Takes longer than 5 seconds |
| `performance` | Benchmark tests |

Coverage is measured across six components: `RequestValidator`, `ResponseFormatter`, `MultiModalHandler`, `BaseHandler`, `LoggingConfig`, and integration routing.

---

## Model storage layout

```
/runpod-volume/models/
├── flux/           ~15GB   FLUX.1 Schnell fp8
├── controlnet/      ~4GB   Canny + Depth adapters
├── animatediff/     ~2GB   Motion adapter
├── ltx-video/       ~8GB   LTX-Video 2B distilled
├── inpaint/         ~6GB   SDXL Inpainting
├── camera/          ~1GB   CameraCtrl
└── shared/          ~4GB   VAE, tokenizers, CLIP
```

Total: ~40GB. A 100GB network volume leaves ~60GB headroom for additional LoRAs, ControlNet variants, and output caching.
