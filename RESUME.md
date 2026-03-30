# RESUME.md — Server Testing & Continuation Guide

## What's Been Done

### ✅ Implemented (Ready for Testing)

1. **VRAMManager Service** (`backend/services/vram_manager/`)
   - Tier classification: HIGH (≥24GB), MEDIUM (16-23GB), LOW (12-15GB), VERY_LOW (8-11GB)
   - Resolution and frame limits per tier
   - Offload strategy selection (NONE / SEQUENTIAL / BLOCK_SWAP)
   - GGUF quantization recommendations per tier
   - Profile endpoint data for frontend

2. **GGUF Model Loader** (`backend/services/gguf_loader/`)
   - Discovers GGUF files in `models/gguf/` or `models/` directory
   - Supports Q8_0, Q5_1, Q4_K_M, Q4_0 quantization levels
   - Loads from `unsloth/LTX-2.3-GGUF` or `Kijai/LTX2.3_comfy` repos
   - State dict loading and key remapping for diffusers compatibility
   - Added `gguf>=0.6.0` to `pyproject.toml` dependencies
   - Added `gguf_checkpoint` model type to download specs

3. **Block Swap** (`backend/services/block_swap/`)
   - ComfyUI-style transformer block swapping between CPU and GPU
   - Configurable blocks to keep on GPU per tier
   - Async CUDA stream prefetch for overlapping compute and transfer
   - Forward hooks on each block for automatic swap during inference
   - Handles models with `transformer_blocks`, `blocks`, or `layers` containers

4. **Low-VRAM Pipeline** (`backend/services/fast_video_pipeline/ltx_low_vram_pipeline.py`)
   - Sequential 6-phase offloading: text encode → image condition → denoise → VAE video → VAE audio → encode
   - Integrates GGUF loading when available
   - Integrates block swap for LOW/VERY_LOW tiers
   - Adaptive VAE tiling config per VRAM tier
   - Conforms to `FastVideoPipeline` protocol (drop-in replacement)

5. **Runtime Policy Change** (`backend/runtime_config/runtime_policy.py`)
   - VRAM gate lowered from 31GB to 8GB (`MIN_LOCAL_VRAM_GB = 8`)
   - RTX 4090 (24GB), RTX 4080 (16GB), RTX 3060 (12GB), RTX 4060 (8GB) all allowed

6. **Pipeline Handler** (`backend/handlers/pipelines_handler.py`)
   - VRAM-aware pipeline selection: standard path for ≥31GB, low-VRAM path for <31GB
   - Auto-discovers GGUF models and uses recommended quantization per tier
   - Passes `gpu_info` service for testable VRAM queries

7. **Web Mode** (frontend + backend)
   - `frontend/lib/electron-shim.ts` — Full `window.electronAPI` shim for browser operation
   - `frontend/main.tsx` — Imports shim before app initialization
   - `frontend/lib/backend.ts` — Web-mode detection for backend URL
   - `backend/_routes/web_routes.py` — `/web/app-info`, `/web/gpu-info`, `/web/vram-profile`, `/web/file/read`, `/web/file/save`, `/web/gguf-models`
   - `backend/app_factory.py` — Web routes registered
   - `run.sh` / `run.py` — Cross-platform launcher scripts

8. **Local-Only Default**
   - `settings.json` now defaults to `use_local_text_encoder: true`
   - No API key needed for basic text-to-video generation

9. **Tests** — 332 tests passing (all existing + new)
   - `tests/test_vram_manager.py` — 34 tests (tier, resolution, frames, strategy, GGUF, profile)
   - `tests/test_gguf_loader.py` — 13 tests (discovery, info, quant detection)
   - `tests/test_block_swap.py` — 10 tests (setup, forward pass, offload, stats)
   - `tests/test_web_routes.py` — 6 tests (app-info, gpu-info, vram-profile, file ops)
   - `tests/test_runtime_policy_decision.py` — 17 tests (updated for 8GB threshold)

---

## What Needs Testing on Server (with GPU)

### Step 1: Environment Setup

```bash
cd /path/to/LTX-Desktop-Linux/backend
uv sync
```

### Step 2: Download GGUF Models

Place GGUF models in the models directory:

```bash
# Option A: Download from HuggingFace directly
export LTX_APP_DATA_DIR="$HOME/.ltx-desktop"
mkdir -p "$LTX_APP_DATA_DIR/models/gguf"

# For 24GB GPU (Q8_0 — best quality, ~12GB):
huggingface-cli download unsloth/LTX-2.3-GGUF distilled/ltx-2.3-22b-distilled-Q8_0.gguf \
  --local-dir "$LTX_APP_DATA_DIR/models/gguf"

# For 16GB GPU (Q5_1 — good quality, ~8GB):
huggingface-cli download unsloth/LTX-2.3-GGUF distilled/ltx-2.3-22b-distilled-Q5_1.gguf \
  --local-dir "$LTX_APP_DATA_DIR/models/gguf"

# For 12GB GPU (Q4_K_M — decent quality, ~6GB):
huggingface-cli download unsloth/LTX-2.3-GGUF distilled/ltx-2.3-22b-distilled-Q4_K_M.gguf \
  --local-dir "$LTX_APP_DATA_DIR/models/gguf"

# For 8GB GPU (Q4_0 — lowest quality, ~5GB):
huggingface-cli download unsloth/LTX-2.3-GGUF distilled/ltx-2.3-22b-distilled-Q4_0.gguf \
  --local-dir "$LTX_APP_DATA_DIR/models/gguf"
```

Also need the standard models (text encoder, upsampler, etc.):
```bash
# Standard checkpoint (needed even with GGUF for architecture init):
huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-22b-distilled.safetensors \
  --local-dir "$LTX_APP_DATA_DIR/models"

# Upsampler:
huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --local-dir "$LTX_APP_DATA_DIR/models"

# Text encoder (for local-only mode):
huggingface-cli download Lightricks/gemma-3-12b-it-qat-q4_0-unquantized \
  --local-dir "$LTX_APP_DATA_DIR/models/gemma-3-12b-it-qat-q4_0-unquantized"
```

### Step 3: Run the Backend

```bash
# Web mode:
LTX_APP_DATA_DIR="$HOME/.ltx-desktop" python run.py --port 8000

# Or directly:
cd backend
LTX_APP_DATA_DIR="$HOME/.ltx-desktop" LTX_WEB_MODE=1 python ltx2_server.py
```

### Step 4: Test Endpoints

```bash
# Check VRAM profile
curl http://localhost:8000/web/vram-profile | python -m json.tool

# Check GGUF models
curl http://localhost:8000/web/gguf-models | python -m json.tool

# Check GPU info
curl http://localhost:8000/web/gpu-info | python -m json.tool

# Check health
curl http://localhost:8000/health | python -m json.tool
```

### Step 5: Test Generation

```bash
# Text-to-video at 540p (should work on all GPUs ≥8GB)
curl -X POST http://localhost:8000/generate-video \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A golden sunset over calm ocean waves",
    "resolution": "540p",
    "aspectRatio": "16:9",
    "duration": "4",
    "fps": "24",
    "model": "fast",
    "cameraMotion": "none",
    "negativePrompt": "",
    "audio": false
  }'
```

### Step 6: Monitor VRAM During Generation

In a separate terminal:
```bash
watch -n 0.5 nvidia-smi
```

Expected VRAM usage patterns:
- **24GB GPU**: Peak ~18-22GB (sequential offload, no block swap)
- **16GB GPU**: Peak ~12-14GB (block swap, Q5_1 GGUF)
- **12GB GPU**: Peak ~8-10GB (aggressive block swap, Q4_K_M GGUF)

---

## What Still Needs Work

### Must Do Before Release
- [ ] Test GGUF loading actually works end-to-end (the `gguf` library tensor format → torch conversion)
- [ ] Verify `TilingConfig` constructor accepts the parameters we're passing (check ltx-core API)
- [ ] Verify block swap finds the right block container name in LTX transformer
- [ ] Test on actual 12GB, 16GB, and 24GB GPUs
- [ ] Build frontend and test static file serving in web mode
- [ ] Add `aiofiles` or similar for proper static file serving in `app_factory.py`

### Nice To Have
- [ ] VRAM-aware resolution selector in frontend `GenSpace.tsx`
- [ ] Real-time VRAM usage indicator in frontend
- [ ] GGUF model download UI in settings
- [ ] INT4/INT8 quantization via bitsandbytes as alternative to GGUF
- [ ] MPS (macOS) local generation support
- [ ] Progress reporting for model offloading phases
- [ ] CPU fallback for VAE decode on very low VRAM

### Known Issues to Investigate
- The GGUF state dict key mapping may need adjustment — test with actual models
- Block swap CUDA stream prefetch needs GPU testing (may need `non_blocking=True` tuning)
- The `DistilledPipeline` from ltx-pipelines may have its own offloading that conflicts with ours
- Some ltx-core TilingConfig constructors may have different parameter names

---

## File Inventory

### New Files (15)
```
backend/services/vram_manager/__init__.py
backend/services/vram_manager/vram_manager.py
backend/services/gguf_loader/__init__.py
backend/services/gguf_loader/gguf_loader.py
backend/services/block_swap/__init__.py
backend/services/block_swap/block_swap.py
backend/services/fast_video_pipeline/ltx_low_vram_pipeline.py
backend/_routes/web_routes.py
backend/tests/test_vram_manager.py
backend/tests/test_gguf_loader.py
backend/tests/test_block_swap.py
backend/tests/test_web_routes.py
frontend/lib/electron-shim.ts
run.sh
run.py
```

### Modified Files (12)
```
backend/api_types.py                          # Added gguf_checkpoint to ModelFileType
backend/app_factory.py                        # Added web_router
backend/app_handler.py                        # Added gguf_checkpoint to available_files, wired gpu_info to pipelines
backend/handlers/pipelines_handler.py         # VRAM-aware pipeline selection, low-VRAM path
backend/pyproject.toml                        # Added gguf dependency
backend/runtime_config/model_download_specs.py # Added gguf_checkpoint spec, updated MODEL_FILE_ORDER
backend/runtime_config/runtime_policy.py      # Lowered VRAM gate from 31GB to 8GB
backend/tests/test_runtime_policy_decision.py # Updated for new VRAM thresholds
frontend/lib/backend.ts                       # Web-mode URL detection
frontend/main.tsx                             # Import electron-shim
settings.json                                 # Default use_local_text_encoder: true
PLAN.md                                       # Architecture plan
```
