# LTX Desktop: Low-VRAM Local-Only Web App — Implementation Plan

## Executive Summary

Transform LTX Desktop from an Electron app requiring 31GB+ VRAM (or API keys) into a **local-only, platform-independent web application** that runs on consumer GPUs like the RTX 4090 (24GB), RTX 3090 (24GB), RTX 4080 (16GB), and even 12GB/8GB GPUs. No API keys required. Inspired by ComfyUI's low-VRAM workflow strategies.

### Key Techniques
1. **Sequential Model Offloading** — Only one model on GPU at a time
2. **GGUF Quantized Models** — Load transformer from Q4/Q5/Q8 GGUF files (5-12GB vs 43GB)
3. **Block Swap** — Swap transformer blocks between CPU/GPU during forward pass
4. **Adaptive Tiling** — Smaller VAE decode tiles for lower VRAM

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Architecture Changes: Electron → Web App](#2-architecture-changes-electron--web-app)
3. [Low-VRAM Strategy](#3-low-vram-strategy)
4. [Implementation Phases](#4-implementation-phases)
5. [Testing Plan](#5-testing-plan)
6. [Risk Assessment](#6-risk-assessment)

---

## 1. Current State Analysis

### What blocks low-VRAM GPUs today

```python
# backend/runtime_config/runtime_policy.py
def decide_force_api_generations(system, cuda_available, vram_gb):
    if vram_gb < 31:
        return True  # Forces API mode — no local generation!
```

The app **hard-gates** local generation at 31GB VRAM. A 4090 with 24GB is forced into API-only mode, requiring an LTX API key.

### Current memory bottlenecks

| Component | Approx VRAM (bf16) | Notes |
|---|---|---|
| Transformer (LTX-2 distilled) | ~8-12 GB | Largest single model |
| Text Encoder (Gemma) | ~4-6 GB | Currently offloaded to CPU between uses |
| Video VAE Encoder | ~1-2 GB | Used for image conditioning |
| Video VAE Decoder | ~2-4 GB | Decodes latents to pixels |
| Audio VAE + Vocoder | ~1-2 GB | Decodes audio latents |
| Spatial Upsampler | ~1-2 GB | Post-processing upscale |
| Working memory (activations) | ~4-8 GB | Depends on resolution/frames |
| **Total peak** | **~22-36 GB** | All loaded simultaneously |

### Current memory management (partial)

The codebase already has some memory-aware patterns:
- `DistilledNativePipeline.__call__()` in `ltx_pipeline_common.py` does sequential load/unload: loads text encoder → encodes → deletes → loads transformer → denoises → deletes → loads VAE → decodes
- `LTXTextEncoder` monkey-patches `ModelLedger.text_encoder()` to cache on CPU and move to GPU on demand
- `TorchCleaner` calls `torch.cuda.empty_cache()` + `gc.collect()`
- FP8 quantization is applied when CUDA is available (`QuantizationPolicy.fp8_cast()`)

### What's missing for low-VRAM

1. **No model offloading strategy** — `DistilledPipeline` (used by `LTXFastVideoPipeline`) loads everything to GPU at init
2. **No tiled VAE decoding control** — `TilingConfig.default()` is used but no adaptive sizing
3. **No resolution/frame capping** based on available VRAM
4. **The 31GB gate** blocks everything
5. **No CPU offload for transformer attention** during inference
6. **No quantization below FP8** (e.g., INT8/INT4 for very low VRAM)

### Electron dependencies in frontend

The frontend uses `window.electronAPI` for:
- Backend URL discovery (`getBackend`)
- File system operations (`readLocalFile`, `showSaveDialog`, `saveFile`, `copyToProjectAssets`)
- App metadata (`getAppInfo`, `checkGpu`, `getResourcePath`)
- First-run flow (`checkFirstRun`, `acceptLicense`, `completeSetup`)
- Updates (`checkForUpdates`, `installUpdate`)
- Export (`exportVideo`, `exportStill`)

---

## 2. Architecture Changes: Electron → Web App

### 2.1 Target Architecture

```
┌──────────────────────────────────────────┐
│  Browser (any modern browser)            │
│  React + Tailwind SPA                    │
│  Served by FastAPI static files          │
│  ┌─────────────────────────────────┐     │
│  │  electronAPI shim layer         │     │
│  │  (replaces window.electronAPI)  │     │
│  └─────────────────────────────────┘     │
└──────────────────┬───────────────────────┘
                   │ HTTP / WebSocket
┌──────────────────▼───────────────────────┐
│  Python Backend (FastAPI)                │
│  Port 8000 (configurable)               │
│  ├─ /api/*         existing routes       │
│  ├─ /web/*         new web-mode routes   │
│  │   ├─ /web/file/read                   │
│  │   ├─ /web/file/save                   │
│  │   ├─ /web/file/browse                 │
│  │   ├─ /web/app-info                    │
│  │   └─ /web/gpu-info                    │
│  ├─ /static/*     Vite build output      │
│  └─ /            index.html (SPA)        │
└──────────────────────────────────────────┘
```

### 2.2 ElectronAPI Shim Layer

Create `frontend/lib/electron-shim.ts` that implements every method on `window.electronAPI` but routes through HTTP to the backend instead of IPC.

```typescript
// frontend/lib/electron-shim.ts
const isElectron = typeof window !== 'undefined' && window.electronAPI !== undefined;

const shimAPI: ElectronAPI = {
  getBackend: async () => {
    // In web mode, backend is same origin
    const url = window.location.origin;
    return { url, token: localStorage.getItem('ltx_auth_token') || '' };
  },
  
  readLocalFile: async (filePath: string) => {
    const resp = await fetch(`/web/file/read`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: filePath }),
    });
    return resp.json();
  },
  
  showSaveDialog: async (options) => {
    // In web mode, use browser download or backend file picker
    const resp = await fetch('/web/file/browse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(options),
    });
    const { path } = await resp.json();
    return path;
  },
  
  // ... etc for all methods
};

// Auto-register if not in Electron
if (!isElectron) {
  (window as any).electronAPI = shimAPI;
}
```

### 2.3 New Backend Routes for Web Mode

```
backend/_routes/web_routes.py
```

New endpoints:
| Route | Purpose | Replaces |
|---|---|---|
| `GET /web/app-info` | App version, paths, mode | `getAppInfo` IPC |
| `GET /web/gpu-info` | GPU name, VRAM | `checkGpu` IPC |
| `POST /web/file/read` | Read file as base64 | `readLocalFile` IPC |
| `POST /web/file/save` | Write file to disk | `saveFile` IPC |
| `POST /web/file/browse` | Server-side file dialog (tkinter/zenity) | `showSaveDialog` IPC |
| `POST /web/file/download` | Trigger browser download of a file | `exportVideo` IPC |
| `GET /web/outputs/{filename}` | Serve generated videos/images | Static file serving |

### 2.4 Static File Serving

```python
# In app_factory.py or a new web_app_factory.py
from fastapi.staticfiles import StaticFiles

if web_mode:
    # Serve Vite build
    app.mount("/static", StaticFiles(directory="frontend/dist/assets"), name="static")
    # Serve outputs for video playback
    app.mount("/outputs", StaticFiles(directory=outputs_dir), name="outputs")
    # SPA fallback
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        return FileResponse("frontend/dist/index.html")
```

### 2.5 Launch Script

```bash
#!/bin/bash
# run.sh — one-command local launch
cd "$(dirname "$0")"
export LTX_APP_DATA_DIR="${LTX_APP_DATA_DIR:-$HOME/.ltx-desktop}"
export LTX_WEB_MODE=1
python backend/ltx2_server.py
# Opens browser to http://localhost:8000
```

### 2.6 Files to Change

| File | Change |
|---|---|
| `frontend/lib/backend.ts` | Add web-mode detection, skip `getBackend` IPC |
| `frontend/lib/electron-shim.ts` | **NEW** — full shim implementation |
| `frontend/main.tsx` or entry | Import shim before app init |
| `backend/app_factory.py` | Add static file serving + web routes |
| `backend/_routes/web_routes.py` | **NEW** — file ops, app info, GPU info |
| `backend/ltx2_server.py` | Add `LTX_WEB_MODE` flag, serve on 0.0.0.0 optionally |
| `vite.config.ts` | Ensure build output is servable (already likely fine) |

---

## 3. Low-VRAM Strategy

### 3.1 VRAM Tiers & Capabilities

Inspired by ComfyUI's tiered memory management:

| Tier | VRAM | Strategy | Max Resolution | Max Frames |
|---|---|---|---|---|
| **High** | ≥24 GB | FP8 transformer, all on GPU | 1080p | 161 (6s@25fps) |
| **Medium** | 16-23 GB | FP8 + aggressive offload + tiled VAE | 720p | 97 (4s@24fps) |
| **Low** | 12-15 GB | FP8 + sequential offload + tiled VAE + reduced resolution | 540p | 65 (2.5s@25fps) |
| **Very Low** | 8-11 GB | INT8/INT4 quant + CPU offload + small tiled VAE | 480p | 33 (1.3s@25fps) |

### 3.2 Core Technique: Sequential Model Offloading

This is the ComfyUI approach — only one major model on GPU at a time.

```python
# New file: backend/services/vram_manager/vram_manager.py

class VRAMTier(Enum):
    HIGH = "high"       # ≥24 GB
    MEDIUM = "medium"   # 16-23 GB
    LOW = "low"         # 12-15 GB
    VERY_LOW = "very_low"  # 8-11 GB

class VRAMManager:
    """ComfyUI-inspired VRAM management for LTX pipelines."""
    
    def __init__(self, device: torch.device, total_vram_gb: int):
        self.device = device
        self.total_vram_gb = total_vram_gb
        self.tier = self._classify_tier(total_vram_gb)
        self._loaded_models: dict[str, torch.nn.Module] = {}
    
    @staticmethod
    def _classify_tier(vram_gb: int) -> VRAMTier:
        if vram_gb >= 24:
            return VRAMTier.HIGH
        if vram_gb >= 16:
            return VRAMTier.MEDIUM
        if vram_gb >= 12:
            return VRAMTier.LOW
        return VRAMTier.VERY_LOW
    
    def ensure_on_gpu(self, model_name: str, model: torch.nn.Module) -> None:
        """Move model to GPU, evicting others if necessary."""
        if model_name in self._loaded_models:
            return
        
        # Evict all other models to CPU first
        for name, m in list(self._loaded_models.items()):
            if name != model_name:
                m.to("cpu")
                del self._loaded_models[name]
        
        torch.cuda.empty_cache()
        gc.collect()
        model.to(self.device)
        self._loaded_models[model_name] = model
    
    def offload_to_cpu(self, model_name: str) -> None:
        if model_name in self._loaded_models:
            self._loaded_models[model_name].to("cpu")
            del self._loaded_models[model_name]
            torch.cuda.empty_cache()
    
    def get_max_resolution(self) -> tuple[int, int]:
        """Return (width, height) for 16:9 based on tier."""
        match self.tier:
            case VRAMTier.HIGH:
                return (1920, 1088)
            case VRAMTier.MEDIUM:
                return (1280, 704)
            case VRAMTier.LOW:
                return (960, 544)
            case VRAMTier.VERY_LOW:
                return (768, 448)
    
    def get_max_frames(self, width: int, height: int, fps: int) -> int:
        """Estimate max frames based on VRAM budget."""
        pixels_per_frame = width * height
        # Rough heuristic: each frame in latent space costs ~X MB
        vram_for_inference_gb = self.total_vram_gb * 0.4  # Reserve 60% for model
        frames_budget = int((vram_for_inference_gb * 1024) / (pixels_per_frame * 0.001))
        frames_budget = ((frames_budget // 8) * 8) + 1
        return max(9, min(frames_budget, 201))
    
    def get_quantization_policy(self):
        """Return appropriate quantization for this tier."""
        from ltx_core.quantization import QuantizationPolicy
        match self.tier:
            case VRAMTier.HIGH | VRAMTier.MEDIUM:
                return QuantizationPolicy.fp8_cast()
            case VRAMTier.LOW:
                return QuantizationPolicy.fp8_cast()  # Could add INT8 later
            case VRAMTier.VERY_LOW:
                return QuantizationPolicy.fp8_cast()  # Placeholder for INT4
```

### 3.3 Modified Pipeline: Low-VRAM Distilled Pipeline

Create a new pipeline implementation that wraps the existing `DistilledNativePipeline` pattern (which already does sequential offloading) and makes it the default for low-VRAM:

```python
# New file: backend/services/fast_video_pipeline/ltx_low_vram_pipeline.py

class LTXLowVRAMPipeline:
    """Low-VRAM pipeline using sequential model offloading (ComfyUI-style)."""
    pipeline_kind: Final = "fast"
    
    def __init__(self, checkpoint_path, gemma_root, upsampler_path, device, vram_manager):
        self.vram_manager = vram_manager
        self.device = device
        
        # Load model components to CPU initially
        self.model_ledger = ModelLedger(
            dtype=torch.bfloat16,
            device=torch.device("cpu"),  # Key: load to CPU first!
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            loras=None,
            quantization=vram_manager.get_quantization_policy(),
        )
        self.pipeline_components = PipelineComponents(
            dtype=torch.bfloat16, 
            device=device,
        )
        # Upsampler loaded on demand
        self._upsampler_path = upsampler_path
        self._upsampler = None
    
    @torch.inference_mode()
    def generate(self, prompt, seed, height, width, num_frames, 
                 frame_rate, images, output_path):
        # Phase 1: Text encoding (text encoder on GPU)
        text_encoder = self.model_ledger.text_encoder()
        self.vram_manager.ensure_on_gpu("text_encoder", text_encoder)
        context = encode_text(text_encoder, [prompt])[0]
        video_context, audio_context = context
        self.vram_manager.offload_to_cpu("text_encoder")
        
        # Phase 2: Image conditioning (video encoder on GPU briefly)
        if images:
            video_encoder = self.model_ledger.video_encoder()
            self.vram_manager.ensure_on_gpu("video_encoder", video_encoder)
            conditionings = image_conditionings_by_replacing_latent(...)
            self.vram_manager.offload_to_cpu("video_encoder")
        
        # Phase 3: Denoising (transformer on GPU)
        transformer = self.model_ledger.transformer()
        self.vram_manager.ensure_on_gpu("transformer", transformer)
        video_state, audio_state = denoise_audio_video(...)
        self.vram_manager.offload_to_cpu("transformer")
        
        # Phase 4: VAE decode (decoder on GPU, with tiling for low VRAM)
        tiling_config = self._adaptive_tiling_config()
        video_decoder = self.model_ledger.video_decoder()
        self.vram_manager.ensure_on_gpu("video_decoder", video_decoder)
        decoded_video = vae_decode_video(
            video_state.latent, video_decoder, tiling_config
        )
        self.vram_manager.offload_to_cpu("video_decoder")
        
        # Phase 5: Audio decode
        audio_decoder = self.model_ledger.audio_decoder()
        vocoder = self.model_ledger.vocoder()
        self.vram_manager.ensure_on_gpu("audio_decoder", audio_decoder)
        decoded_audio = vae_decode_audio(
            audio_state.latent, audio_decoder, vocoder
        )
        self.vram_manager.offload_to_cpu("audio_decoder")
        
        # Phase 6: Encode output
        encode_video(video=decoded_video, audio=decoded_audio, 
                     fps=int(frame_rate), output_path=output_path)
    
    def _adaptive_tiling_config(self):
        """Smaller tiles for lower VRAM tiers."""
        from ltx_core.model.video_vae import TilingConfig
        match self.vram_manager.tier:
            case VRAMTier.HIGH:
                return TilingConfig.default()
            case VRAMTier.MEDIUM:
                return TilingConfig(spatial_tile_size=128, temporal_tile_size=8)
            case _:
                return TilingConfig(spatial_tile_size=64, temporal_tile_size=4)
```

### 3.4 Changes to Runtime Policy

```python
# backend/runtime_config/runtime_policy.py — MODIFIED

def decide_force_api_generations(system, cuda_available, vram_gb):
    """No longer force API mode for low-VRAM CUDA GPUs."""
    if system == "Darwin":
        # macOS MPS support is experimental; keep API for now
        # (could be relaxed later)
        return True
    
    if system in ("Windows", "Linux"):
        if not cuda_available:
            return True
        if vram_gb is None:
            return True
        # NEW: Allow local generation for GPUs with ≥8 GB
        return vram_gb < 8
    
    return True
```

### 3.5 Changes to Pipeline Selection

```python
# backend/handlers/pipelines_handler.py — MODIFIED

def _create_video_pipeline(self, model_type):
    # Determine VRAM and pick implementation
    vram_gb = self._get_vram_gb()
    
    if vram_gb >= 31:
        # Original path — everything on GPU
        return self._create_high_vram_pipeline(model_type)
    else:
        # Low-VRAM path — sequential offloading
        return self._create_low_vram_pipeline(model_type, vram_gb)

def _create_low_vram_pipeline(self, model_type, vram_gb):
    from services.fast_video_pipeline.ltx_low_vram_pipeline import LTXLowVRAMPipeline
    from services.vram_manager.vram_manager import VRAMManager
    
    vram_manager = VRAMManager(self.config.device, vram_gb)
    pipeline = LTXLowVRAMPipeline(
        checkpoint_path=...,
        gemma_root=...,
        upsampler_path=...,
        device=self.config.device,
        vram_manager=vram_manager,
    )
    return VideoPipelineState(pipeline=pipeline, ...)
```

### 3.6 Frontend VRAM-Aware UI

Expose VRAM tier info to frontend so the UI can:
1. Show max resolution/duration constraints
2. Display VRAM usage indicator
3. Auto-select optimal settings

```python
# New route: /api/vram-profile
@router.get("/vram-profile")
def get_vram_profile(handler = Depends(get_state_service)):
    vram_gb = handler.gpu_info.get_vram_total_gb() or 0
    manager = VRAMManager(handler.config.device, vram_gb)
    max_w, max_h = manager.get_max_resolution()
    return {
        "tier": manager.tier.value,
        "vram_total_gb": vram_gb,
        "max_resolution_width": max_w,
        "max_resolution_height": max_h,
        "max_frames_540p": manager.get_max_frames(960, 544, 25),
        "max_frames_720p": manager.get_max_frames(1280, 704, 25),
        "max_frames_1080p": manager.get_max_frames(1920, 1088, 25),
        "quantization": "fp8",
        "offload_strategy": "sequential",
    }
```

### 3.7 Remove API Key Requirement for Local Use

Changes to settings and frontend to remove API key gates:

1. **`AppSettingsContext.tsx`**: Remove any UI that gates features behind `hasLtxApiKey` when `forceApiGenerations` is false
2. **`settings.json`**: Remove `ltx_api_key` and `fal_api_key` from default settings
3. **Text encoding**: Default to local text encoding (`use_local_text_encoder: true`) when no API key is set
4. **`GenSpace.tsx`**: Remove "API key required" prompts/banners for local generation mode

---

## 4. Implementation Phases

### Phase 1: Unlock Low-VRAM Local Generation (Core — Week 1-2)

**Goal**: Make local generation work on 24GB GPUs.

| # | Task | Files | Priority |
|---|---|---|---|
| 1.1 | Change VRAM gate from 31GB to 8GB | `runtime_policy.py` | P0 |
| 1.2 | Create `VRAMManager` service | `services/vram_manager/` (NEW) | P0 |
| 1.3 | Create `LTXLowVRAMPipeline` | `services/fast_video_pipeline/ltx_low_vram_pipeline.py` (NEW) | P0 |
| 1.4 | Modify `PipelinesHandler` to select pipeline by VRAM | `handlers/pipelines_handler.py` | P0 |
| 1.5 | Add `/vram-profile` endpoint | `_routes/` | P1 |
| 1.6 | Default to local text encoding when no API key | `handlers/text_handler.py`, `state/app_settings.py` | P0 |
| 1.7 | Add VRAMManager to `ServiceBundle` and fakes | `app_handler.py`, `tests/fakes/` | P0 |

### Phase 2: Web App Mode (Week 2-3)

**Goal**: Run as standalone web app without Electron.

| # | Task | Files | Priority |
|---|---|---|---|
| 2.1 | Create `electron-shim.ts` | `frontend/lib/electron-shim.ts` (NEW) | P0 |
| 2.2 | Create web-mode backend routes | `backend/_routes/web_routes.py` (NEW) | P0 |
| 2.3 | Add static file serving to `app_factory.py` | `backend/app_factory.py` | P0 |
| 2.4 | Create `run.sh` / `run.py` launcher | Root directory (NEW) | P1 |
| 2.5 | Modify `vite.config.ts` for web build | `vite.config.ts` | P1 |
| 2.6 | Handle file downloads via browser API | `frontend/lib/electron-shim.ts` | P1 |
| 2.7 | Replace `window.electronAPI` typing with conditional | `frontend/types/` | P1 |
| 2.8 | Video/image playback via backend URLs | `frontend/` components | P1 |

### Phase 3: Medium/Low VRAM Support (Week 3-4)

**Goal**: Support 16GB and 12GB GPUs.

| # | Task | Files | Priority |
|---|---|---|---|
| 3.1 | Adaptive tiling config based on VRAM | `ltx_low_vram_pipeline.py` | P0 |
| 3.2 | Resolution auto-capping in frontend | `GenSpace.tsx` | P1 |
| 3.3 | Frame count limits based on VRAM profile | `GenSpace.tsx`, `VideoGenerationHandler` | P1 |
| 3.4 | Memory estimation before generation | `VRAMManager` | P1 |
| 3.5 | Graceful OOM handling and recovery | `video_generation_handler.py` | P1 |
| 3.6 | VRAM usage monitoring endpoint | `_routes/`, `gpu_info` | P2 |

### Phase 4: Polish & Very Low VRAM (Week 4-5)

| # | Task | Files | Priority |
|---|---|---|---|
| 4.1 | INT8 quantization option | `VRAMManager`, new quantization service | P2 |
| 4.2 | CPU fallback for VAE decode | `ltx_low_vram_pipeline.py` | P2 |
| 4.3 | Progress reporting during offloading | `generation_handler.py` | P2 |
| 4.4 | Settings UI for VRAM tier override | Frontend settings | P2 |
| 4.5 | Documentation (README update) | `README.md` | P1 |

---

## 5. Testing Plan

### 5.1 Principles

Following the existing project conventions:
- **Integration-first** using `TestClient` against real FastAPI app
- **No mocks** — swap services via `ServiceBundle` fakes only
- **Pyright strict mode** enforced
- **Existing CI**: `pnpm typecheck` + `pnpm backend:test` + frontend Vite build

### 5.2 Unit-Level Tests (New Files)

#### `tests/test_vram_manager.py` — VRAM tier classification & limits

```python
"""Test VRAM tier classification, resolution limits, and frame budgets."""

def test_vram_tier_classification():
    """Verify correct tier assignment for known VRAM values."""
    assert VRAMManager.classify_tier(24) == VRAMTier.HIGH
    assert VRAMManager.classify_tier(16) == VRAMTier.MEDIUM
    assert VRAMManager.classify_tier(12) == VRAMTier.LOW
    assert VRAMManager.classify_tier(8) == VRAMTier.VERY_LOW

def test_max_resolution_per_tier():
    """Each tier should have expected max resolution."""
    high = VRAMManager(torch.device("cpu"), 24)
    assert high.get_max_resolution() == (1920, 1088)
    
    medium = VRAMManager(torch.device("cpu"), 16)
    assert medium.get_max_resolution() == (1280, 704)
    
    low = VRAMManager(torch.device("cpu"), 12)
    assert low.get_max_resolution() == (960, 544)

def test_max_frames_decreases_with_resolution():
    """Higher resolution should allow fewer frames."""
    mgr = VRAMManager(torch.device("cpu"), 24)
    frames_540 = mgr.get_max_frames(960, 544, 25)
    frames_1080 = mgr.get_max_frames(1920, 1088, 25)
    assert frames_540 > frames_1080

def test_quantization_policy_by_tier():
    """All tiers should return a valid quantization policy."""
    for vram in [8, 12, 16, 24]:
        mgr = VRAMManager(torch.device("cpu"), vram)
        policy = mgr.get_quantization_policy()
        assert policy is not None

def test_min_frames_floor():
    """Even at lowest tier, should return at least 9 frames."""
    mgr = VRAMManager(torch.device("cpu"), 8)
    frames = mgr.get_max_frames(1920, 1088, 25)
    assert frames >= 9
```

#### `tests/test_runtime_policy_low_vram.py` — Updated policy tests

```python
"""Test that runtime policy allows local generation on consumer GPUs."""

def test_4090_allows_local_generation():
    assert not decide_force_api_generations("Linux", True, 24)

def test_3060_allows_local_generation():
    assert not decide_force_api_generations("Linux", True, 12)

def test_8gb_allows_local_generation():
    assert not decide_force_api_generations("Linux", True, 8)

def test_6gb_forces_api():
    assert decide_force_api_generations("Linux", True, 6)

def test_no_cuda_forces_api():
    assert decide_force_api_generations("Linux", False, 24)

def test_macos_forces_api():
    """macOS still uses API (MPS not yet supported for low VRAM)."""
    assert decide_force_api_generations("Darwin", False, 32)
```

#### `tests/test_vram_profile_endpoint.py` — VRAM profile API

```python
"""Test /vram-profile endpoint returns valid tier info."""

def test_vram_profile_returns_tier(client, fake_services):
    fake_services.gpu_info.set_vram_gb(24)
    resp = client.get("/vram-profile")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tier"] == "high"
    assert data["vram_total_gb"] == 24
    assert data["max_resolution_width"] == 1920

def test_vram_profile_low_vram(client, fake_services):
    fake_services.gpu_info.set_vram_gb(12)
    resp = client.get("/vram-profile")
    data = resp.json()
    assert data["tier"] == "low"
    assert data["max_resolution_width"] == 960
```

### 5.3 Integration Tests — Low-VRAM Pipeline Selection

#### `tests/test_low_vram_pipeline.py`

```python
"""Test that correct pipeline is selected based on VRAM."""

def test_low_vram_selects_offloading_pipeline(client, test_state, 
                                                create_fake_model_files,
                                                fake_services):
    """When VRAM < 31GB, the low-VRAM pipeline should be used."""
    create_fake_model_files()
    fake_services.gpu_info.set_vram_gb(24)
    test_state.models.refresh_available_files()
    
    # Trigger pipeline load
    state = test_state.pipelines.load_gpu_pipeline("fast", should_warm=False)
    assert state.pipeline.pipeline_kind == "fast"
    # Verify it's the low-VRAM variant (check for vram_manager attribute)
    assert hasattr(state.pipeline, 'vram_manager')

def test_high_vram_selects_standard_pipeline(client, test_state,
                                              create_fake_model_files,
                                              fake_services):
    """When VRAM >= 31GB, the standard pipeline should be used."""
    create_fake_model_files()
    fake_services.gpu_info.set_vram_gb(48)
    test_state.models.refresh_available_files()
    
    state = test_state.pipelines.load_gpu_pipeline("fast", should_warm=False)
    assert state.pipeline.pipeline_kind == "fast"
    assert not hasattr(state.pipeline, 'vram_manager')
```

### 5.4 Integration Tests — Web Mode Routes

#### `tests/test_web_routes.py`

```python
"""Test web-mode file and app-info endpoints."""

def test_web_app_info(client):
    resp = client.get("/web/app-info")
    assert resp.status_code == 200
    data = resp.json()
    assert "version" in data
    assert data["mode"] == "web"

def test_web_gpu_info(client, fake_services):
    fake_services.gpu_info.set_vram_gb(24)
    resp = client.get("/web/gpu-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert data["vram"] == 24 * 1024

def test_web_file_read(client, tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")
    resp = client.post("/web/file/read", json={"path": str(test_file)})
    assert resp.status_code == 200

def test_web_file_read_rejects_traversal(client):
    resp = client.post("/web/file/read", json={"path": "/etc/passwd"})
    assert resp.status_code in (400, 403)

def test_web_file_save(client, tmp_path):
    out = tmp_path / "output.txt"
    resp = client.post("/web/file/save", json={
        "path": str(out),
        "content": "hello world",
    })
    assert resp.status_code == 200
    assert out.read_text() == "hello world"
```

### 5.5 Integration Tests — Generation Without API Key

#### `tests/test_local_only_generation.py`

```python
"""Test that video generation works without any API keys."""

def test_generate_video_no_api_key(client, test_state, create_fake_model_files, 
                                    fake_services):
    """Local generation should work with empty API keys."""
    create_fake_model_files()
    test_state.models.refresh_available_files()
    # Ensure no API keys set
    test_state.state.app_settings.ltx_api_key = ""
    test_state.state.app_settings.fal_api_key = ""
    test_state.state.app_settings.use_local_text_encoder = True
    
    resp = client.post("/generate-video", json={
        "prompt": "A sunset over the ocean",
        "resolution": "540p",
        "aspectRatio": "16:9",
        "duration": "4",
        "fps": "24",
        "model": "fast",
        "cameraMotion": "none",
        "negativePrompt": "",
        "audio": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "complete"

def test_text_encoding_defaults_to_local_without_key(test_state):
    """When no API key is set, text encoding should use local encoder."""
    test_state.state.app_settings.ltx_api_key = ""
    handler = test_state.text
    assert handler.should_use_local_encoding()
```

### 5.6 Fake Services Updates

#### `tests/fakes/services.py` — New/Modified Fakes

```python
# Add to existing FakeGpuInfo
class FakeGpuInfo:
    _vram_gb: int = 24
    
    def set_vram_gb(self, gb: int) -> None:
        self._vram_gb = gb
    
    def get_vram_total_gb(self) -> int:
        return self._vram_gb
    
    def get_gpu_info(self) -> GpuTelemetryPayload:
        return {"name": "Fake GPU", "vram": self._vram_gb * 1024, "vramUsed": 0}

# New fake for VRAMManager
class FakeVRAMManager:
    """Fake VRAM manager that doesn't actually move tensors."""
    def __init__(self, tier: VRAMTier = VRAMTier.HIGH):
        self.tier = tier
    
    def ensure_on_gpu(self, name, model):
        pass  # No-op in tests
    
    def offload_to_cpu(self, name):
        pass  # No-op in tests

# New fake pipeline that supports vram_manager
class FakeLowVRAMPipeline:
    pipeline_kind = "fast"
    
    def __init__(self, vram_manager=None, **kwargs):
        self.vram_manager = vram_manager
    
    @staticmethod
    def create(checkpoint_path, gemma_root, upsampler_path, device):
        return FakeLowVRAMPipeline()
    
    def generate(self, **kwargs):
        # Write a tiny valid mp4
        output_path = kwargs["output_path"]
        Path(output_path).write_bytes(b"\x00" * 100)
    
    def warmup(self, output_path):
        pass
    
    def compile_transformer(self):
        pass
```

### 5.7 Existing Test Compatibility

All existing tests must continue to pass. Changes to verify:

| Test File | Expected Impact | Verification |
|---|---|---|
| `test_runtime_policy_decision.py` | **MUST UPDATE** — current tests assert 24GB forces API | Update assertions |
| `test_generation.py` | Should pass — uses fake pipeline | Run as-is |
| `test_health.py` | Should pass | Run as-is |
| `test_settings.py` | May need update if settings schema changes | Check after changes |
| `test_pyright.py` | Must pass — new code must be pyright-strict clean | Run after all changes |
| `test_no_mock_usage.py` | Must pass — no mocks in new tests | Verify |
| `test_import_safety.py` | Must pass — new lazy imports must be safe | Verify |

### 5.8 Manual Testing Checklist

#### GPU Testing Matrix

| GPU | VRAM | Test Scenario | Expected |
|---|---|---|---|
| RTX 4090 | 24 GB | 540p 4s t2v | ✅ Works, ~30s |
| RTX 4090 | 24 GB | 720p 4s t2v | ✅ Works, ~45s |
| RTX 4090 | 24 GB | 1080p 6s t2v | ✅ Works, ~90s |
| RTX 4080 | 16 GB | 540p 4s t2v | ✅ Works |
| RTX 4080 | 16 GB | 720p 4s t2v | ✅ Works |
| RTX 4080 | 16 GB | 1080p t2v | ⚠️ Should auto-cap or warn |
| RTX 3060 | 12 GB | 540p 2.5s t2v | ✅ Works |
| RTX 3060 | 12 GB | 720p t2v | ⚠️ May OOM, should warn |
| No GPU | 0 | Any local generation | ❌ Forced API or disabled |

#### Web Mode Testing

| # | Scenario | Steps | Expected |
|---|---|---|---|
| W1 | Start web server | Run `python backend/ltx2_server.py` with `LTX_WEB_MODE=1` | Server starts, prints URL |
| W2 | Open in browser | Navigate to `http://localhost:8000` | SPA loads, no console errors |
| W3 | Generate video | Enter prompt, click generate | Video generates and plays in browser |
| W4 | Download video | Click download on generated video | Browser downloads MP4 |
| W5 | Settings persist | Change settings, reload page | Settings retained |
| W6 | Cross-browser | Test in Chrome, Firefox, Safari | All work |
| W7 | Mobile browser | Open on phone (same network) | UI loads (responsive) |

### 5.9 Performance Benchmarks (Automated)

```python
# tests/test_performance_benchmarks.py (skip in CI, run manually)
import pytest

@pytest.mark.skip(reason="Requires GPU — run manually")
class TestPerformanceBenchmarks:
    def test_540p_generation_time(self):
        """540p 4s generation should complete in under 120 seconds."""
        start = time.time()
        # ... trigger generation ...
        elapsed = time.time() - start
        assert elapsed < 120, f"Generation took {elapsed:.1f}s (limit: 120s)"
    
    def test_peak_vram_usage_24gb(self):
        """Peak VRAM during 720p generation should stay under 22GB."""
        torch.cuda.reset_peak_memory_stats()
        # ... trigger generation ...
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        assert peak_gb < 22, f"Peak VRAM: {peak_gb:.1f}GB (limit: 22GB)"
    
    def test_offloading_overhead(self):
        """Model offloading overhead should be < 5s per generation."""
        # Compare low-VRAM pipeline time vs high-VRAM
        pass
```

### 5.10 CI Pipeline Extension

```yaml
# Additions to CI (conceptual)
jobs:
  test:
    steps:
      - pnpm typecheck          # existing
      - pnpm backend:test       # existing — must pass with new tests
      - pnpm build:frontend     # existing
      # New:
      - pnpm backend:test -- tests/test_vram_manager.py
      - pnpm backend:test -- tests/test_runtime_policy_low_vram.py
      - pnpm backend:test -- tests/test_web_routes.py
      - pnpm backend:test -- tests/test_local_only_generation.py
```

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| OOM on 12GB GPUs at higher resolutions | High | Medium | VRAM estimator prevents generation; show warning |
| `ltx-core`/`ltx-pipelines` don't support CPU model offloading cleanly | Medium | High | Use `DistilledNativePipeline` pattern which already does this |
| Sequential offloading is too slow | Low | Medium | Most overhead is in CPU→GPU transfer (~1-3s per model); acceptable for local use |
| Web mode file security (path traversal) | Medium | High | Whitelist allowed directories; reject paths outside app data |
| TilingConfig parameters don't exist in ltx-core | Medium | Medium | Check ltx-core API; fall back to defaults |
| Electron-dependent frontend code breaks in web mode | High | Medium | Thorough shim + integration test of all electron API calls |
| Audio generation fails on low VRAM (extra models) | Medium | Medium | Disable audio generation below 16GB tier |

---

## File Summary: All New & Modified Files

### New Files
| Path | Purpose |
|---|---|
| `backend/services/vram_manager/__init__.py` | VRAM tier manager |
| `backend/services/vram_manager/vram_manager.py` | Core VRAM management logic |
| `backend/services/fast_video_pipeline/ltx_low_vram_pipeline.py` | Low-VRAM pipeline with sequential offloading |
| `backend/_routes/web_routes.py` | Web-mode file/app-info endpoints |
| `backend/tests/test_vram_manager.py` | VRAM manager unit tests |
| `backend/tests/test_runtime_policy_low_vram.py` | Updated policy tests |
| `backend/tests/test_vram_profile_endpoint.py` | VRAM profile endpoint tests |
| `backend/tests/test_web_routes.py` | Web mode route tests |
| `backend/tests/test_local_only_generation.py` | No-API-key generation tests |
| `backend/tests/fakes/fake_vram_manager.py` | Fake VRAM manager for tests |
| `frontend/lib/electron-shim.ts` | ElectronAPI shim for web mode |
| `run.sh` | One-command launcher |
| `run.py` | Cross-platform Python launcher |

### Modified Files
| Path | Change |
|---|---|
| `backend/runtime_config/runtime_policy.py` | Lower VRAM threshold from 31GB to 8GB |
| `backend/handlers/pipelines_handler.py` | VRAM-aware pipeline selection |
| `backend/app_handler.py` | Wire VRAMManager into ServiceBundle |
| `backend/app_factory.py` | Static file serving for web mode |
| `backend/ltx2_server.py` | Web mode flag, bind to configurable host |
| `backend/handlers/text_handler.py` | Default to local encoding when no API key |
| `backend/state/app_settings.py` | `use_local_text_encoder` default to True |
| `backend/services/gpu_info/gpu_info.py` | Add `set_vram_gb` to protocol (for testing) |
| `backend/tests/conftest.py` | Wire fake VRAM manager |
| `backend/tests/fakes/services.py` | Add FakeGpuInfo.set_vram_gb, FakeVRAMManager |
| `backend/tests/test_runtime_policy_decision.py` | Update assertions for new thresholds |
| `frontend/lib/backend.ts` | Web-mode detection |
| `frontend/views/GenSpace.tsx` | VRAM-aware resolution/duration limits |
| `frontend/contexts/AppSettingsContext.tsx` | Remove API key gates for local mode |
| `vite.config.ts` | Web build output configuration |
| `settings.json` | Remove default API key fields; add `use_local_text_encoder: true` |
| `README.md` | Document web mode and low-VRAM support |
