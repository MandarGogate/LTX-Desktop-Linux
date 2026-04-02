# Plan: Model Performance, VRAM Optimization & Bug Fixes

## Part 1: VRAM Reduction

### 1.1 Fix double GGUF state_dict load (HIGH impact)
**File:** `backend/services/fast_video_pipeline/ltx_low_vram_pipeline.py:583-586`

The transformer state dict is loaded twice — once into `base_model` (line 583) and again into `X0Model` wrapper (line 586). For a Q8_0 model (~12GB), this means ~24GB peak RAM during loading.

**Fix:** After `base_model.load_state_dict()`, delete the state_dict and call `gc.collect()` + `empty_cache()` before constructing `X0Model`. Or reuse weights via `assign=True` on the X0Model load.

### 1.2 Use lazy GGUF loader in low-VRAM pipeline (HIGH impact)
**File:** `backend/services/fast_video_pipeline/ltx_low_vram_pipeline.py:548`

The low-VRAM pipeline still uses `GGUFModelLoader.load_gguf_sd_for_diffusers()` which dequantizes everything to bf16 at load time (~43GB float32 intermediate for Q8_0). The lazy loader (`gguf_lazy_loader.py`) already exists and keeps weights quantized, dequantizing per-layer at forward time.

**Fix:** Integrate `GGUFLazyLoader` into `_load_gguf_transformer()` so weights stay quantized in RAM/VRAM.

### 1.3 Cache LoRA matrices on GPU instead of transferring per-forward (MEDIUM impact)
**File:** `backend/services/fast_video_pipeline/ltx_optimized_pipeline.py:448-456`

LoRA A/B matrices are transferred to GPU on every forward hook call (384 transfers per generation with 48 blocks × 8 steps).

**Fix:** Pre-move LoRA matrices to GPU once during setup, reuse across all forward passes. Move back to CPU only when offloading the transformer.

### 1.4 Stream FP8 text encoder weights instead of full copy (MEDIUM impact)
**File:** `backend/services/text_encoder/ltx_text_encoder.py:90-97`

Each FP8 Linear forward creates a full bf16 copy of the weight matrix (`lin.weight.to(x.dtype)`). For a 12B model, this is ~24GB of temporary allocations per forward pass.

**Fix:** Use `torch.compile` with `mode="reduce-overhead"` on the FP8 forward, or implement a custom CUDA kernel that computes with FP8 weights directly without materializing the bf16 copy.

### 1.5 Reduce triple gc.collect() cleanup overhead (LOW impact)
**File:** `backend/services/vram_manager/vram_manager.py:299-307`

Three rounds of `gc.collect()` + `empty_cache()` adds 1-3s per phase. Called 6+ times per generation = 6-18s wasted.

**Fix:** Reduce to one `gc.collect()` + `empty_cache()` cycle. The second pass rarely frees anything. Use `gc.collect(2)` (full collection) explicitly if needed.

### 1.6 Dynamic VRAM-based resolution adjustment (MEDIUM impact)
**File:** `backend/services/vram_manager/vram_manager.py:40-58`

Resolution limits are fixed per tier. If another process is using VRAM, the pre-set resolution can OOM.

**Fix:** Query actual free VRAM via `torch.cuda.mem_get_info()` and adjust resolution dynamically rather than relying on static tier tables.

---

## Part 2: Performance Improvements

### 2.1 Adopt optimized pipeline as the default low-VRAM pipeline
**File:** `backend/handlers/pipelines_handler.py`

The optimized pipeline (`ltx_optimized_pipeline.py`) has prompt caching, lazy GGUF loading, and StateDictRegistry — but the standard low-VRAM pipeline is used by default.

**Fix:** Switch default to `LTXOptimizedPipeline` when GGUF models are available.

### 2.2 Pre-compute text encoder warmup (LOW impact)
**File:** `backend/handlers/pipelines_handler.py`

The text encoder is loaded and quantized on first use. This adds several seconds to the first generation.

**Fix:** Warm up the text encoder during pipeline initialization (like the transformer warmup in `ltx_fast_video_pipeline.py:133-153`).

### 2.3 Enable torch.compile for transformer in low-VRAM pipeline (MEDIUM impact)
**File:** `backend/services/fast_video_pipeline/ltx_fast_video_pipeline.py:155-162`

`torch.compile` is only used in the standard (high-VRAM) pipeline. Even with block swap, compiling individual transformer blocks could yield speedups.

**Fix:** Apply `torch.compile` to individual transformer blocks that are kept on GPU during block swap.

### 2.4 Batch audio decoding (LOW impact)
Audio decode happens after VAE decode. If generating multiple clips, audio decoding could be batched.

---

## Part 3: Bug Fixes

### CRITICAL Bugs

| # | File:Line | Bug | Fix |
|---|-----------|-----|-----|
| 1 | `video_generation_handler.py:118` | Missing `case _:` default in aspect ratio match — `UnboundLocalError` on unexpected input | Add `case _:` default that raises `HTTPError(400, "Invalid aspect ratio")` |
| 2 | `app_factory.py:181` | Path traversal via prefix check bypass (`/assets-extra/` passes `startswith("/assets")`) | Use `Path.is_relative_to()` or append `os.sep` to prefix |
| 3 | `web_routes.py:128` | Same prefix-check bypass in `_is_path_allowed` | Same fix as #2 |
| 4 | `use-generation.ts:338` | `generateImage` leaks polling `setInterval` on error — no `finally` block | Add `try/finally` to always call `clearInterval(progressInterval)` |
| 4a | `frontend/components/SettingsModal.tsx`, `frontend/views/editor/useGapGeneration.ts`, `backend/handlers/suggest_gap_prompt_handler.py` | Gemini API key purpose is easy to lose track of; it is only used for prompt suggestion flows, not core generation | Document this in UI/README/TODO and keep the key scoped to `/api/suggest-gap-prompt` usage only |
| 4b | `backend/services/fast_video_pipeline/ltx_low_vram_pipeline.py`, model selection/runtime config | Black-frame outputs after GGUF + split-component fallback likely indicate a model/component mismatch or wrong schedule path | Start from an official LTX checkpoint from the upstream repo as the baseline, verify non-black output there first, then reintroduce GGUF/split-component support only after matching outputs |

### HIGH Bugs

| # | File:Line | Bug | Fix |
|---|-----------|-----|-----|
| 5 | `video_generation_handler.py:215,305` | `NamedTemporaryFile` handle leaked (never closed) | Use `tempfile.mktemp()` or close handle before use |
| 6 | `video_generation_handler.py:343` | `_generate_a2v` swallows `HTTPError`, wraps as 500 | Add `except HTTPError` before generic `Exception` catch |
| 7 | `models_handler.py:292` | `has_api_key` parameter shadowed to `False` — dead code | Remove `has_api_key = False` assignment, use the parameter |
| 8 | `python-backend.ts:372` | `backendOwnership = 'managed'` set after crash (should be `null`) | Set `backendOwnership = null` on unexpected exit |
| 9 | `video_generation_handler.py:143` | TOCTOU race: `load_gpu_pipeline` then `start_generation` without atomic lock | Move both inside the generation lock |

### MEDIUM Bugs

| # | File:Line | Bug | Fix |
|---|-----------|-----|-----|
| 10 | `models_handler.py:125` | `_has_any_diffusion_model` returns `True` for any `.safetensors` (not just diffusion models) | Add filename filter like the GGUF check |
| 11 | `video-processing-handlers.ts:29` | Extracted frame temp files never cleaned up | Add cleanup on app quit or use LRU eviction |
| 12 | `python-backend.ts:417` | `pythonProcess = null` before force-kill can cause port conflicts on restart | Delay null assignment until after force-kill timeout |
| 13 | `backend.ts:37` | WebSocket URL doesn't handle `https://` → `wss://` | Add `https://` → `wss://` replacement |
| 14 | `use-generation.ts:280` | Image generation cancel leaves backend job running | Add cancel endpoint for image generation |
| 15 | `conditioning_cache.py:47` | `__del__` cleanup unreliable during interpreter shutdown | Use context manager or explicit `close()` instead |

---

## Execution Order

1. **Bug fixes first** (Part 3) — these are low-risk, high-impact
2. **VRAM reduction** (Part 1.1–1.2) — highest-impact VRAM wins
3. **VRAM reduction** (Part 1.3–1.6) — incremental improvements
4. **Performance** (Part 2.1–2.3) — after VRAM is stable

---

## Progress

### Bug Fixes

- [ ] Bug 1: Missing `case _:` default in aspect ratio match
- [ ] Bug 2: Path traversal in `app_factory.py:181`
- [ ] Bug 3: Prefix-check bypass in `web_routes.py:128`
- [ ] Bug 4: `generateImage` interval leak in `use-generation.ts:338`
- [ ] Bug 4a: Document that Gemini API key is only used for prompt suggestion flows (`/api/suggest-gap-prompt`)
- [ ] Bug 4b: Rebaseline black-frame issue against an official upstream LTX checkpoint before debugging GGUF/split fallback further
- [ ] Bug 5: `NamedTemporaryFile` handle leak in `video_generation_handler.py`
- [ ] Bug 6: `_generate_a2v` swallows `HTTPError` in `video_generation_handler.py:343`
- [ ] Bug 7: `has_api_key` shadowed in `models_handler.py:292`
- [ ] Bug 8: `backendOwnership` wrong after crash in `python-backend.ts:372`
- [ ] Bug 9: TOCTOU race in `video_generation_handler.py:143`
- [ ] Bug 10: `_has_any_diffusion_model` false positive in `models_handler.py:125`
- [ ] Bug 11: Temp file leak in `video-processing-handlers.ts`
- [ ] Bug 12: `pythonProcess = null` too early in `python-backend.ts:417`
- [ ] Bug 13: HTTPS WebSocket URL in `backend.ts:37`
- [ ] Bug 14: Image cancel leaves backend running in `use-generation.ts`
- [ ] Bug 15: `__del__` cleanup in `conditioning_cache.py:47`

### VRAM Reduction

- [ ] 1.1 Fix double GGUF state_dict load
- [ ] 1.2 Use lazy GGUF loader in low-VRAM pipeline
- [ ] 1.3 Cache LoRA matrices on GPU
- [ ] 1.4 Stream FP8 text encoder weights
- [ ] 1.5 Reduce triple gc.collect() cleanup
- [ ] 1.6 Dynamic VRAM-based resolution

### Performance Improvements

- [ ] 2.1 Adopt optimized pipeline as default
- [ ] 2.2 Pre-compute text encoder warmup
- [ ] 2.3 Enable torch.compile in low-VRAM pipeline
- [ ] 2.4 Batch audio decoding
