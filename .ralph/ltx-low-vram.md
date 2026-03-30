
# LTX Desktop: Low-VRAM + GGUF + BlockSwap Implementation

## Goals
- Add GGUF model loading for LTX-2.3 (from unsloth/Kijai repos)
- Add block swap (ComfyUI-style) for transformer blocks — offload inactive blocks to CPU
- Implement sequential model offloading (VRAMManager)
- Lower VRAM gate from 31GB to 8GB
- Default to local-only (no API key required)
- Create web app mode (electron shim + static serving)
- Update PLAN.md with GGUF + blockswap details
- Do NOT download models — just wire the code; testing happens on server

## Checklist
- [x] Research GGUF format for LTX models and block swap technique
- [x] Update PLAN.md with GGUF + blockswap sections
- [x] Implement VRAMManager service (tier classification, resolution/frame limits)
- [x] Implement GGUF model loader service
- [x] Implement block swap mechanism for transformer
- [x] Create LTXLowVRAMPipeline with sequential offloading + GGUF + blockswap
- [x] Modify runtime_policy.py (lower VRAM gate to 8GB)
- [x] Modify pipelines_handler.py for VRAM-aware pipeline selection
- [x] Wire VRAMManager into ServiceBundle + AppHandler
- [x] Default to local text encoding (no API key needed)
- [x] Create electron-shim.ts for web mode
- [x] Create web_routes.py for web mode backend
- [x] Add static file serving to app_factory.py
- [x] Create launcher scripts (run.sh / run.py)
- [x] Create fake services for testing
- [x] Create test files
- [x] Update existing tests that assert on old VRAM thresholds
- [x] Write RESUME.md with server testing instructions

## Status: COMPLETE
All 332 backend tests passing. See RESUME.md for server testing instructions.
