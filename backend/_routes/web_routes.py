"""Web-mode routes for running LTX Desktop as a standalone web app.

These routes replace Electron IPC calls with HTTP endpoints, enabling
the frontend to run in any browser without Electron.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from state.deps import get_state_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/web", tags=["web"])


# ============================================================
# Request/Response Models
# ============================================================


class FileReadRequest(BaseModel):
    path: str


class FileReadResponse(BaseModel):
    data: str  # base64 encoded
    mimeType: str


class FileSaveRequest(BaseModel):
    path: str
    content: str
    encoding: str = "utf-8"


class FileSaveResponse(BaseModel):
    success: bool
    path: str | None = None
    error: str | None = None


class AppInfoResponse(BaseModel):
    version: str
    mode: str
    modelsPath: str
    outputsPath: str
    isPackaged: bool


class GpuInfoResponse(BaseModel):
    available: bool
    name: str | None = None
    vram: int | None = None


class VRAMProfileResponse(BaseModel):
    tier: str
    vram_total_gb: int
    offload_strategy: str
    block_swap_blocks_on_gpu: int
    max_resolution_width: int
    max_resolution_height: int
    available_resolutions: dict[str, dict[str, int]]
    max_frames_540p_25fps: int
    max_frames_720p_25fps: int
    max_frames_1080p_25fps: int
    gguf_recommended: bool
    gguf_quant_level: str
    fp8_enabled: bool


# ============================================================
# Allowed paths validation
# ============================================================

# Only allow reading/writing within these directories
_ALLOWED_READ_PREFIXES: list[str] = []
_ALLOWED_WRITE_PREFIXES: list[str] = []


def configure_allowed_paths(models_dir: str, outputs_dir: str, app_data_dir: str) -> None:
    """Configure allowed file system paths (called at startup)."""
    _ALLOWED_READ_PREFIXES.clear()
    _ALLOWED_WRITE_PREFIXES.clear()
    _ALLOWED_READ_PREFIXES.extend([models_dir, outputs_dir, app_data_dir])
    _ALLOWED_WRITE_PREFIXES.extend([outputs_dir, app_data_dir])


def _is_path_allowed(path: str, prefixes: list[str]) -> bool:
    """Check if a path is within allowed directories."""
    resolved = str(Path(path).resolve())
    return any(resolved.startswith(str(Path(p).resolve())) for p in prefixes)


# ============================================================
# Routes
# ============================================================


@router.get("/app-info", response_model=AppInfoResponse)
def get_app_info(handler: Any = Depends(get_state_service)) -> AppInfoResponse:
    """Return application info (replaces Electron's getAppInfo IPC)."""
    return AppInfoResponse(
        version="1.0.0-web",
        mode="web",
        modelsPath=str(handler.config.default_models_dir),
        outputsPath=str(handler.config.outputs_dir),
        isPackaged=False,
    )


@router.get("/gpu-info", response_model=GpuInfoResponse)
def get_gpu_info(handler: Any = Depends(get_state_service)) -> GpuInfoResponse:
    """Return GPU information (replaces Electron's checkGpu IPC)."""
    gpu_info = handler.gpu_info.get_gpu_info()
    return GpuInfoResponse(
        available=handler.gpu_info.get_gpu_available(),
        name=gpu_info.get("name"),
        vram=gpu_info.get("vram"),
    )


@router.get("/vram-profile", response_model=VRAMProfileResponse)
def get_vram_profile(handler: Any = Depends(get_state_service)) -> dict[str, Any]:
    """Return VRAM tier and capability profile."""
    from services.vram_manager.vram_manager import VRAMManager

    vram_gb = handler.gpu_info.get_vram_total_gb() or 0
    manager = VRAMManager(handler.config.device, vram_gb)
    return manager.to_profile_dict()  # type: ignore[return-value]


@router.post("/file/read", response_model=FileReadResponse)
def read_file(req: FileReadRequest) -> FileReadResponse:
    """Read a file and return as base64 (replaces Electron's readLocalFile IPC)."""
    if not _is_path_allowed(req.path, _ALLOWED_READ_PREFIXES):
        return FileReadResponse(data="", mimeType="application/octet-stream")

    path = Path(req.path)
    if not path.exists() or not path.is_file():
        return FileReadResponse(data="", mimeType="application/octet-stream")

    mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")

    return FileReadResponse(data=b64, mimeType=mime_type)


@router.post("/file/save", response_model=FileSaveResponse)
def save_file(req: FileSaveRequest) -> FileSaveResponse:
    """Save content to a file (replaces Electron's saveFile IPC)."""
    if not _is_path_allowed(req.path, _ALLOWED_WRITE_PREFIXES):
        return FileSaveResponse(success=False, error="Path not allowed")

    try:
        path = Path(req.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(req.content, encoding=req.encoding)
        return FileSaveResponse(success=True, path=str(path))
    except Exception as e:
        return FileSaveResponse(success=False, error=str(e))


@router.get("/gguf-models")
def get_gguf_models(handler: Any = Depends(get_state_service)) -> dict[str, Any]:
    """List available GGUF models."""
    from services.gguf_loader.gguf_loader import GGUFModelLoader

    loader = GGUFModelLoader(handler.config.default_models_dir)
    return loader.get_gguf_info()
