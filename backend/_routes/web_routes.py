"""Web-mode routes for running LTX Desktop as a standalone web app.

These routes replace Electron IPC calls with HTTP endpoints, enabling
the frontend to run in any browser without Electron.
"""

from __future__ import annotations

import base64
import mimetypes
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel

from state.app_settings import UpdateSettingsRequest
from state.deps import get_state_service

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


class FileUploadResponse(BaseModel):
    path: str
    url: str


class ProjectAssetsPathRequest(BaseModel):
    path: str


class ProjectAssetsPathResponse(BaseModel):
    path: str


class ProjectAssetsCopyRequest(BaseModel):
    srcPath: str
    projectId: str


class ProjectAssetsCopyResponse(BaseModel):
    success: bool
    path: str | None = None
    url: str | None = None
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


class WebLogsResponse(BaseModel):
    logPath: str
    lines: list[str]
    error: str | None = None


# ============================================================
# Allowed paths validation
# ============================================================

# Only allow reading/writing within these directories
_ALLOWED_READ_PREFIXES: list[str] = []
_ALLOWED_WRITE_PREFIXES: list[str] = []


def configure_allowed_paths(
    models_dir: str, outputs_dir: str, app_data_dir: str
) -> None:
    """Configure allowed file system paths (called at startup)."""
    _ALLOWED_READ_PREFIXES.clear()
    _ALLOWED_WRITE_PREFIXES.clear()
    _ALLOWED_READ_PREFIXES.extend([models_dir, outputs_dir, app_data_dir])
    _ALLOWED_WRITE_PREFIXES.extend([outputs_dir, app_data_dir])


def _is_path_allowed(path: str, prefixes: list[str]) -> bool:
    """Check if a path is within allowed directories."""
    resolved = Path(path).resolve()
    return any(resolved.is_relative_to(Path(p).resolve()) for p in prefixes)


def _get_default_project_assets_dir(handler: Any) -> Path:
    return handler.config.outputs_dir.parent / "project-assets"


def _get_project_assets_dir(handler: Any) -> Path:
    configured = handler.settings.get_settings_snapshot().project_assets_dir.strip()
    target = (
        Path(configured) if configured else _get_default_project_assets_dir(handler)
    )
    target.mkdir(parents=True, exist_ok=True)
    return target.resolve()


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


@router.get("/logs", response_model=WebLogsResponse)
def get_logs(limit: int = 200) -> WebLogsResponse:
    """Return recent backend/server logs for the web-mode log viewer."""
    from web_log_buffer import get_recent_lines

    safe_limit = max(1, min(limit, 2000))
    return WebLogsResponse(
        logPath="server://in-memory",
        lines=get_recent_lines(safe_limit),
    )


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


@router.post("/file/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    kind: str = Form("generic"),
    handler: Any = Depends(get_state_service),
) -> FileUploadResponse:
    """Persist a browser-uploaded file under the app data dir and return a serveable URL."""
    safe_kind = (
        "".join(ch for ch in kind.lower() if ch.isalnum() or ch in {"-", "_"})
        or "generic"
    )
    upload_dir = handler.config.outputs_dir.parent / "uploads" / safe_kind
    upload_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(file.filename or "upload.bin").name
    dest = upload_dir / f"{uuid.uuid4().hex[:8]}_{original_name}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    relative = dest.relative_to(handler.config.outputs_dir.parent).as_posix()
    return FileUploadResponse(path=str(dest), url=f"/serve/files/{relative}")


@router.get("/project-assets/path", response_model=ProjectAssetsPathResponse)
def get_project_assets_path(
    handler: Any = Depends(get_state_service),
) -> ProjectAssetsPathResponse:
    return ProjectAssetsPathResponse(path=str(_get_project_assets_dir(handler)))


@router.post("/project-assets/path", response_model=ProjectAssetsPathResponse)
def set_project_assets_path(
    req: ProjectAssetsPathRequest,
    handler: Any = Depends(get_state_service),
) -> ProjectAssetsPathResponse:
    target = Path(req.path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    handler.settings.update_settings(
        UpdateSettingsRequest(projectAssetsDir=str(target))
    )
    return ProjectAssetsPathResponse(path=str(target))


@router.post("/project-assets/copy", response_model=ProjectAssetsCopyResponse)
def copy_to_project_assets(
    req: ProjectAssetsCopyRequest,
    handler: Any = Depends(get_state_service),
) -> ProjectAssetsCopyResponse:
    if not req.srcPath or not req.projectId:
        return ProjectAssetsCopyResponse(
            success=False, error="Missing srcPath or projectId"
        )
    if not _is_path_allowed(req.srcPath, _ALLOWED_READ_PREFIXES):
        return ProjectAssetsCopyResponse(success=False, error="Path not allowed")

    try:
        src = Path(req.srcPath).resolve()
        if not src.exists() or not src.is_file():
            return ProjectAssetsCopyResponse(
                success=False, error="Source file not found"
            )

        assets_root = _get_project_assets_dir(handler)
        dest_dir = (assets_root / req.projectId).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        shutil.copyfile(src, dest)
        return ProjectAssetsCopyResponse(
            success=True,
            path=str(dest),
            url=f"/serve/project-assets/{req.projectId}/{dest.name}",
        )
    except Exception as exc:
        return ProjectAssetsCopyResponse(success=False, error=str(exc))


@router.get("/gguf-models")
def get_gguf_models(handler: Any = Depends(get_state_service)) -> dict[str, Any]:
    """List available GGUF models."""
    from services.gguf_loader.gguf_loader import GGUFModelLoader

    loader = GGUFModelLoader(handler.config.default_models_dir)
    return loader.get_gguf_info()


class ExtractFrameRequest(BaseModel):
    video_path: str
    time_seconds: float
    width: int = 512


class ExtractFrameResponse(BaseModel):
    frame: str  # base64 encoded image


@router.post("/extract-frame", response_model=ExtractFrameResponse)
def extract_frame(req: ExtractFrameRequest) -> ExtractFrameResponse:
    """Extract a frame from a video at the given timestamp."""
    import cv2
    import tempfile

    if not _is_path_allowed(req.video_path, _ALLOWED_READ_PREFIXES):
        return ExtractFrameResponse(frame="")

    cap = cv2.VideoCapture(req.video_path)
    if not cap.isOpened():
        return ExtractFrameResponse(frame="")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(req.time_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return ExtractFrameResponse(frame="")

    h, w = frame.shape[:2]
    new_height = int(h * (req.width / w))
    frame_resized = cv2.resize(
        frame, (req.width, new_height), interpolation=cv2.INTER_AREA
    )

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
        with open(tmp.name, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        Path(tmp.name).unlink(missing_ok=True)

    return ExtractFrameResponse(frame=f"data:image/jpeg;base64,{b64}")
