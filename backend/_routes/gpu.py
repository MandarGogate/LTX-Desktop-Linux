"""Route handlers for /api/gpu — VRAM profile, model recommendations, LoRA management."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File
from pydantic import BaseModel

from _routes._errors import HTTPError
from state import get_state_service
from app_handler import AppHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/gpu", tags=["gpu"])


class VRAMProfileResponse(BaseModel):
    tier: str
    vram_total_gb: int
    offload_strategy: str
    max_resolution: str
    max_frames_1080p: int
    max_frames_720p: int
    max_frames_540p: int
    recommended_gguf_quant: str
    gguf_recommended: bool
    sage_attention_available: bool
    model_recommendations: list[dict[str, str]]


class VideoModelInfo(BaseModel):
    filename: str
    path: str
    size_mb: float
    model_type: str
    quant_level: str | None = None


class VideoModelListResponse(BaseModel):
    models: list[VideoModelInfo]


class TextEncoderVariantInfo(BaseModel):
    filename: str
    path: str
    size_mb: float
    format: str
    quant_level: str | None = None


class TextEncoderVariantListResponse(BaseModel):
    variants: list[TextEncoderVariantInfo]


class GGUFModelInfo(BaseModel):
    filename: str
    path: str
    size_mb: float
    quant_level: str


class LoRAInfo(BaseModel):
    filename: str
    path: str
    size_mb: float
    is_distilled: bool
    is_ic_lora: bool
    suggested_strength: float


class LoRAListResponse(BaseModel):
    loras: list[LoRAInfo]
    lora_dir: str


def _excluded_model_dirs(models_dir: Path) -> tuple[Path, ...]:
    return (
        models_dir / "Z-Image-Turbo",
        models_dir / "text_encoders",
        models_dir / "gemma-3-12b-it-qat-q4_0-unquantized",
        models_dir / "dpt-hybrid-midas",
        models_dir / ".cache",
        models_dir / "gguf",
        models_dir / "loras",
        models_dir / "upscale_models",
        models_dir / "vae",
    )


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _iter_video_model_files(models_dir: Path) -> list[VideoModelInfo]:
    files: list[VideoModelInfo] = []
    diffusion_dir = models_dir / "diffusion_models"
    legacy_gguf_dir = models_dir / "gguf"

    # GGUF models — search diffusion_models/ and legacy gguf/
    for search_dir in [diffusion_dir, legacy_gguf_dir]:
        if not search_dir.exists():
            continue
        for gguf_file in search_dir.rglob("*.gguf"):
            name_lower = gguf_file.name.lower()
            # Skip non-video-generation models (Z-Image, etc.)
            if "z-image" in name_lower or "zimage" in name_lower or "z_image" in name_lower:
                continue
            size_mb = gguf_file.stat().st_size / (1024 * 1024)
            quant_level = None
            for q in [
                "Q8_0",
                "Q5_1",
                "Q5_0",
                "Q4_K_M",
                "Q4_K_S",
                "Q4_1",
                "Q4_0",
                "Q3_K_M",
                "Q2_K",
            ]:
                if q in gguf_file.name:
                    quant_level = q
                    break
            files.append(
                VideoModelInfo(
                    filename=gguf_file.name,
                    path=str(gguf_file),
                    size_mb=round(size_mb, 1),
                    model_type="gguf",
                    quant_level=quant_level,
                )
            )

    # Safetensors checkpoints — search diffusion_models/ and root
    excluded_dirs = _excluded_model_dirs(models_dir)
    for search_dir in [diffusion_dir, models_dir]:
        if not search_dir.exists():
            continue
        for ckpt in search_dir.rglob("*.safetensors"):
            if search_dir == models_dir and any(_is_under(ckpt, parent) for parent in excluded_dirs):
                continue
            if search_dir == models_dir and _is_under(ckpt, diffusion_dir):
                continue
            name_lower = ckpt.name.lower()
            if (
                "upscaler" in name_lower
                or "ic-lora" in name_lower
                or "ic_lora" in name_lower
                or "lora" in name_lower
                or "_vae_" in name_lower
                or "video_vae" in name_lower
                or "audio_vae" in name_lower
                or "text_projection" in name_lower
            ):
                continue
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            files.append(
                VideoModelInfo(
                    filename=ckpt.name,
                    path=str(ckpt),
                    size_mb=round(size_mb, 1),
                    model_type="checkpoint",
                )
            )

    return sorted(files, key=lambda item: (item.model_type, item.filename.lower()))


def _infer_quant_level(filename: str) -> str | None:
    name = filename.lower()
    for quant in (
        "q8_0",
        "q6_k",
        "q5_1",
        "q5_0",
        "q4_k_m",
        "q4_k_s",
        "q4_1",
        "q4_0",
        "q3_k_m",
        "q2_k",
        "fp8",
        "fp4",
        "bf16",
    ):
        if quant in name:
            return quant.upper()
    if "fpmixed" in name:
        return "FP-MIXED"
    return None


def _iter_text_encoder_variants(models_dir: Path) -> list[TextEncoderVariantInfo]:
    variants: list[TextEncoderVariantInfo] = []
    text_encoder_dir = models_dir / "text_encoders"
    if not text_encoder_dir.exists():
        return variants

    for path in sorted(text_encoder_dir.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.suffix in {".safetensors", ".gguf"}:
            variants.append(
                TextEncoderVariantInfo(
                    filename=path.name,
                    path=str(path),
                    size_mb=round(path.stat().st_size / (1024 * 1024), 1),
                    format=path.suffix.lstrip("."),
                    quant_level=_infer_quant_level(path.name),
                )
            )
        elif path.is_dir() and any(path.iterdir()):
            variants.append(
                TextEncoderVariantInfo(
                    filename=path.name,
                    path=str(path),
                    size_mb=round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024), 1),
                    format="folder",
                    quant_level=_infer_quant_level(path.name),
                )
            )

    return variants


def _iter_lora_files(models_dir: Path) -> list[LoRAInfo]:
    lora_dir = models_dir / "loras"
    lora_dir.mkdir(exist_ok=True)
    loras: list[LoRAInfo] = []
    seen: set[str] = set()

    search_dirs = [lora_dir, models_dir]
    excluded_dirs = _excluded_model_dirs(models_dir)

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for file_path in search_dir.rglob("*.safetensors"):
            if file_path.name in seen:
                continue
            if any(
                _is_under(file_path, parent)
                for parent in excluded_dirs
                if parent != lora_dir
            ):
                continue
            name_lower = file_path.name.lower()
            if (
                "upscaler" in name_lower
                or "ic-lora" in name_lower
                or "ic_lora" in name_lower
                or "_vae_" in name_lower
                or "video_vae" in name_lower
                or "audio_vae" in name_lower
            ):
                continue
            size_mb = file_path.stat().st_size / (1024 * 1024)
            # Most LoRAs are much smaller than full checkpoints. Also always include files in models/loras.
            if not _is_under(file_path, lora_dir) and (
                "lora" not in name_lower and size_mb > 5_000
            ):
                continue
            seen.add(file_path.name)
            is_distilled = "distilled" in name_lower
            suggested = 0.6 if is_distilled else 0.8
            loras.append(
                LoRAInfo(
                    filename=file_path.name,
                    path=str(file_path),
                    size_mb=round(size_mb, 1),
                    is_distilled=is_distilled,
                    is_ic_lora=False,
                    suggested_strength=suggested,
                )
            )

    return sorted(loras, key=lambda item: item.filename.lower())


@router.get("/vram-profile", response_model=VRAMProfileResponse)
def route_vram_profile(
    handler: AppHandler = Depends(get_state_service),
) -> VRAMProfileResponse:
    """Get GPU VRAM tier, recommended settings, and model suggestions."""
    import torch

    from services.vram_manager.vram_manager import VRAMManager

    settings = handler.settings.get_settings_snapshot()
    user_blocks = settings.num_blocks_to_swap
    user_run_mode = settings.run_mode

    if not torch.cuda.is_available():
        return VRAMProfileResponse(
            tier="cpu_only",
            vram_total_gb=0,
            offload_strategy="none",
            max_resolution="480p",
            max_frames_1080p=0,
            max_frames_720p=0,
            max_frames_540p=9,
            recommended_gguf_quant="Q4_0",
            gguf_recommended=True,
            sage_attention_available=False,
            model_recommendations=[],
        )

    # Use ceiling division so 24564 MB → 24 GB (not 23)
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = int((total_bytes + (1024**3 - 1)) // (1024**3))
    mgr = VRAMManager(
        torch.device("cuda"),
        vram_gb,
        user_blocks_on_gpu=user_blocks,
        user_run_mode=user_run_mode,
    )

    # Check SageAttention
    sage_available = False
    try:
        from sageattention import sageattn  # type: ignore[reportMissingImports]

        sage_available = True
    except ImportError:
        pass

    # Build recommendations based on VRAM
    recs: list[dict[str, str]] = []
    if vram_gb >= 24:
        recs.append(
            {
                "model": "Distilled (FP8)",
                "desc": "Fastest — 8 steps, full quality",
                "recommended": "true",
            }
        )
        recs.append(
            {
                "model": "Distilled GGUF Q4_0",
                "desc": "Lighter weights, ~same quality",
                "recommended": "false",
            }
        )
        recs.append(
            {
                "model": "Dev GGUF Q4_K_M + Distilled LoRA",
                "desc": "Best quality — 8 steps with LoRA",
                "recommended": "false",
            }
        )
    elif vram_gb >= 16:
        recs.append(
            {
                "model": "Distilled GGUF Q4_0",
                "desc": "Best for 16 GB — fast 8-step generation",
                "recommended": "true",
            }
        )
        recs.append(
            {
                "model": "Dev GGUF Q4_K_M + Distilled LoRA",
                "desc": "Higher quality — 8 steps with LoRA",
                "recommended": "false",
            }
        )
    elif vram_gb >= 12:
        recs.append(
            {
                "model": "Distilled GGUF Q4_0",
                "desc": "Recommended for 12 GB",
                "recommended": "true",
            }
        )
    else:
        recs.append(
            {
                "model": "Distilled GGUF Q4_0",
                "desc": "Only option for <12 GB",
                "recommended": "true",
            }
        )

    max_w, max_h = mgr.get_max_resolution()
    max_res = (
        "1080p"
        if max_w >= 1920
        else "720p"
        if max_w >= 1280
        else "540p"
        if max_w >= 960
        else "480p"
    )

    return VRAMProfileResponse(
        tier=mgr.tier.value,
        vram_total_gb=vram_gb,
        offload_strategy=mgr.offload_strategy.value,
        max_resolution=max_res,
        max_frames_1080p=mgr.get_max_frames(1920, 1088, 25),
        max_frames_720p=mgr.get_max_frames(1280, 704, 25),
        max_frames_540p=mgr.get_max_frames(960, 544, 25),
        recommended_gguf_quant=mgr.get_recommended_gguf_quant(),
        gguf_recommended=mgr.should_use_gguf(),
        sage_attention_available=sage_available,
        model_recommendations=recs,
    )


@router.get("/video-models", response_model=VideoModelListResponse)
def route_list_video_models(
    handler: AppHandler = Depends(get_state_service),
) -> VideoModelListResponse:
    models_dir = handler.models.models_dir
    return VideoModelListResponse(models=_iter_video_model_files(models_dir))


@router.get("/text-encoders", response_model=TextEncoderVariantListResponse)
def route_list_text_encoders(
    handler: AppHandler = Depends(get_state_service),
) -> TextEncoderVariantListResponse:
    models_dir = handler.models.models_dir
    return TextEncoderVariantListResponse(variants=_iter_text_encoder_variants(models_dir))


@router.get("/gguf-models")
def route_gguf_models(
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, object]:
    """List available GGUF models."""
    from services.gguf_loader.gguf_loader import GGUFModelLoader

    models_dir = handler.models.models_dir
    loader = GGUFModelLoader(models_dir)
    return loader.get_gguf_info()


@router.get("/loras", response_model=LoRAListResponse)
def route_list_loras(
    handler: AppHandler = Depends(get_state_service),
) -> LoRAListResponse:
    """List available LoRA files in the models directory."""
    models_dir = handler.models.models_dir
    lora_dir = models_dir / "loras"
    lora_dir.mkdir(exist_ok=True)
    return LoRAListResponse(loras=_iter_lora_files(models_dir), lora_dir=str(lora_dir))


@router.post("/loras/upload")
async def route_upload_lora(
    file: UploadFile = File(...),
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, str]:
    """Upload a custom LoRA safetensors file."""
    if not file.filename or not file.filename.endswith(".safetensors"):
        raise HTTPError(400, "Only .safetensors files are supported")

    models_dir = handler.models.models_dir
    lora_dir = models_dir / "loras"
    lora_dir.mkdir(exist_ok=True)

    dest = lora_dir / file.filename
    content = await file.read()
    dest.write_bytes(content)

    logger.info(
        "Uploaded LoRA: %s (%d MB)", file.filename, len(content) // (1024 * 1024)
    )
    return {"status": "ok", "path": str(dest), "filename": file.filename}


class GpuStatsResponse(BaseModel):
    gpu_name: str
    vram_used_mb: int
    vram_total_mb: int
    gpu_utilization: int
    temperature: int


class ExternalModelInfo(BaseModel):
    id: str
    filename: str
    repo_id: str
    description: str
    size_gb: float
    quant_level: str | None = None
    model_type: str  # "gguf" | "lora" | "checkpoint" | "text_encoder"


class ExternalModelsResponse(BaseModel):
    models: list[ExternalModelInfo]


class ExternalModelDownloadRequest(BaseModel):
    repo_id: str
    filename: str | None = None
    is_folder: bool = False
    target_subdir: str = ""  # relative to models dir, e.g. "gguf" or ""


class ExternalModelDownloadResponse(BaseModel):
    status: str
    session_id: str | None = None
    message: str | None = None


# Available models from HuggingFace for download
_EXTERNAL_GGUF_MODELS: list[ExternalModelInfo] = [
    ExternalModelInfo(
        id="gguf-q8_0",
        filename="ltx-2.3-22b-dev-Q8_0.gguf",
        repo_id="unsloth/LTX-2.3-GGUF",
        description="Highest quality GGUF quantization (~22.8 GB). Best for 24+ GB GPUs.",
        size_gb=22.8,
        quant_level="Q8_0",
        model_type="gguf",
    ),
    ExternalModelInfo(
        id="gguf-q5_1",
        filename="ltx-2.3-22b-dev-Q5_1.gguf",
        repo_id="unsloth/LTX-2.3-GGUF",
        description="Good quality GGUF quantization (~16.3 GB). Balanced for 16 GB GPUs.",
        size_gb=16.3,
        quant_level="Q5_1",
        model_type="gguf",
    ),
    ExternalModelInfo(
        id="gguf-q4_k_m",
        filename="ltx-2.3-22b-dev-Q4_K_M.gguf",
        repo_id="unsloth/LTX-2.3-GGUF",
        description="Medium quality GGUF quantization (~14.3 GB). Good for 12 GB GPUs.",
        size_gb=14.3,
        quant_level="Q4_K_M",
        model_type="gguf",
    ),
    ExternalModelInfo(
        id="gguf-q4_0",
        filename="ltx-2.3-22b-dev-Q4_0.gguf",
        repo_id="unsloth/LTX-2.3-GGUF",
        description="Smallest GGUF quantization (~12.7 GB). For 8 GB GPUs.",
        size_gb=12.7,
        quant_level="Q4_0",
        model_type="gguf",
    ),
]

_EXTERNAL_LORA_MODELS: list[ExternalModelInfo] = [
    ExternalModelInfo(
        id="distilled-lora",
        filename="ltx-2-19b-distilled-lora-384.safetensors",
        repo_id="Lightricks/LTX-2",
        description="Distilled LoRA — enables fast 8-step generation with the dev base model.",
        size_gb=0.4,
        model_type="lora",
    ),
]

_EXTERNAL_CHECKPOINT_MODELS: list[ExternalModelInfo] = [
    ExternalModelInfo(
        id="ltx-2.3-distilled",
        filename="ltx-2.3-22b-distilled.safetensors",
        repo_id="Lightricks/LTX-2.3",
        description="Full distilled checkpoint (~43 GB). Requires high VRAM or will use FP8/block-swap.",
        size_gb=43.0,
        model_type="checkpoint",
    ),
    ExternalModelInfo(
        id="ltx-2.3-upsampler",
        filename="ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
        repo_id="Lightricks/LTX-2.3",
        description="2x spatial upscaler (~1.9 GB).",
        size_gb=1.9,
        model_type="upscaler",
    ),
]

_EXTERNAL_ZIT_GGUF_MODELS: list[ExternalModelInfo] = [
    ExternalModelInfo(
        id="zit-gguf-bf16",
        filename="z-image-turbo-BF16.gguf",
        repo_id="unsloth/Z-Image-Turbo-GGUF",
        description="Z-Image Turbo BF16 GGUF (~12.3 GB). Full precision, for high VRAM.",
        size_gb=12.3,
        quant_level="BF16",
        model_type="gguf",
    ),
    ExternalModelInfo(
        id="zit-gguf-q8_0",
        filename="z-image-turbo-Q8_0.gguf",
        repo_id="unsloth/Z-Image-Turbo-GGUF",
        description="Z-Image Turbo Q8_0 GGUF (~7.2 GB). Good quality.",
        size_gb=7.2,
        quant_level="Q8_0",
        model_type="gguf",
    ),
    ExternalModelInfo(
        id="zit-gguf-q4_0",
        filename="z-image-turbo-Q4_0.gguf",
        repo_id="unsloth/Z-Image-Turbo-GGUF",
        description="Z-Image Turbo Q4_0 GGUF (~4.6 GB). Smallest, for low VRAM.",
        size_gb=4.6,
        quant_level="Q4_0",
        model_type="gguf",
    ),
]

_EXTERNAL_TEXT_ENCODER_MODELS: list[ExternalModelInfo] = [
    ExternalModelInfo(
        id="te-bf16-comfy",
        filename="split_files/text_encoders/gemma_3_12B_it.safetensors",
        repo_id="Comfy-Org/ltx-2",
        description="BF16 Gemma text encoder (~22.7 GB). Full precision.",
        size_gb=22.71,
        quant_level="BF16",
        model_type="text_encoder",
    ),
    ExternalModelInfo(
        id="te-fp8-comfy",
        filename="split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors",
        repo_id="Comfy-Org/ltx-2",
        description="FP8 Gemma text encoder (~12.3 GB). Faster load, lower VRAM.",
        size_gb=12.30,
        quant_level="FP8",
        model_type="text_encoder",
    ),
    ExternalModelInfo(
        id="te-fp4-comfy",
        filename="split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
        repo_id="Comfy-Org/ltx-2",
        description="FP4 mixed Gemma text encoder (~8.8 GB). Lowest VRAM option.",
        size_gb=8.80,
        quant_level="FP4",
        model_type="text_encoder",
    ),
    ExternalModelInfo(
        id="te-fpmixed-comfy",
        filename="split_files/text_encoders/gemma_3_12B_it_fpmixed.safetensors",
        repo_id="Comfy-Org/ltx-2",
        description="FP mixed Gemma text encoder (~12.8 GB). Balanced option.",
        size_gb=12.77,
        quant_level="FP-MIXED",
        model_type="text_encoder",
    ),
    ExternalModelInfo(
        id="te-text-projection",
        filename="text_encoders/ltx-2.3_text_projection_bf16.safetensors",
        repo_id="Kijai/LTX2.3_comfy",
        description="Text projection weights used by some ComfyUI LTX 2.3 setups.",
        size_gb=2.15,
        quant_level="BF16",
        model_type="text_encoder",
    ),
]


@router.get("/external-models", response_model=ExternalModelsResponse)
def route_external_models(
    handler: AppHandler = Depends(get_state_service),
) -> ExternalModelsResponse:
    """List available models that can be downloaded from HuggingFace."""
    models_dir = handler.models.models_dir
    all_models = (
        _EXTERNAL_GGUF_MODELS + _EXTERNAL_LORA_MODELS + _EXTERNAL_CHECKPOINT_MODELS
        + _EXTERNAL_ZIT_GGUF_MODELS + _EXTERNAL_TEXT_ENCODER_MODELS
    )

    # Check which ones already exist
    result: list[ExternalModelInfo] = []
    for model in all_models:
        if model.model_type == "gguf":
            target = models_dir / "diffusion_models" / model.filename
        elif model.model_type == "lora":
            target = models_dir / "loras" / model.filename
        elif model.model_type == "text_encoder":
            target = models_dir / "text_encoders" / Path(model.filename).name
        elif model.model_type == "checkpoint":
            target = models_dir / "diffusion_models" / model.filename
        elif model.model_type == "upscaler":
            target = models_dir / "upscale_models" / model.filename
        else:
            target = models_dir / model.filename
        # Add downloaded status via description suffix
        if target.exists():
            model = model.model_copy(
                update={"description": model.description + " [DOWNLOADED]"}
            )
        result.append(model)

    return ExternalModelsResponse(models=result)


@router.post("/download-external-model", response_model=ExternalModelDownloadResponse)
def route_download_external_model(
    req: ExternalModelDownloadRequest,
    handler: AppHandler = Depends(get_state_service),
) -> ExternalModelDownloadResponse:
    """Download a model from HuggingFace to the local models directory."""
    if handler.downloads.is_download_running():
        raise HTTPError(409, "Download already in progress")

    models_dir = handler.models.models_dir

    # Determine target directory
    if req.target_subdir:
        target_dir = models_dir / req.target_subdir
    else:
        target_dir = models_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if already exists
    if req.filename and not req.is_folder:
        target_file = target_dir / Path(req.filename).name
        if target_file.exists():
            return ExternalModelDownloadResponse(
                status="already_downloaded",
                message=f"{Path(req.filename).name} already exists",
            )

    from state.app_state_types import ModelFileType

    # Use a synthetic model type for tracking
    session_id = handler.downloads.start_download({"checkpoint"})  # placeholder type

    def worker() -> None:
        try:
            handler.downloads.start_file("checkpoint", req.filename or req.repo_id)
            progress_cb = handler.downloads._make_progress_callback("checkpoint")

            if req.is_folder:
                handler.model_downloader.download_snapshot(
                    repo_id=req.repo_id,
                    local_dir=str(
                        target_dir / (req.filename or req.repo_id.split("/")[-1])
                    ),
                    on_progress=progress_cb,
                )
            else:
                downloaded_path = handler.model_downloader.download_file(
                    repo_id=req.repo_id,
                    filename=req.filename or "",
                    local_dir=str(target_dir),
                    on_progress=progress_cb,
                )
                # Flatten nested HF paths into the selected target dir.
                final_target = target_dir / Path(req.filename or downloaded_path.name).name
                if downloaded_path != final_target:
                    final_target.parent.mkdir(parents=True, exist_ok=True)
                    if final_target.exists():
                        final_target.unlink()
                    downloaded_path.rename(final_target)
                    # Best-effort cleanup of now-empty intermediate dirs.
                    try:
                        parent = downloaded_path.parent
                        while parent != target_dir and parent.exists():
                            parent.rmdir()
                            parent = parent.parent
                    except OSError:
                        pass
            handler.downloads.finish_download()
            handler.models.refresh_available_files()
        except Exception:
            handler.downloads.cleanup_downloading_dir()
            raise

    handler.task_runner.run_background(
        worker,
        task_name="external-model-download",
        on_error=handler.downloads._on_background_download_error,
        daemon=True,
    )

    return ExternalModelDownloadResponse(
        status="started",
        session_id=session_id,
        message=f"Downloading {req.filename or req.repo_id}",
    )


@router.get("/stats", response_model=GpuStatsResponse)
def route_gpu_stats() -> GpuStatsResponse:
    """Live GPU stats for the topbar widget."""
    import torch

    if not torch.cuda.is_available():
        return GpuStatsResponse(
            gpu_name="CPU",
            vram_used_mb=0,
            vram_total_mb=0,
            gpu_utilization=0,
            temperature=0,
        )

    props = torch.cuda.get_device_properties(0)
    name = props.name
    total_mb = int(props.total_memory // (1024 * 1024))

    # Get used VRAM from nvidia-smi via pynvml for accurate readings
    used_mb = 0
    utilization = 0
    temp = 0
    try:
        import pynvml  # type: ignore[reportMissingImports]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mb = int(mem_info.used // (1024 * 1024))
        total_mb = int(mem_info.total // (1024 * 1024))
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utilization = int(util.gpu)
        except Exception:
            pass
        try:
            temp = int(
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            )
        except Exception:
            pass
    except Exception:
        used_mb = int(torch.cuda.memory_allocated(0) // (1024 * 1024))

    return GpuStatsResponse(
        gpu_name=name,
        vram_used_mb=used_mb,
        vram_total_mb=total_mb,
        gpu_utilization=utilization,
        temperature=temp,
    )
