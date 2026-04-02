"""FastAPI composition root for the LTX backend server."""

import os
import sys
from typing import Any, cast

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

if os.environ.get("BACKEND_DEBUG") == "1":
    try:
        import debugpy  # type: ignore[reportMissingImports]

        if not bool(debugpy.is_client_connected()):  # type: ignore[reportUnknownMemberType]
            try:
                # Connect to an already-listening IDE debugger (compound launch)
                debugpy.connect(("127.0.0.1", 5678))  # type: ignore[reportUnknownMemberType]
            except (ConnectionRefusedError, ConnectionError, OSError):
                # IDE not listening — start a debug server for manual attach
                debugpy.listen(("127.0.0.1", 5678))  # type: ignore[reportUnknownMemberType]
    except (ImportError, RuntimeError) as exc:
        print(f"Debugpy setup failed: {exc}", file=sys.stderr)

import logging
from pathlib import Path
import threading
import ast

# Note: expandable_segments is not supported on all platforms

import torch
from state.app_settings import AppSettings

# ============================================================
# Logging Configuration
# ============================================================

import platform

# Backend logs to console only — Electron captures stdout/stderr and writes
# them to the session log file. This ensures *all* output (including early
# import errors and unhandled tracebacks) reaches the log, not just messages
# that go through Python's logging module.
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO, handlers=[console_handler])
logger = logging.getLogger(__name__)


def _is_known_harmless_uninitialized_name(name: object) -> bool:
    if not isinstance(name, str):
        return False
    return (
        name == "vision_tower"
        or name.startswith("vision_model.")
        or name.startswith("model.vision_tower.")
        or name.startswith("model.model.vision_tower.")
        or ".vision_tower." in name
    )


class _ExpectedUninitializedWeightsFilter(logging.Filter):
    """Suppress only known-harmless missing-weight warnings.

    Some text-only Gemma variants intentionally omit multimodal `vision_tower`
    weights. Those should not spam the logs, but transformer-wide missing
    weights must remain visible because they indicate a bad checkpoint/model
    pairing.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "ltx_core.loader.single_gpu_model_builder":
            return True

        message = record.getMessage()
        prefix = "Uninitialized parameters or buffers: "
        if not message.startswith(prefix):
            return True

        try:
            missing = ast.literal_eval(message[len(prefix):])
        except Exception:
            return True

        if not isinstance(missing, list) or not missing:
            return True

        return not all(_is_known_harmless_uninitialized_name(name) for name in missing)


logging.getLogger("ltx_core.loader.single_gpu_model_builder").addFilter(
    _ExpectedUninitializedWeightsFilter()
)

# ============================================================
# SageAttention Integration
# ============================================================
use_sage_attention = os.environ.get("USE_SAGE_ATTENTION", "1") == "1"
_sageattention_runtime_fallback_logged = False

if use_sage_attention:
    try:
        from sageattention import sageattn  # type: ignore[reportMissingImports]
        import torch.nn.functional as F

        _original_sdpa = F.scaled_dot_product_attention

        _SAGE_SUPPORTED_HEADDIMS = {64, 96, 128}

        def patched_sdpa(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            scale: float | None = None,
            **kwargs: Any,
        ) -> torch.Tensor:
            global _sageattention_runtime_fallback_logged
            try:
                use_sdpa = False
                if (
                    query.dim() != 4
                    or attn_mask is not None
                    or dropout_p != 0.0
                    or query.shape[-1] not in _SAGE_SUPPORTED_HEADDIMS
                    or not (query.is_cuda and key.is_cuda and value.is_cuda)
                    or query.dtype != key.dtype
                    or key.dtype != value.dtype
                ):
                    use_sdpa = True

                if not use_sdpa:
                    return cast(
                        torch.Tensor,
                        sageattn(
                            query, key, value, is_causal=is_causal, tensor_layout="HND"
                        ),
                    )  # type: ignore[reportUnnecessaryCast]

                return _original_sdpa(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    **kwargs,
                )
            except Exception:
                if not _sageattention_runtime_fallback_logged:
                    logger.warning(
                        "SageAttention failed during runtime; falling back to default attention",
                        exc_info=True,
                    )
                    _sageattention_runtime_fallback_logged = True
                # Cast to common dtype to avoid dtype mismatch errors
                common_dtype = query.dtype
                return _original_sdpa(
                    query,
                    key.to(common_dtype),
                    value.to(common_dtype),
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    **kwargs,
                )

        F.scaled_dot_product_attention = patched_sdpa
        logger.info("SageAttention enabled - attention operations will be faster")
    except ImportError:
        logger.warning("SageAttention not installed - using default attention")
        use_sage_attention = False
    except Exception:
        logger.warning("Failed to enable SageAttention", exc_info=True)
        use_sage_attention = False

# ============================================================
# Constants & Paths
# ============================================================

PORT = int(os.environ.get("LTX_PORT", "8000"))


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _get_device()
DTYPE = torch.bfloat16


def _resolve_app_data_dir() -> Path:
    env_path = os.environ.get("LTX_APP_DATA_DIR")
    if not env_path:
        raise RuntimeError(
            "LTX_APP_DATA_DIR environment variable must be set. "
            "When running standalone, set it to the desired data directory."
        )
    candidate = Path(env_path)
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


APP_DATA_DIR = _resolve_app_data_dir()

DEFAULT_MODELS_DIR = APP_DATA_DIR / "models"
DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = APP_DATA_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Models directory layout migration
# ============================================================

def _migrate_models_layout(models_dir: Path) -> None:
    """Migrate flat models directory to organized subdirectory layout.

    Moves files into: diffusion_models/, text_encoders/, loras/, upscale_models/
    Runs once; subsequent calls are no-ops because source files no longer exist.
    """
    import shutil

    migrations: list[tuple[Path, Path]] = [
        # Diffusion checkpoints (safetensors) from root → diffusion_models/
        (models_dir / "ltx-2.3-22b-distilled.safetensors",
         models_dir / "diffusion_models" / "ltx-2.3-22b-distilled.safetensors"),
        # Upscaler from root → upscale_models/
        (models_dir / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
         models_dir / "upscale_models" / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
        # IC-LoRA from root → loras/
        (models_dir / "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
         models_dir / "loras" / "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"),
        # Default text encoder folder from root → text_encoders/
        (models_dir / "gemma-3-12b-it-qat-q4_0-unquantized",
         models_dir / "text_encoders" / "gemma-3-12b-it-qat-q4_0-unquantized"),
    ]

    # Migrate GGUF files from gguf/ → diffusion_models/
    legacy_gguf_dir = models_dir / "gguf"
    if legacy_gguf_dir.exists():
        for gguf_file in legacy_gguf_dir.rglob("*.gguf"):
            # Flatten nested dirs (e.g. gguf/distilled/file.gguf → diffusion_models/file.gguf)
            dest = models_dir / "diffusion_models" / gguf_file.name
            if not dest.exists():
                migrations.append((gguf_file, dest))

    moved_count = 0
    for src, dst in migrations:
        if not src.exists():
            continue
        if dst.exists():
            logger.info("Migration skip (target exists): %s → %s", src, dst)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if src.is_dir():
                shutil.move(str(src), str(dst))
            else:
                src.rename(dst)
            moved_count += 1
            logger.info("Migrated model: %s → %s", src, dst)
        except Exception:
            logger.warning("Failed to migrate %s → %s", src, dst, exc_info=True)

    # Clean up empty legacy gguf/ dir
    if legacy_gguf_dir.exists():
        try:
            _remove_empty_dirs(legacy_gguf_dir)
        except Exception:
            pass

    if moved_count > 0:
        logger.info("Models layout migration complete: moved %d item(s)", moved_count)

    # Migrate saved settings paths that reference old locations
    _migrate_settings_paths(models_dir)


def _migrate_settings_paths(models_dir: Path) -> None:
    """Rewrite saved settings that reference old model paths."""
    import json

    settings_file = models_dir.parent / "settings.json"
    if not settings_file.exists():
        return

    try:
        data = json.loads(settings_file.read_text())
    except Exception:
        return

    changed = False
    models_str = str(models_dir)

    # Rewrite preferred_model_path: gguf/... → diffusion_models/..., root .safetensors → diffusion_models/
    for key in ("preferred_model_path", "preferredModelPath"):
        val = data.get(key, "")
        if not val:
            continue
        p = Path(val)
        # Absolute path under models_dir
        if str(p).startswith(models_str):
            rel = p.relative_to(models_dir)
            parts = rel.parts
            if parts and parts[0] == "gguf" and p.suffix == ".gguf":
                new_path = models_dir / "diffusion_models" / p.name
                if new_path.exists():
                    data[key] = str(new_path)
                    changed = True
            elif p.suffix == ".safetensors" and parts and parts[0] not in (
                "diffusion_models", "loras", "upscale_models", "text_encoders",
            ):
                # Root-level checkpoint moved to diffusion_models/
                new_path = models_dir / "diffusion_models" / p.name
                if new_path.exists():
                    data[key] = str(new_path)
                    changed = True

    # Rewrite preferred_text_encoder_path (relative to models_dir)
    for key in ("preferred_text_encoder_path", "preferredTextEncoderPath"):
        val = data.get(key, "")
        if not val:
            continue
        # Already under text_encoders/? Skip.
        if val.startswith("text_encoders/"):
            continue
        # Bare filename → prefix with text_encoders/
        if "/" not in val:
            new_val = f"text_encoders/{val}"
            if (models_dir / new_val).exists():
                data[key] = new_val
                changed = True

    # Rewrite preferred_zit_model_path: gguf/... → diffusion_models/...
    for key in ("preferred_zit_model_path", "preferredZitModelPath"):
        val = data.get(key, "")
        if not val:
            continue
        if val.startswith("diffusion_models/"):
            continue
        # Relative path starting with gguf/
        if val.startswith("gguf/"):
            filename = Path(val).name
            new_val = f"diffusion_models/{filename}"
            if (models_dir / new_val).exists():
                data[key] = new_val
                changed = True

    if changed:
        try:
            settings_file.write_text(json.dumps(data, indent=4))
            logger.info("Migrated settings paths to new models layout")
        except Exception:
            logger.warning("Failed to migrate settings paths", exc_info=True)


def _remove_empty_dirs(path: Path) -> None:
    """Recursively remove empty directories."""
    if not path.is_dir():
        return
    for child in list(path.iterdir()):
        if child.is_dir():
            _remove_empty_dirs(child)
    # Remove if now empty (ignoring .cache dirs)
    remaining = [p for p in path.iterdir() if p.name != ".cache"]
    if not remaining:
        # Remove .cache too if present
        cache = path / ".cache"
        if cache.exists():
            import shutil
            shutil.rmtree(cache, ignore_errors=True)
        if not any(path.iterdir()):
            path.rmdir()


_migrate_models_layout(DEFAULT_MODELS_DIR)

# Create the new subdirectories
for _subdir in ("diffusion_models", "text_encoders", "loras", "upscale_models"):
    (DEFAULT_MODELS_DIR / _subdir).mkdir(exist_ok=True)

logger.info(f"Models directory: {DEFAULT_MODELS_DIR}")

# ============================================================
# Settings
# ============================================================

SETTINGS_DIR = APP_DATA_DIR
SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE = SETTINGS_DIR / "settings.json"

DEFAULT_APP_SETTINGS = AppSettings()

from app_factory import DEFAULT_ALLOWED_ORIGINS, create_app
from state import RuntimeConfig, build_initial_state
from runtime_config.model_download_specs import (
    DEFAULT_MODEL_DOWNLOAD_SPECS,
    DEFAULT_REQUIRED_MODEL_TYPES,
)
from state.app_state_types import ModelFileType
from server_utils.model_layout_migration import migrate_legacy_models_layout
from services.gpu_info.gpu_info_impl import GpuInfoImpl

migrate_legacy_models_layout(APP_DATA_DIR)

LTX_API_BASE_URL = "https://api.ltx.video"


def _resolve_force_api_generations() -> bool:
    gpu_info = GpuInfoImpl()
    system = platform.system()
    cuda_available = gpu_info.get_cuda_available()
    vram_gb = gpu_info.get_vram_total_gb()

    # Local-processing-only mode: never force cloud/API generation paths.
    force_api_generations = False
    logger.info(
        "Runtime policy force_api_generations=%s (system=%s cuda_available=%s vram_gb=%s)",
        force_api_generations,
        system,
        cuda_available,
        vram_gb,
    )
    return force_api_generations


FORCE_API_GENERATIONS = _resolve_force_api_generations()
REQUIRED_MODEL_TYPES: frozenset[ModelFileType] = (
    frozenset() if FORCE_API_GENERATIONS else DEFAULT_REQUIRED_MODEL_TYPES
)

CAMERA_MOTION_PROMPTS = {
    "none": "",
    "static": ", static camera, locked off shot, no camera movement",
    "focus_shift": ", focus shift, rack focus, changing focal point",
    "dolly_in": ", dolly in, camera pushing forward, smooth forward movement",
    "dolly_out": ", dolly out, camera pulling back, smooth backward movement",
    "dolly_left": ", dolly left, camera tracking left, lateral movement",
    "dolly_right": ", dolly right, camera tracking right, lateral movement",
    "jib_up": ", jib up, camera rising up, upward crane movement",
    "jib_down": ", jib down, camera lowering down, downward crane movement",
}

DEFAULT_NEGATIVE_PROMPT = """blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field"""

runtime_config = RuntimeConfig(
    device=DEVICE,
    default_models_dir=DEFAULT_MODELS_DIR,
    model_download_specs=DEFAULT_MODEL_DOWNLOAD_SPECS,
    required_model_types=REQUIRED_MODEL_TYPES,
    outputs_dir=OUTPUTS_DIR,
    settings_file=SETTINGS_FILE,
    ltx_api_base_url=LTX_API_BASE_URL,
    force_api_generations=FORCE_API_GENERATIONS,
    use_sage_attention=use_sage_attention,
    camera_motion_prompts=CAMERA_MOTION_PROMPTS,
    default_negative_prompt=DEFAULT_NEGATIVE_PROMPT,
)

handler = build_initial_state(runtime_config, DEFAULT_APP_SETTINGS)

auth_token = os.environ.get("LTX_AUTH_TOKEN", "")
admin_token = os.environ.get("LTX_ADMIN_TOKEN", "")

app = create_app(
    handler=handler,
    allowed_origins=DEFAULT_ALLOWED_ORIGINS,
    auth_token=auth_token,
    admin_token=admin_token,
)


def precache_model_files(model_dir: Path) -> int:
    if not model_dir.exists():
        return 0
    total_bytes = 0
    for f in model_dir.rglob("*"):
        if f.is_file() and f.suffix in (
            ".safetensors",
            ".bin",
            ".pt",
            ".pth",
            ".onnx",
            ".model",
        ):
            try:
                size = f.stat().st_size
                with open(f, "rb") as fh:
                    while fh.read(8 * 1024 * 1024):
                        pass
                total_bytes += size
            except Exception:
                logger.warning("Failed to precache model file: %s", f, exc_info=True)
    return total_bytes


def background_warmup() -> None:
    handler.health.default_warmup()


def log_hardware_info() -> None:
    """Log runtime hardware and environment details."""
    gpu = GpuInfoImpl()
    gpu_info = gpu.get_gpu_info()
    vram_gb = gpu_info["vram"] // 1024 if gpu_info["vram"] else 0

    logger.info(f"Platform: {platform.system()} ({platform.machine()})")
    logger.info(f"Device: {DEVICE}  |  Dtype: {DTYPE}")
    logger.info(f"GPU: {gpu_info['name']}  |  VRAM: {vram_gb} GB")
    logger.info(f"SageAttention: {'enabled' if use_sage_attention else 'disabled'}")
    logger.info(f"Python: {sys.version.split()[0]}  |  Torch: {torch.__version__}")


if __name__ == "__main__":
    import asyncio
    import uvicorn

    port = int(os.environ.get("LTX_PORT", "") or PORT)
    logger.info("=" * 60)
    logger.info("LTX-2 Video Generation Server (FastAPI + Uvicorn)")
    log_hardware_info()
    logger.info("=" * 60)

    warmup_thread = threading.Thread(target=background_warmup, daemon=True)
    warmup_thread.start()

    # Use our root logging config so uvicorn logs go to stdout (not its
    # default stderr), letting Electron tag them correctly as INFO.
    log_config: dict[str, object] = {
        "version": 1,
        "disable_existing_loggers": False,
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO"},
            "uvicorn.error": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }

    import socket as _socket

    import os

    host = os.environ.get("BACKEND_HOST", "127.0.0.1")

    sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    actual_port = int(sock.getsockname()[1])

    config = uvicorn.Config(
        app,
        host=host,
        port=actual_port,
        log_level="info",
        access_log=False,
        log_config=log_config,
    )
    server = uvicorn.Server(config)

    _orig_startup = server.startup

    async def _startup_with_ready_msg(
        sockets: list[_socket.socket] | None = None,
    ) -> None:
        await _orig_startup(sockets=sockets)
        if server.started:
            # Machine-parseable ready message — Electron matches this line
            print(f"Server running on http://{host}:{actual_port}", flush=True)

    server.startup = _startup_with_ready_msg  # type: ignore[assignment]

    asyncio.run(server.serve(sockets=[sock]))
