"""Pipeline lifecycle and warmup handler."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from handlers.base import StateHandlerBase
from handlers.text_handler import TextHandler
from runtime_config.model_download_specs import resolve_model_path
from services.interfaces import (
    A2VPipeline,
    DepthProcessorPipeline,
    FastVideoPipeline,
    GpuInfo,
    ImageGenerationPipeline,
    GpuCleaner,
    IcLoraPipeline,
    PoseProcessorPipeline,
    RetakePipeline,
    VideoPipelineModelType,
)
from services.services_utils import device_supports_fp8, get_device_type
from services.vram_manager.vram_manager import VRAMManager
from state.app_state_types import (
    A2VPipelineState,
    AppState,
    CpuSlot,
    GenerationRunning,
    GpuSlot,
    ICLoraState,
    RetakePipelineState,
    VideoPipelineState,
    VideoPipelineWarmth,
)

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

# VRAM threshold (GB) above which we use the standard DistilledPipeline.
# The standard pipeline already does sequential offloading via cleanup_memory()
# between phases (text encode → denoise → decode). With FP8 quantization and
# the text-encoder monkey-patch (which caches embeddings), it fits in ~20-22 GB.
_HIGH_VRAM_THRESHOLD = 48  # Only GPUs with ≥48GB can hold all models simultaneously


def _display_model_path(path: str | None) -> str:
    if not path:
        return "none"
    return Path(path).name


def _display_lora_selection(
    primary_lora_path: str | None,
    primary_lora_strength: float,
    extra_loras: list[tuple[str, float]] | None,
) -> str:
    selections: list[str] = []
    if primary_lora_path is not None:
        selections.append(f"{Path(primary_lora_path).name}@{primary_lora_strength:.2f}")
    if extra_loras is not None:
        selections.extend(
            f"{Path(path).name}@{strength:.2f}" for path, strength in extra_loras
        )
    return ", ".join(selections) if selections else "none"


def _resolve_text_encoder_variant_path(models_dir: Path, preferred_path: str) -> str | None:
    candidate = models_dir / preferred_path
    if not candidate.exists():
        return None

    if candidate.is_file() and candidate.suffix.lower() == ".safetensors":
        parent = candidate.parent
        if candidate.name.startswith("model-") and "-of-" in candidate.stem:
            shard_paths = sorted(parent.glob("model-*.safetensors"))
            if len(shard_paths) > 1:
                logger.info(
                    "Using text encoder variant directory %s for shard selection %s",
                    parent,
                    candidate.name,
                )
                return str(parent)

    return str(candidate)


def _looks_like_incomplete_transformer_checkpoint(checkpoint_path: Path) -> bool:
    """Detect the known 4-tensor safetensors shim that is not a full transformer."""
    if checkpoint_path.suffix.lower() != ".safetensors":
        return False
    try:
        from safetensors import safe_open

        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
            tensor_count = len(handle.keys())
        return tensor_count < 100
    except Exception:
        logger.warning(
            "Failed to inspect checkpoint %s", checkpoint_path, exc_info=True
        )
        return False


def _find_full_checkpoint_candidate(models_dir: Path) -> Path | None:
    """Find a full safetensors checkpoint usable for VAE/audio/vocoder builders."""
    search_roots = [
        models_dir / "diffusion_models",
        models_dir,
    ]
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.safetensors"):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            if _looks_like_incomplete_transformer_checkpoint(path):
                continue
            candidates.append(path)

    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_size)


def _find_dev_checkpoint_candidate(models_dir: Path) -> Path | None:
    """Prefer an official dev safetensors checkpoint for quality mode when available."""
    search_roots = [models_dir / "diffusion_models", models_dir]
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*dev*.safetensors"):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            if _looks_like_incomplete_transformer_checkpoint(path):
                continue
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_size)


def _find_dev_gguf_candidate(
    models_dir: Path, preferred_quant: str = "Q8_0"
) -> Path | None:
    """Find a dev GGUF, preferring the requested quant level when available."""
    search_roots = [models_dir / "diffusion_models", models_dir / "gguf", models_dir]
    search_order = [preferred_quant]
    for quant in [
        "Q8_0",
        "Q6_K",
        "Q5_1",
        "Q5_0",
        "Q4_K_M",
        "Q4_K_S",
        "Q4_1",
        "Q4_0",
        "Q3_K_M",
        "Q2_K",
    ]:
        if quant not in search_order:
            search_order.append(quant)

    for quant in search_order:
        for root in search_roots:
            if not root.exists():
                continue
            for path in root.rglob(f"*dev*{quant}*.gguf"):
                if path.is_file():
                    return path

    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*dev*.gguf"):
            if path.is_file():
                return path

    return None


def _has_split_ltx_component_fallback(models_dir: Path) -> bool:
    """Whether split LTX VAE/audio component weights are available locally."""
    video_candidates = (
        models_dir / "vae" / "LTX23_video_vae_bf16.safetensors",
        models_dir / "vae" / "LTX2_video_vae_bf16.safetensors",
    )
    audio_candidates = (
        models_dir / "vae" / "LTX23_audio_vae_bf16.safetensors",
        models_dir / "vae" / "LTX2_audio_vae_bf16.safetensors",
    )
    return any(path.exists() for path in video_candidates) and any(
        path.exists() for path in audio_candidates
    )


def _ensure_split_ltx_component_fallback(models_dir: Path) -> bool:
    """Ensure split LTX VAE/audio weights exist locally for GGUF decode fallback."""
    if _has_split_ltx_component_fallback(models_dir):
        return True

    vae_dir = models_dir / "vae"
    vae_dir.mkdir(parents=True, exist_ok=True)
    required_files = (
        "vae/LTX23_video_vae_bf16.safetensors",
        "vae/LTX23_audio_vae_bf16.safetensors",
    )

    try:
        from huggingface_hub import hf_hub_download  # type: ignore[reportUnknownVariableType]

        for filename in required_files:
            target = vae_dir / Path(filename).name
            if target.exists():
                continue

            logger.info("Downloading split LTX component for GGUF decode: %s", filename)
            downloaded = Path(
                str(
                    hf_hub_download(  # type: ignore[reportUnknownMemberType]
                        repo_id="Kijai/LTX2.3_comfy",
                        filename=filename,
                    )
                )
            )
            if downloaded != target:
                shutil.copy2(str(downloaded), str(target))

    except Exception:
        logger.warning(
            "Failed to auto-download split LTX component fallback", exc_info=True
        )

    return _has_split_ltx_component_fallback(models_dir)


def _path_looks_distilled(path: str | None) -> bool:
    if path is None:
        return False
    return "distilled" in Path(path).name.lower()


def _should_use_standard_fast_pipeline(
    *,
    model_type: VideoPipelineModelType,
    checkpoint_path: str | None,
    gguf_path: str | None,
) -> bool:
    return (
        model_type == "fast"
        and gguf_path is None
        and _path_looks_distilled(checkpoint_path)
    )


class PipelinesHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        text_handler: TextHandler,
        gpu_cleaner: GpuCleaner,
        fast_video_pipeline_class: type[FastVideoPipeline],
        image_generation_pipeline_class: type[ImageGenerationPipeline],
        ic_lora_pipeline_class: type[IcLoraPipeline],
        depth_processor_pipeline_class: type[DepthProcessorPipeline],
        pose_processor_pipeline_class: type[PoseProcessorPipeline],
        a2v_pipeline_class: type[A2VPipeline],
        retake_pipeline_class: type[RetakePipeline],
        config: RuntimeConfig,
        gpu_info: GpuInfo | None = None,
    ) -> None:
        super().__init__(state, lock, config)
        self._text_handler = text_handler
        self._gpu_cleaner = gpu_cleaner
        self._fast_video_pipeline_class = fast_video_pipeline_class
        self._image_generation_pipeline_class = image_generation_pipeline_class
        self._ic_lora_pipeline_class = ic_lora_pipeline_class
        self._depth_processor_pipeline_class = depth_processor_pipeline_class
        self._pose_processor_pipeline_class = pose_processor_pipeline_class
        self._a2v_pipeline_class = a2v_pipeline_class
        self._retake_pipeline_class = retake_pipeline_class
        self._gpu_info = gpu_info
        self._runtime_device = get_device_type(self.config.device)

    def _ensure_no_running_generation(self) -> None:
        match self.state.gpu_slot:
            case GpuSlot(generation=GenerationRunning()):
                raise RuntimeError("Generation already running; cannot swap pipelines")
            case _:
                return

    def _compute_pipeline_fingerprint(self, model_type: VideoPipelineModelType) -> str:
        """Compute a fingerprint of all settings that affect pipeline construction.

        If any of these change, the pipeline must be rebuilt.
        """
        import hashlib

        settings = self.state.app_settings
        parts = [
            model_type,
            settings.preferred_model_path.strip(),
            settings.preferred_gguf_path.strip(),
            str(settings.custom_model.steps),
            str(settings.pro_model.steps),
            settings.preferred_text_encoder_path.strip(),
            str(settings.num_blocks_to_swap),
            settings.run_mode,
        ]
        # Include all selected LoRAs (path + strength)
        for lora in settings.selected_loras:
            parts.append(f"{lora.path.strip()}:{lora.strength}")
        raw = "|".join(parts)
        return hashlib.md5(raw.encode()).hexdigest()

    def _pipeline_matches_config(self, model_type: VideoPipelineModelType) -> bool:
        """Check if the current GPU pipeline matches the requested model type AND settings."""
        match self.state.gpu_slot:
            case GpuSlot(active_pipeline=VideoPipelineState(pipeline=pipeline)):
                if getattr(pipeline, "_model_mode", None) != model_type:
                    return False
                stored_fp = getattr(pipeline, "_config_fingerprint", None)
                if stored_fp is None:
                    return False
                return stored_fp == self._compute_pipeline_fingerprint(model_type)
            case _:
                return False

    def _assert_invariants(self) -> None:
        gpu_is_zit = False
        match self.state.gpu_slot:
            case GpuSlot(
                active_pipeline=VideoPipelineState()
                | ICLoraState()
                | A2VPipelineState()
                | RetakePipelineState()
            ):
                gpu_is_zit = False
            case GpuSlot():
                gpu_is_zit = True
            case _:
                gpu_is_zit = False

        if gpu_is_zit and self.state.cpu_slot is not None:
            raise RuntimeError(
                "Invariant violation: ZIT cannot be in both GPU and CPU slots"
            )

    def _install_text_patches_if_needed(self) -> None:
        te = self.state.text_encoder
        if te is None:
            return
        te.service.install_patches(lambda: self.state)

    def _compile_if_enabled(self, state: VideoPipelineState) -> VideoPipelineState:
        if not self.state.app_settings.use_torch_compile:
            return state
        if state.is_compiled:
            return state
        if self._runtime_device == "mps":
            logger.info(
                "Skipping torch.compile() for %s - not supported on MPS",
                state.pipeline.pipeline_kind,
            )
            return state

        try:
            state.pipeline.compile_transformer()
            state.is_compiled = True
        except Exception as exc:
            logger.warning("Failed to compile transformer: %s", exc, exc_info=True)
        return state

    def _get_vram_gb(self) -> int | None:
        """Query total VRAM via the injected GPU info service."""
        if self._gpu_info is None:
            return None
        try:
            return self._gpu_info.get_vram_total_gb()
        except Exception:
            logger.warning("Could not query VRAM", exc_info=True)
            return None

    def _create_video_pipeline(
        self, model_type: VideoPipelineModelType
    ) -> VideoPipelineState:
        vram_gb = self._get_vram_gb()
        preferred_checkpoint_path, preferred_gguf_path = (
            self._resolve_preferred_model_paths()
        )

        if model_type == "quality" and preferred_gguf_path is not None:
            dev_checkpoint = _find_dev_checkpoint_candidate(self.models_dir)
            if dev_checkpoint is not None:
                logger.info(
                    "Quality mode prefers the official dev safetensors checkpoint over GGUF: %s",
                    dev_checkpoint,
                )
                preferred_checkpoint_path = str(dev_checkpoint)
                preferred_gguf_path = None

        if model_type == "custom":
            return self._create_low_vram_pipeline(
                model_type,
                vram_gb or 0,
                preferred_checkpoint_path=preferred_checkpoint_path,
                preferred_gguf_path=preferred_gguf_path,
                force_gguf=False,
                skip_loras=False,
            )

        # For "fast" mode: prefer a distilled base, no LoRAs, 8 steps
        # For "balanced" mode: dev checkpoint/GGUF + distilled LoRA, 8 steps
        # For "quality" mode: no LoRAs, use dev checkpoint / GGUF with custom steps
        force_gguf = model_type == "fast"
        skip_loras = model_type in ("fast", "quality")

        gguf_path_for_mode = preferred_gguf_path
        if model_type == "fast":
            # Prefer smaller distilled quants for speed/stability in local mode.
            # If the user selected a GGUF, we still try to match that quant first.
            quant = "Q4_K_M"
            gguf_path_for_mode = self._resolve_distilled_gguf_path(
                preferred_gguf_path, quant
            )
            if gguf_path_for_mode is None:
                logger.warning(
                    "Fast mode selected but no distilled GGUF found; will fall back to checkpoint path if available"
                )

        if _should_use_standard_fast_pipeline(
            model_type=model_type,
            checkpoint_path=preferred_checkpoint_path,
            gguf_path=gguf_path_for_mode,
        ):
            logger.info(
                "Fast mode using original two-stage distilled pipeline with checkpoint=%s",
                preferred_checkpoint_path,
            )
            return self._create_standard_pipeline(
                model_type,
                preferred_checkpoint_path=preferred_checkpoint_path,
                skip_loras=skip_loras,
            )

        use_low_vram = gguf_path_for_mode is not None or (
            vram_gb is not None and vram_gb < _HIGH_VRAM_THRESHOLD
        )

        if use_low_vram:
            return self._create_low_vram_pipeline(
                model_type,
                vram_gb or 0,
                preferred_checkpoint_path=preferred_checkpoint_path,
                preferred_gguf_path=gguf_path_for_mode,
                force_gguf=force_gguf,
                skip_loras=skip_loras,
            )
        return self._create_standard_pipeline(
            model_type,
            preferred_checkpoint_path=preferred_checkpoint_path,
            skip_loras=skip_loras,
        )

    def _resolve_selected_loras(
        self, *, allow_legacy_fallback: bool = True
    ) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for item in self.state.app_settings.selected_loras:
            path = item.path.strip()
            if path and Path(path).exists():
                out.append((path, item.strength))

        # Backward-compatibility for older settings.
        if not out and allow_legacy_fallback:
            preferred_path = self.state.app_settings.preferred_lora_path.strip()
            if preferred_path and Path(preferred_path).exists():
                out.append(
                    (preferred_path, self.state.app_settings.preferred_lora_strength)
                )
        return out

    def _resolve_default_distilled_lora(self) -> tuple[str, float] | None:
        candidates = [
            self.models_dir / "loras" / "ltx-2.3-22b-distilled-lora-384.safetensors",
            self.models_dir / "ltx-2.3-22b-distilled-lora-384.safetensors",
            self.models_dir / "ltx-2-19b-distilled-lora-384.safetensors",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return str(candidate), 0.6

        lora_dir = self.models_dir / "loras"
        if lora_dir.exists():
            for candidate in sorted(lora_dir.glob("*distilled*lora*.safetensors")):
                if candidate.is_file():
                    return str(candidate), 0.6
        return None

    def _base_model_is_distilled(
        self,
        *,
        preferred_checkpoint_path: str | None,
        preferred_gguf_path: str | None,
    ) -> bool:
        if preferred_gguf_path is not None:
            return _path_looks_distilled(preferred_gguf_path)
        return _path_looks_distilled(preferred_checkpoint_path)

    def _resolve_pipeline_loras(
        self,
        *,
        model_type: VideoPipelineModelType,
        skip_loras: bool,
        preferred_checkpoint_path: str | None,
        preferred_gguf_path: str | None,
    ) -> tuple[str | None, float, list[tuple[str, float]] | None]:
        primary_lora_path: str | None = None
        primary_lora_strength = 1.0
        extra_loras: list[tuple[str, float]] | None = None

        if skip_loras:
            return primary_lora_path, primary_lora_strength, extra_loras

        selected_loras = self._resolve_selected_loras(
            allow_legacy_fallback=model_type == "balanced"
        )
        base_is_distilled = self._base_model_is_distilled(
            preferred_checkpoint_path=preferred_checkpoint_path,
            preferred_gguf_path=preferred_gguf_path,
        )

        if not selected_loras and model_type == "balanced" and not base_is_distilled:
            default_lora = self._resolve_default_distilled_lora()
            if default_lora is not None:
                selected_loras = [default_lora]

        if base_is_distilled and any(
            "distilled" in Path(path).name.lower() for path, _ in selected_loras
        ):
            logger.warning(
                "Skipping distilled LoRAs because the selected base model is already distilled "
                "(checkpoint=%s gguf=%s)",
                preferred_checkpoint_path,
                preferred_gguf_path,
            )
            selected_loras = [
                (path, strength)
                for path, strength in selected_loras
                if "distilled" not in Path(path).name.lower()
            ]

        primary_lora_path = selected_loras[0][0] if selected_loras else None
        primary_lora_strength = selected_loras[0][1] if selected_loras else 1.0
        extra_loras = selected_loras[1:] if len(selected_loras) > 1 else None
        return primary_lora_path, primary_lora_strength, extra_loras

    def _resolve_preferred_model_paths(self) -> tuple[str | None, str | None]:
        from services.gguf_loader.gguf_loader import GGUFModelLoader

        preferred_path = self.state.app_settings.preferred_model_path.strip()
        preferred_gguf_path = self.state.app_settings.preferred_gguf_path.strip()

        if preferred_gguf_path and Path(preferred_gguf_path).exists():
            return None, preferred_gguf_path

        if preferred_path and Path(preferred_path).exists():
            if preferred_path.lower().endswith(".gguf"):
                return None, preferred_path
            preferred_checkpoint = Path(preferred_path)
            if _looks_like_incomplete_transformer_checkpoint(preferred_checkpoint):
                gguf_loader = GGUFModelLoader(self.models_dir)
                fallback_gguf = gguf_loader.find_gguf_model("Q8_0")
                if fallback_gguf is not None:
                    logger.warning(
                        "Preferred checkpoint %s looks incomplete; using GGUF %s instead",
                        preferred_checkpoint,
                        fallback_gguf,
                    )
                    return None, str(fallback_gguf)
                logger.warning(
                    "Preferred checkpoint %s looks incomplete and no GGUF fallback was found",
                    preferred_checkpoint,
                )
            return preferred_path, None

        return None, None

    def _resolve_distilled_gguf_path(
        self, preferred_gguf_path: str | None, preferred_quant: str
    ) -> str | None:
        """Find a distilled GGUF, preferably matching the selected quant level."""
        search_dirs = [
            self.models_dir / "diffusion_models",
            self.models_dir / "gguf",
            self.models_dir,
        ]

        # If user picked a dev GGUF, try the distilled sibling with same quant.
        if preferred_gguf_path is not None:
            selected = Path(preferred_gguf_path)
            name = selected.name
            for quant in [
                "Q8_0",
                "Q6_K",
                "Q5_1",
                "Q5_0",
                "Q4_K_M",
                "Q4_K_S",
                "Q4_1",
                "Q4_0",
                "Q3_K_M",
                "Q2_K",
            ]:
                if quant in name:
                    preferred_quant = quant
                    break

        # Prefer exact distilled quant match.
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for gguf_file in search_dir.rglob(f"*distilled*{preferred_quant}*.gguf"):
                return str(gguf_file)

        # Fallback: any distilled GGUF.
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for gguf_file in search_dir.rglob("*distilled*.gguf"):
                return str(gguf_file)

        return None

    def _create_standard_pipeline(
        self,
        model_type: VideoPipelineModelType,
        *,
        preferred_checkpoint_path: str | None = None,
        skip_loras: bool = False,
    ) -> VideoPipelineState:
        """Original full-GPU pipeline for high-VRAM systems (≥31 GB)."""
        gemma_root = self._text_handler.resolve_gemma_root()

        checkpoint_path = preferred_checkpoint_path or str(
            resolve_model_path(
                self.models_dir, self.config.model_download_specs, "checkpoint"
            )
        )
        upsampler_path = str(
            resolve_model_path(
                self.models_dir, self.config.model_download_specs, "upsampler"
            )
        )

        primary_lora_path, primary_lora_strength, extra_loras = (
            self._resolve_pipeline_loras(
                model_type=model_type,
                skip_loras=skip_loras,
                preferred_checkpoint_path=checkpoint_path,
                preferred_gguf_path=None,
            )
        )

        pipeline = self._fast_video_pipeline_class.create(
            checkpoint_path,
            gemma_root,
            upsampler_path,
            self.config.device,
            lora_path=primary_lora_path,
            lora_strength=primary_lora_strength,
            extra_loras=extra_loras,
        )
        pipeline._model_mode = model_type  # type: ignore[attr-defined]
        pipeline._config_fingerprint = self._compute_pipeline_fingerprint(model_type)  # type: ignore[attr-defined]

        logger.info(
            "Pipeline selection: kind=video mode=%s checkpoint=%s gguf=%s loras=%s",
            model_type,
            _display_model_path(checkpoint_path),
            "none",
            _display_lora_selection(
                primary_lora_path,
                primary_lora_strength,
                extra_loras,
            ),
        )

        state = VideoPipelineState(
            pipeline=pipeline,
            warmth=VideoPipelineWarmth.COLD,
            is_compiled=False,
        )
        return self._compile_if_enabled(state)

    def _create_low_vram_pipeline(
        self,
        model_type: VideoPipelineModelType,
        vram_gb: int,
        *,
        preferred_checkpoint_path: str | None = None,
        preferred_gguf_path: str | None = None,
        force_gguf: bool = False,
        skip_loras: bool = False,
    ) -> VideoPipelineState:
        """Low-VRAM pipeline with sequential offloading + GGUF + block swap."""
        from services.fast_video_pipeline.ltx_low_vram_pipeline import (
            LTXLowVRAMPipeline,
        )
        from services.gguf_loader.gguf_loader import GGUFModelLoader

        gemma_root = self._text_handler.resolve_gemma_root()
        checkpoint_path = preferred_checkpoint_path or str(
            resolve_model_path(
                self.models_dir, self.config.model_download_specs, "checkpoint"
            )
        )
        upsampler_path = str(
            resolve_model_path(
                self.models_dir, self.config.model_download_specs, "upsampler"
            )
        )

        vram_manager = VRAMManager(
            self.config.device,
            vram_gb,
            user_blocks_on_gpu=self.state.app_settings.num_blocks_to_swap,
            user_run_mode=self.state.app_settings.run_mode,
        )

        # Default to the full unquantized checkpoint unless the user explicitly
        # selected a GGUF model. This makes it easier to compare prompt
        # adherence and isolate GGUF-specific regressions.
        gguf_loader = GGUFModelLoader(self.models_dir)
        gguf_path: str | None = preferred_gguf_path
        should_auto_use_gguf = False
        if gguf_path is None and should_auto_use_gguf:
            recommended_quant = vram_manager.get_recommended_gguf_quant()
            if model_type == "fast":
                gguf_file_str = self._resolve_distilled_gguf_path(
                    None, recommended_quant
                )
                gguf_file = Path(gguf_file_str) if gguf_file_str is not None else None
            else:
                gguf_file = gguf_loader.find_gguf_model(recommended_quant)
            if gguf_file is not None:
                gguf_path = str(gguf_file)
                logger.info(
                    "Using GGUF model: %s (recommended: %s)",
                    gguf_path,
                    recommended_quant,
                )
            else:
                if model_type == "fast":
                    logger.warning(
                        "Fast mode requested but no distilled GGUF model was found; using distilled checkpoint instead"
                    )
                else:
                    logger.info(
                        "No GGUF model found; using standard checkpoint with low-VRAM offloading"
                    )
        elif gguf_path is not None:
            logger.info("Using user-selected GGUF model: %s", gguf_path)
        else:
            logger.info("Using standard unquantized checkpoint by default")

        if gguf_path is not None and _looks_like_incomplete_transformer_checkpoint(
            Path(checkpoint_path)
        ):
            full_checkpoint = _find_full_checkpoint_candidate(self.models_dir)
            if full_checkpoint is not None:
                logger.warning(
                    "Checkpoint %s is transformer-only; using full checkpoint %s for VAE/audio/vocoder components",
                    checkpoint_path,
                    full_checkpoint,
                )
                checkpoint_path = str(full_checkpoint)
            elif _ensure_split_ltx_component_fallback(self.models_dir):
                logger.warning(
                    "Checkpoint %s is transformer-only; using split LTX component weights from %s/vae",
                    checkpoint_path,
                    self.models_dir,
                )
            else:
                raise RuntimeError(
                    "The selected GGUF can run denoising, but local decode also needs a full "
                    "LTX safetensors checkpoint or split LTX VAE/audio component weights for "
                    "the VAE/audio/vocoder components. "
                    f"Current checkpoint {checkpoint_path} is transformer-only/incomplete. "
                    "Install the full LTX checkpoint bundle or the split LTX23/LTX2 VAE files, then try again."
                )

        primary_lora_path, primary_lora_strength, extra_loras = (
            self._resolve_pipeline_loras(
                model_type=model_type,
                skip_loras=skip_loras,
                preferred_checkpoint_path=checkpoint_path,
                preferred_gguf_path=gguf_path,
            )
        )

        # Fast and balanced are both fixed 8-step modes.
        num_inference_steps: int | None = None
        if model_type in ("fast", "balanced"):
            num_inference_steps = 8
        elif model_type == "quality":
            num_inference_steps = self.state.app_settings.pro_model.steps
        elif model_type == "custom":
            num_inference_steps = self.state.app_settings.custom_model.steps

        preferred_text_encoder_path = (
            self.state.app_settings.preferred_text_encoder_path.strip()
        )
        text_encoder_variant_path = None
        if preferred_text_encoder_path:
            text_encoder_variant_path = _resolve_text_encoder_variant_path(
                self.models_dir,
                preferred_text_encoder_path,
            )

        if model_type in ("fast", "balanced"):
            use_upscaler = self.state.app_settings.fast_model.use_upscaler
        elif model_type == "quality":
            use_upscaler = self.state.app_settings.pro_model.use_upscaler
        else:
            use_upscaler = self.state.app_settings.custom_model.use_upscaler
        a2v_decode_tiling = self.state.app_settings.a2v_decode_tiling

        pipeline = LTXLowVRAMPipeline.create(
            checkpoint_path,
            gemma_root,
            upsampler_path,
            self.config.device,
            vram_manager=vram_manager,
            gguf_path=gguf_path,
            lora_path=primary_lora_path,
            lora_strength=primary_lora_strength,
            extra_loras=extra_loras,
            use_sage_attention=self.config.use_sage_attention,
            num_inference_steps=num_inference_steps,
            text_encoder_variant_path=text_encoder_variant_path,
            use_upscaler=use_upscaler,
        )
        pipeline._model_mode = model_type  # type: ignore[attr-defined]
        pipeline._config_fingerprint = self._compute_pipeline_fingerprint(model_type)  # type: ignore[attr-defined]

        logger.info(
            "Created low-VRAM pipeline: mode=%s tier=%s strategy=%s checkpoint=%s gguf=%s loras=%s",
            model_type,
            vram_manager.tier.value,
            vram_manager.offload_strategy.value,
            _display_model_path(checkpoint_path),
            _display_model_path(gguf_path),
            _display_lora_selection(
                primary_lora_path,
                primary_lora_strength,
                extra_loras,
            ),
        )

        return VideoPipelineState(
            pipeline=pipeline,
            warmth=VideoPipelineWarmth.COLD,
            is_compiled=False,
        )

    def unload_gpu_pipeline(self) -> None:
        with self._lock:
            self._ensure_no_running_generation()
            self.state.gpu_slot = None
            self._assert_invariants()
        self._gpu_cleaner.cleanup()

    def recover_after_oom(self) -> None:
        cached_encoder = None

        with self._lock:
            self.state.gpu_slot = None

            te = self.state.text_encoder
            if te is not None:
                te.api_embeddings = None
                te.prompt_cache.clear()
                cached_encoder = te.cached_encoder
                te.cached_encoder = None

            self._assert_invariants()

        if cached_encoder is not None:
            try:
                cached_encoder.to("cpu")
            except Exception:
                logger.warning(
                    "Failed to offload cached text encoder during OOM recovery",
                    exc_info=True,
                )

        self._gpu_cleaner.cleanup()

    def park_zit_on_cpu(self) -> None:
        zit: ImageGenerationPipeline | None = None

        with self._lock:
            if self.state.gpu_slot is None:
                return

            active = self.state.gpu_slot.active_pipeline
            if isinstance(
                active,
                (
                    VideoPipelineState,
                    ICLoraState,
                    A2VPipelineState,
                    RetakePipelineState,
                ),
            ):
                return

            generation = self.state.gpu_slot.generation
            if isinstance(generation, GenerationRunning):
                raise RuntimeError("Cannot park ZIT while generation is running")

            zit = active
            self.state.gpu_slot = None

        assert zit is not None
        zit.to("cpu")
        self._gpu_cleaner.cleanup()

        with self._lock:
            self.state.cpu_slot = CpuSlot(active_pipeline=zit)
            self._assert_invariants()

    def _resolve_preferred_zit_path(self) -> str | None:
        preferred_zit = self.state.app_settings.preferred_zit_model_path.strip()
        if not preferred_zit:
            return None

        preferred_path = Path(preferred_zit)
        resolved = (
            preferred_path
            if preferred_path.is_absolute()
            else self.models_dir / preferred_zit
        )
        if not resolved.exists():
            return None
        return str(resolved)

    def _find_zit_gguf(self) -> str | None:
        """Search for a Z-Image-Turbo GGUF file in diffusion_models/ and legacy gguf/."""
        for search_dir in (
            self.models_dir / "diffusion_models",
            self.models_dir / "gguf",
            self.models_dir,
        ):
            if not search_dir.exists():
                continue
            for f in search_dir.rglob("*.gguf"):
                name = f.name.lower()
                if "z-image" in name or "zimage" in name or "z_image" in name:
                    logger.info("Found ZIT GGUF model: %s", f)
                    return str(f)
        return None

    def load_zit_to_gpu(self) -> ImageGenerationPipeline:
        with self._lock:
            if self.state.gpu_slot is not None:
                active = self.state.gpu_slot.active_pipeline
                if not isinstance(
                    active,
                    (
                        VideoPipelineState,
                        ICLoraState,
                        A2VPipelineState,
                        RetakePipelineState,
                    ),
                ):
                    return active
                self._ensure_no_running_generation()

        zit_service: ImageGenerationPipeline | None = None

        with self._lock:
            match self.state.cpu_slot:
                case CpuSlot(active_pipeline=stored):
                    zit_service = stored
                    self.state.cpu_slot = None
                case _:
                    zit_service = None

        if zit_service is None:
            # Check for user-preferred ZIT model path first
            preferred_zit_path = self._resolve_preferred_zit_path()
            if preferred_zit_path is not None:
                zit_path_str = preferred_zit_path
            else:
                zit_path = resolve_model_path(
                    self.models_dir, self.config.model_download_specs, "zit"
                )
                if zit_path.exists() and any(zit_path.iterdir()):
                    zit_path_str = str(zit_path)
                else:
                    # Search for GGUF ZIT files in diffusion_models/
                    gguf_zit = self._find_zit_gguf()
                    if gguf_zit is not None:
                        zit_path_str = gguf_zit
                    else:
                        raise RuntimeError(
                            "Z-Image-Turbo model not found. Download it from Settings → Model Downloads "
                            "(either the full model folder or a GGUF variant)."
                        )
            zit_service = self._image_generation_pipeline_class.create(
                zit_path_str, self._runtime_device
            )
        else:
            zit_service.to(self._runtime_device)

        self._gpu_cleaner.cleanup()

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=zit_service, generation=None)
            self._assert_invariants()

        return zit_service

    def preload_zit_to_cpu(self) -> ImageGenerationPipeline:
        with self._lock:
            match self.state.cpu_slot:
                case CpuSlot(active_pipeline=existing):
                    return existing
                case _:
                    pass

        preferred_zit_path = self._resolve_preferred_zit_path()
        if preferred_zit_path is not None:
            zit_path_str = preferred_zit_path
        else:
            zit_path = resolve_model_path(
                self.models_dir, self.config.model_download_specs, "zit"
            )
            if zit_path.exists() and any(zit_path.iterdir()):
                zit_path_str = str(zit_path)
            else:
                gguf_zit = self._find_zit_gguf()
                if gguf_zit is not None:
                    zit_path_str = gguf_zit
                else:
                    raise RuntimeError(
                        "Z-Image-Turbo model not found. Download it from Settings → Model Downloads."
                    )

        zit_service = self._image_generation_pipeline_class.create(zit_path_str, None)
        with self._lock:
            if self.state.cpu_slot is None:
                self.state.cpu_slot = CpuSlot(active_pipeline=zit_service)
                self._assert_invariants()
                return zit_service
            return self.state.cpu_slot.active_pipeline

    def _evict_gpu_pipeline_for_swap(self) -> None:
        should_park_zit = False
        should_cleanup = False

        with self._lock:
            self._ensure_no_running_generation()
            if self.state.gpu_slot is None:
                return

            active = self.state.gpu_slot.active_pipeline
            if isinstance(
                active,
                (
                    VideoPipelineState,
                    ICLoraState,
                    A2VPipelineState,
                    RetakePipelineState,
                ),
            ):
                self.state.gpu_slot = None
                self._assert_invariants()
                should_cleanup = True
            else:
                should_park_zit = True

        if should_park_zit:
            self.park_zit_on_cpu()
        elif should_cleanup:
            self._gpu_cleaner.cleanup()

    def load_gpu_pipeline(
        self, model_type: VideoPipelineModelType, should_warm: bool = False
    ) -> VideoPipelineState:
        self._install_text_patches_if_needed()

        state: VideoPipelineState | None = None
        with self._lock:
            if self._pipeline_matches_config(model_type):
                match self.state.gpu_slot:
                    case GpuSlot(
                        active_pipeline=VideoPipelineState() as existing_state
                    ):
                        state = existing_state
                    case _:
                        pass

        if state is None:
            self._evict_gpu_pipeline_for_swap()
            state = self._create_video_pipeline(model_type)
            with self._lock:
                self.state.gpu_slot = GpuSlot(active_pipeline=state, generation=None)
                self._assert_invariants()

        if should_warm and state.warmth == VideoPipelineWarmth.COLD:
            with self._lock:
                state.warmth = VideoPipelineWarmth.WARMING

            self.warmup_pipeline(model_type)
            with self._lock:
                if state.warmth == VideoPipelineWarmth.WARMING:
                    state.warmth = VideoPipelineWarmth.WARM

        return state

    def load_ic_lora(
        self,
        lora_path: str,
        depth_model_path: str | None = None,
        pose_model_path: str | None = None,
        person_detector_model_path: str | None = None,
    ) -> ICLoraState:
        self._install_text_patches_if_needed()

        vram_gb = 0
        if self._gpu_info is not None:
            vram_gb = self._gpu_info.get_vram_total_gb() or 0
        vram_manager = VRAMManager(
            self.config.device,
            vram_gb,
            user_blocks_on_gpu=self.state.app_settings.num_blocks_to_swap,
            user_run_mode=self.state.app_settings.run_mode,
        )

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(
                    active_pipeline=ICLoraState(
                        lora_path=current_lora_path,
                        depth_model_path=current_depth_model_path,
                        pose_model_path=current_pose_model_path,
                        person_detector_model_path=current_person_detector_model_path,
                    ) as state
                ) if (
                    current_lora_path == lora_path
                    and current_depth_model_path == depth_model_path
                    and current_pose_model_path == pose_model_path
                    and current_person_detector_model_path == person_detector_model_path
                ):
                    return state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()

        pipeline = self._ic_lora_pipeline_class.create(
            str(
                resolve_model_path(
                    self.models_dir, self.config.model_download_specs, "checkpoint"
                )
            ),
            self._text_handler.resolve_gemma_root(),
            str(
                resolve_model_path(
                    self.models_dir, self.config.model_download_specs, "upsampler"
                )
            ),
            lora_path,
            self.config.device,
            vram_manager,
        )
        depth_pipeline = None
        if depth_model_path is not None:
            depth_pipeline = self._depth_processor_pipeline_class.create(
                depth_model_path, self.config.device
            )
        pose_pipeline = None
        if pose_model_path is not None and person_detector_model_path is not None:
            pose_pipeline = self._pose_processor_pipeline_class.create(
                pose_model_path,
                person_detector_model_path,
                self.config.device,
            )
        state = ICLoraState(
            pipeline=pipeline,
            lora_path=lora_path,
            depth_pipeline=depth_pipeline,
            depth_model_path=depth_model_path,
            pose_pipeline=pose_pipeline,
            pose_model_path=pose_model_path,
            person_detector_model_path=person_detector_model_path,
        )

        logger.info(
            "Pipeline selection: kind=ic_lora checkpoint=%s lora=%s depth=%s pose=%s person_detector=%s",
            _display_model_path(
                str(
                    resolve_model_path(
                        self.models_dir, self.config.model_download_specs, "checkpoint"
                    )
                )
            ),
            _display_model_path(lora_path),
            _display_model_path(depth_model_path),
            _display_model_path(pose_model_path),
            _display_model_path(person_detector_model_path),
        )

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state, generation=None)
            self._assert_invariants()
        return state

    def reload_ic_lora_during_generation(
        self,
        lora_path: str,
        depth_model_path: str | None = None,
        pose_model_path: str | None = None,
        person_detector_model_path: str | None = None,
    ) -> ICLoraState:
        self._install_text_patches_if_needed()

        generation = None
        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(active_pipeline=ICLoraState(), generation=current_generation):
                    generation = current_generation
                    self.state.gpu_slot = None
                    self._assert_invariants()
                case _:
                    pass

        self._gpu_cleaner.cleanup()

        vram_gb = 0
        if self._gpu_info is not None:
            vram_gb = self._gpu_info.get_vram_total_gb() or 0
        vram_manager = VRAMManager(
            self.config.device,
            vram_gb,
            user_blocks_on_gpu=self.state.app_settings.num_blocks_to_swap,
            user_run_mode=self.state.app_settings.run_mode,
        )

        pipeline = self._ic_lora_pipeline_class.create(
            str(
                resolve_model_path(
                    self.models_dir, self.config.model_download_specs, "checkpoint"
                )
            ),
            self._text_handler.resolve_gemma_root(),
            str(
                resolve_model_path(
                    self.models_dir, self.config.model_download_specs, "upsampler"
                )
            ),
            lora_path,
            self.config.device,
            vram_manager,
        )
        depth_pipeline = None
        if depth_model_path is not None:
            depth_pipeline = self._depth_processor_pipeline_class.create(
                depth_model_path, self.config.device
            )
        pose_pipeline = None
        if pose_model_path is not None and person_detector_model_path is not None:
            pose_pipeline = self._pose_processor_pipeline_class.create(
                pose_model_path,
                person_detector_model_path,
                self.config.device,
            )
        state = ICLoraState(
            pipeline=pipeline,
            lora_path=lora_path,
            depth_pipeline=depth_pipeline,
            depth_model_path=depth_model_path,
            pose_pipeline=pose_pipeline,
            pose_model_path=pose_model_path,
            person_detector_model_path=person_detector_model_path,
        )

        logger.info(
            "Pipeline selection: kind=ic_lora reload=yes checkpoint=%s lora=%s depth=%s pose=%s person_detector=%s",
            _display_model_path(
                str(
                    resolve_model_path(
                        self.models_dir, self.config.model_download_specs, "checkpoint"
                    )
                )
            ),
            _display_model_path(lora_path),
            _display_model_path(depth_model_path),
            _display_model_path(pose_model_path),
            _display_model_path(person_detector_model_path),
        )

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state, generation=generation)
            self._assert_invariants()
        return state

    def load_a2v_pipeline(
        self, model_type: VideoPipelineModelType = "quality"
    ) -> A2VPipelineState:
        self._install_text_patches_if_needed()

        self._evict_gpu_pipeline_for_swap()

        vram_gb = 0
        if self._gpu_info is not None:
            vram_gb = self._gpu_info.get_vram_total_gb() or 0
        vram_manager = VRAMManager(
            self.config.device,
            vram_gb,
            user_blocks_on_gpu=self.state.app_settings.num_blocks_to_swap,
            user_run_mode=self.state.app_settings.run_mode,
        )

        preferred_checkpoint_path, preferred_gguf_path = (
            self._resolve_preferred_model_paths()
        )
        checkpoint_path = preferred_checkpoint_path or str(
            resolve_model_path(
                self.models_dir, self.config.model_download_specs, "checkpoint"
            )
        )

        # Match T2V model-selection semantics.
        skip_loras = model_type in ("fast", "quality")

        gguf_path_for_mode = preferred_gguf_path
        if model_type == "quality" and gguf_path_for_mode is not None:
            dev_checkpoint = _find_dev_checkpoint_candidate(self.models_dir)
            if dev_checkpoint is not None:
                logger.info(
                    "A2V quality mode prefers dev safetensors over GGUF: %s",
                    dev_checkpoint,
                )
                checkpoint_path = str(dev_checkpoint)
                gguf_path_for_mode = None

        if model_type == "fast":
            gguf_path_for_mode = self._resolve_distilled_gguf_path(
                preferred_gguf_path, "Q4_K_M"
            )
            if gguf_path_for_mode is None:
                logger.warning(
                    "A2V fast mode selected but no distilled GGUF found; using checkpoint path"
                )
        elif model_type == "balanced":
            preferred_quant = "Q8_0"
            preferred_model = (
                self.state.app_settings.preferred_model_path.strip().lower()
            )
            preferred_gguf = (
                preferred_gguf_path.lower() if preferred_gguf_path is not None else ""
            )
            for quant in (
                "Q8_0",
                "Q6_K",
                "Q5_1",
                "Q5_0",
                "Q4_K_M",
                "Q4_K_S",
                "Q4_1",
                "Q4_0",
                "Q3_K_M",
                "Q2_K",
            ):
                if quant.lower() in preferred_model or quant.lower() in preferred_gguf:
                    preferred_quant = quant
                    break

            dev_gguf = _find_dev_gguf_candidate(self.models_dir, preferred_quant)
            if dev_gguf is not None:
                gguf_path_for_mode = str(dev_gguf)
                logger.info(
                    "A2V balanced mode prefers dev GGUF with distilled LoRA: %s",
                    dev_gguf,
                )
            else:
                dev_checkpoint = _find_dev_checkpoint_candidate(self.models_dir)
                if dev_checkpoint is not None:
                    checkpoint_path = str(dev_checkpoint)
                    gguf_path_for_mode = None
                    logger.info(
                        "A2V balanced mode prefers dev safetensors with distilled LoRA: %s",
                        dev_checkpoint,
                    )

        if (
            gguf_path_for_mode is not None
            and _looks_like_incomplete_transformer_checkpoint(Path(checkpoint_path))
        ):
            full_checkpoint = _find_full_checkpoint_candidate(self.models_dir)
            if full_checkpoint is not None:
                logger.warning(
                    "A2V GGUF selected; using full checkpoint %s for VAE/audio/vocoder components",
                    full_checkpoint,
                )
                checkpoint_path = str(full_checkpoint)

        primary_lora_path, primary_lora_strength, extra_loras = (
            self._resolve_pipeline_loras(
                model_type=model_type,
                skip_loras=skip_loras,
                preferred_checkpoint_path=checkpoint_path,
                preferred_gguf_path=gguf_path_for_mode,
            )
        )

        if model_type in ("fast", "balanced"):
            num_inference_steps = 8
        elif model_type == "quality":
            num_inference_steps = self.state.app_settings.pro_model.steps
        else:
            num_inference_steps = self.state.app_settings.custom_model.steps

        preferred_text_encoder_path = (
            self.state.app_settings.preferred_text_encoder_path.strip()
        )
        text_encoder_variant_path = None
        if preferred_text_encoder_path:
            text_encoder_variant_path = _resolve_text_encoder_variant_path(
                self.models_dir,
                preferred_text_encoder_path,
            )

        if model_type in ("fast", "balanced"):
            use_upscaler = self.state.app_settings.fast_model.use_upscaler
        elif model_type == "quality":
            use_upscaler = self.state.app_settings.pro_model.use_upscaler
        else:
            use_upscaler = self.state.app_settings.custom_model.use_upscaler
        a2v_decode_tiling = self.state.app_settings.a2v_decode_tiling

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(
                    active_pipeline=A2VPipelineState(
                        model_type=current_model_type,
                        checkpoint_path=current_checkpoint_path,
                        gguf_path=current_gguf_path,
                        lora_path=current_lora_path,
                        num_inference_steps=current_num_inference_steps,
                        use_upscaler=current_use_upscaler,
                        a2v_decode_tiling=current_a2v_decode_tiling,
                    ) as state
                ) if (
                    current_model_type == model_type
                    and current_checkpoint_path == checkpoint_path
                    and current_gguf_path == gguf_path_for_mode
                    and current_lora_path == primary_lora_path
                    and current_num_inference_steps == num_inference_steps
                    and current_use_upscaler == use_upscaler
                    and current_a2v_decode_tiling == a2v_decode_tiling
                ):
                    return state
                case _:
                    pass

        pipeline = self._a2v_pipeline_class.create(
            checkpoint_path,
            self._text_handler.resolve_gemma_root(),
            str(
                resolve_model_path(
                    self.models_dir, self.config.model_download_specs, "upsampler"
                )
            ),
            self.config.device,
            vram_manager=vram_manager,
            use_sage_attention=self.config.use_sage_attention,
            gguf_path=gguf_path_for_mode,
            lora_path=primary_lora_path,
            lora_strength=primary_lora_strength,
            extra_loras=extra_loras,
            num_inference_steps=num_inference_steps,
            text_encoder_variant_path=text_encoder_variant_path,
            use_upscaler=use_upscaler,
            a2v_decode_tiling=a2v_decode_tiling,
        )
        state = A2VPipelineState(
            pipeline=pipeline,
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            gguf_path=gguf_path_for_mode,
            lora_path=primary_lora_path,
            num_inference_steps=num_inference_steps,
            use_upscaler=use_upscaler,
            a2v_decode_tiling=a2v_decode_tiling,
        )

        logger.info(
            "Pipeline selection: kind=a2v mode=%s checkpoint=%s gguf=%s loras=%s steps=%s upscaler=%s",
            model_type,
            _display_model_path(checkpoint_path),
            _display_model_path(gguf_path_for_mode),
            _display_lora_selection(
                primary_lora_path,
                primary_lora_strength,
                extra_loras,
            ),
            num_inference_steps,
            use_upscaler,
        )

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state, generation=None)
            self._assert_invariants()
        return state

    def load_retake_pipeline(self, *, distilled: bool = True) -> RetakePipelineState:
        self._install_text_patches_if_needed()

        vram_gb = 0
        if self._gpu_info is not None:
            vram_gb = self._gpu_info.get_vram_total_gb() or 0
        vram_manager = VRAMManager(
            self.config.device,
            vram_gb,
            user_blocks_on_gpu=self.state.app_settings.num_blocks_to_swap,
            user_run_mode=self.state.app_settings.run_mode,
        )

        quantized = device_supports_fp8(self.config.device)
        preferred_checkpoint_path, _ = self._resolve_preferred_model_paths()
        checkpoint_path = preferred_checkpoint_path or str(
            resolve_model_path(
                self.models_dir, self.config.model_download_specs, "checkpoint"
            )
        )

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(
                    active_pipeline=RetakePipelineState(
                        distilled=current_distilled,
                        quantized=current_quantized,
                        checkpoint_path=current_checkpoint_path,
                    ) as state
                ) if (
                    current_distilled == distilled
                    and current_quantized == quantized
                    and current_checkpoint_path == checkpoint_path
                ):
                    return state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()

        from ltx_core.quantization import QuantizationPolicy

        quantization = QuantizationPolicy.fp8_cast() if quantized else None
        preferred_text_encoder_path = (
            self.state.app_settings.preferred_text_encoder_path.strip()
        )
        text_encoder_variant_path = None
        if preferred_text_encoder_path:
            text_encoder_variant_path = _resolve_text_encoder_variant_path(
                self.models_dir,
                preferred_text_encoder_path,
            )
        pipeline = self._retake_pipeline_class.create(
            checkpoint_path=checkpoint_path,
            gemma_root=self._text_handler.resolve_gemma_root(),
            device=self.config.device,
            loras=[],
            quantization=quantization,
            vram_manager=vram_manager,
            text_encoder_variant_path=text_encoder_variant_path,
        )
        state = RetakePipelineState(
            pipeline=pipeline,
            distilled=distilled,
            quantized=quantized,
            checkpoint_path=checkpoint_path,
        )

        logger.info(
            "Pipeline selection: kind=retake distilled=%s quantized=%s checkpoint=%s loras=%s",
            distilled,
            quantized,
            _display_model_path(checkpoint_path),
            "none",
        )

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state, generation=None)
            self._assert_invariants()
        return state

    def warmup_pipeline(self, model_type: VideoPipelineModelType) -> None:
        state = self.load_gpu_pipeline(model_type, should_warm=False)
        warmup_path = self.config.outputs_dir / f"_warmup_{model_type}.mp4"
        state.pipeline.warmup(output_path=str(warmup_path))
