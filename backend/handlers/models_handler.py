"""Model availability and model status handlers."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from api_types import ModelFileStatus, ModelInfo, ModelReadinessResponse, ModelsStatusResponse, SuggestedModel, TextEncoderStatus
from handlers.base import StateHandlerBase, with_state_lock
from runtime_config.model_download_specs import MODEL_FILE_ORDER, resolve_model_path, resolve_required_model_types
from state.app_state_types import AppState, AvailableFiles, ModelFileType

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


class ModelsHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)

    def _find_text_encoder_variant(self) -> Path | None:
        preferred = self.state.app_settings.preferred_text_encoder_path.strip()
        if preferred:
            preferred_path = self.models_dir / preferred
            if preferred_path.exists() and preferred_path.is_file():
                return preferred_path

        variants_dir = self.models_dir / "text_encoders"
        if not variants_dir.exists():
            return None
        for path in sorted(variants_dir.iterdir(), key=lambda item: item.name.lower()):
            if path.is_file() and path.suffix in {".safetensors", ".gguf"}:
                return path
        return None

    def _find_zit_variant(self) -> Path | None:
        preferred = self.state.app_settings.preferred_zit_model_path.strip()
        if not preferred:
            return None

        preferred_path = Path(preferred)
        if preferred_path.is_absolute():
            return preferred_path if preferred_path.exists() and preferred_path.is_file() else None

        resolved = self.models_dir / preferred
        return resolved if resolved.exists() and resolved.is_file() else None

    @staticmethod
    def _path_size(path: Path, is_folder: bool) -> int:
        if not is_folder:
            return path.stat().st_size
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    def _scan_available_files(self) -> AvailableFiles:
        files: AvailableFiles = {}
        for model_type in MODEL_FILE_ORDER:
            spec = self.config.spec_for(model_type)
            path = resolve_model_path(self.models_dir, self.config.model_download_specs, model_type)
            if spec.is_folder:
                ready = path.exists() and any(path.iterdir()) if path.exists() else False
                files[model_type] = path if ready else None
            else:
                files[model_type] = path if path.exists() else None
        return files

    @with_state_lock
    def refresh_available_files(self) -> AvailableFiles:
        self.state.available_files = self._scan_available_files()
        return self.state.available_files.copy()

    def get_text_encoder_status(self) -> TextEncoderStatus:
        files = self.refresh_available_files()
        text_encoder_path = files["text_encoder"]
        variant_path = self._find_text_encoder_variant()
        exists = text_encoder_path is not None or variant_path is not None
        text_spec = self.config.spec_for("text_encoder")
        if text_encoder_path is not None:
            size_bytes = self._path_size(text_encoder_path, is_folder=True)
        elif variant_path is not None:
            size_bytes = variant_path.stat().st_size
        else:
            size_bytes = 0
        expected = text_spec.expected_size_bytes

        return TextEncoderStatus(
            downloaded=exists,
            size_bytes=size_bytes if exists else expected,
            size_gb=round((size_bytes if exists else expected) / (1024**3), 1),
            expected_size_gb=round(expected / (1024**3), 1),
        )

    def get_models_list(self) -> list[ModelInfo]:
        pro_steps = self.state.app_settings.pro_model.steps
        pro_upscaler = self.state.app_settings.pro_model.use_upscaler
        return [
            ModelInfo(id="fast", name="LTX 2.3 Fast", description="Distilled base model, 8 steps, no LoRA"),
            ModelInfo(id="balanced", name="LTX 2.3 Balanced", description="Dev model + distilled LoRA, 8 steps"),
            ModelInfo(
                id="quality",
                name="LTX 2.3 Quality",
                description=f"Dev model, {pro_steps} steps" + (" + 2x upscaler" if pro_upscaler else ""),
            ),
            ModelInfo(
                id="custom",
                name="LTX 2.3 Custom",
                description=f"Uses selected models from Settings, {self.state.app_settings.custom_model.steps} steps",
            ),
        ]

    def _has_any_diffusion_model(self) -> bool:
        """Check if any usable diffusion model exists (GGUF or safetensors)."""
        dm = self.models_dir / "diffusion_models"
        if dm.exists():
            for f in dm.rglob("*.gguf"):
                name = f.name.lower()
                if "z-image" not in name and "zimage" not in name:
                    return True
            for f in dm.rglob("*.safetensors"):
                return True
        # Legacy locations
        gguf_dir = self.models_dir / "gguf"
        if gguf_dir.exists():
            for f in gguf_dir.rglob("*.gguf"):
                name = f.name.lower()
                if "z-image" not in name and "zimage" not in name:
                    return True
        # Check default checkpoint path
        files = self._scan_available_files()
        if files.get("checkpoint") is not None:
            return True
        if files.get("gguf_checkpoint") is not None:
            return True
        return False

    def _has_any_text_encoder(self) -> bool:
        """Check if any usable text encoder exists."""
        te_dir = self.models_dir / "text_encoders"
        if te_dir.exists():
            for f in te_dir.iterdir():
                if f.is_file() and (f.suffix == ".safetensors" or f.suffix == ".gguf"):
                    return True
                if f.is_dir() and any(f.iterdir()):
                    return True
        # Check default text encoder path
        files = self._scan_available_files()
        if files.get("text_encoder") is not None:
            return True
        return False

    def _has_upscaler(self) -> bool:
        """Check if upscaler exists."""
        um = self.models_dir / "upscale_models"
        if um.exists():
            for f in um.rglob("*.safetensors"):
                return True
        files = self._scan_available_files()
        return files.get("upsampler") is not None

    def _has_split_decode_components(self) -> bool:
        vae_dir = self.models_dir / "vae"
        video_candidates = (
            vae_dir / "LTX23_video_vae_bf16.safetensors",
            vae_dir / "LTX2_video_vae_bf16.safetensors",
        )
        audio_candidates = (
            vae_dir / "LTX23_audio_vae_bf16.safetensors",
            vae_dir / "LTX2_audio_vae_bf16.safetensors",
        )
        return any(path.exists() for path in video_candidates) and any(path.exists() for path in audio_candidates)

    def _has_full_decode_checkpoint(self) -> bool:
        search_roots = (
            self.models_dir / "diffusion_models",
            self.models_dir,
        )
        for root in search_roots:
            if not root.exists():
                continue
            for path in root.rglob("*.safetensors"):
                if not path.is_file():
                    continue
                try:
                    if path.stat().st_size > 10_000_000_000:
                        return True
                except OSError:
                    logger.warning("Failed to inspect checkpoint size for %s", path, exc_info=True)
        return False

    def _has_local_decode_components(self) -> bool:
        return self._has_full_decode_checkpoint() or self._has_split_decode_components()

    def _has_matching_diffusion_gguf(self, *, distilled: bool) -> bool:
        search_roots = (
            self.models_dir / "diffusion_models",
            self.models_dir / "gguf",
        )
        needle = "distilled" if distilled else "dev"
        for root in search_roots:
            if not root.exists():
                continue
            for path in root.rglob("*.gguf"):
                name = path.name.lower()
                if "z-image" in name or "zimage" in name or "z_image" in name:
                    continue
                if needle in name:
                    return True
        return False

    def _has_default_distilled_lora(self) -> bool:
        candidates = (
            self.models_dir / "loras" / "ltx-2.3-22b-distilled-lora-384.safetensors",
            self.models_dir / "ltx-2.3-22b-distilled-lora-384.safetensors",
            self.models_dir / "loras" / "ltx-2-19b-distilled-lora-384.safetensors",
        )
        return any(path.exists() for path in candidates)

    @staticmethod
    def _recommended_quant(vram_gb: int | None) -> str:
        if vram_gb is not None and vram_gb >= 24:
            return "Q8_0"
        if vram_gb is not None and vram_gb >= 12:
            return "Q4_K_M"
        return "Q4_0"

    def get_model_readiness(self, vram_gb: int | None, gpu_name: str | None) -> ModelReadinessResponse:
        """Check if the app can generate and suggest GPU-appropriate models."""
        has_diffusion = self._has_any_diffusion_model()
        has_fast_base = self._has_matching_diffusion_gguf(distilled=True)
        has_dev_base = self._has_matching_diffusion_gguf(distilled=False)
        has_distilled_lora = self._has_default_distilled_lora()
        has_te = self._has_any_text_encoder()
        has_upscaler = self._has_upscaler()
        has_decode = self._has_local_decode_components()
        can_generate = has_fast_base and has_te and has_decode

        suggestions: list[SuggestedModel] = []
        total_gb = 0.0

        for model in self._suggest_video_bundle(
            vram_gb=vram_gb,
            need_fast_base=not has_fast_base,
            need_dev_base=not has_dev_base,
            need_distilled_lora=not has_distilled_lora,
        ):
            suggestions.append(model)
            total_gb += model.size_gb

        if not has_te:
            model = self._suggest_text_encoder(vram_gb)
            suggestions.append(model)
            total_gb += model.size_gb

        if not has_decode:
            for model in self._suggest_split_decode_components():
                suggestions.append(model)
                total_gb += model.size_gb

        if not has_upscaler:
            suggestions.append(SuggestedModel(
                id="upscaler",
                filename="ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
                repo_id="Lightricks/LTX-2.3",
                description="2x spatial upscaler",
                size_gb=1.9,
                target_subdir="upscale_models",
                category="upscaler",
            ))
            total_gb += 1.9

        return ModelReadinessResponse(
            can_generate=can_generate,
            has_diffusion_model=has_diffusion,
            has_text_encoder=has_te,
            has_upscaler=has_upscaler,
            vram_gb=vram_gb,
            gpu_name=gpu_name,
            suggested_models=suggestions,
            total_download_gb=round(total_gb, 1),
        )

    def _suggest_video_bundle(
        self,
        *,
        vram_gb: int | None,
        need_fast_base: bool,
        need_dev_base: bool,
        need_distilled_lora: bool,
    ) -> list[SuggestedModel]:
        quant = self._recommended_quant(vram_gb)
        distilled_sizes = {"Q8_0": 12.3, "Q4_K_M": 7.4, "Q4_0": 6.5}
        dev_sizes = {"Q8_0": 22.8, "Q4_K_M": 14.3, "Q4_0": 12.7}

        suggestions: list[SuggestedModel] = []
        if need_fast_base:
            suggestions.append(SuggestedModel(
                id=f"fast-{quant.lower()}",
                filename=f"distilled/ltx-2.3-22b-distilled-{quant}.gguf",
                repo_id="unsloth/LTX-2.3-GGUF",
                description=f"Fast mode distilled GGUF ({quant})",
                size_gb=distilled_sizes[quant],
                target_subdir="diffusion_models",
                quant_level=quant,
                category="diffusion",
            ))
        if need_dev_base:
            suggestions.append(SuggestedModel(
                id=f"dev-{quant.lower()}",
                filename=f"ltx-2.3-22b-dev-{quant}.gguf",
                repo_id="unsloth/LTX-2.3-GGUF",
                description=f"Balanced/Quality dev GGUF ({quant})",
                size_gb=dev_sizes[quant],
                target_subdir="diffusion_models",
                quant_level=quant,
                category="diffusion",
            ))
        if need_distilled_lora:
            suggestions.append(SuggestedModel(
                id="distilled-lora",
                filename="ltx-2.3-22b-distilled-lora-384.safetensors",
                repo_id="Lightricks/LTX-2",
                description="Distilled LoRA for Balanced mode",
                size_gb=0.4,
                target_subdir="loras",
                quant_level=None,
                category="lora",
            ))
        return suggestions

    @staticmethod
    def _suggest_text_encoder(vram_gb: int | None) -> SuggestedModel:
        """Suggest the best text encoder for the user's GPU."""
        quant = ModelsHandler._recommended_quant(vram_gb)
        sizes = {"Q8_0": 8.2, "Q4_K_M": 4.8, "Q4_0": 4.2}
        return SuggestedModel(
            id=f"te-{quant.lower()}",
            filename=f"gemma-3-12b-it-{quant}.gguf",
            repo_id="unsloth/gemma-3-12b-it-GGUF",
            description=f"Gemma text encoder GGUF ({quant})",
            size_gb=sizes[quant],
            target_subdir="text_encoders",
            quant_level=quant,
            category="text_encoder",
        )

    @staticmethod
    def _suggest_split_decode_components() -> list[SuggestedModel]:
        return [
            SuggestedModel(
                id="ltx23-video-vae",
                filename="vae/LTX23_video_vae_bf16.safetensors",
                repo_id="Kijai/LTX2.3_comfy",
                description="Split video VAE for local GGUF decode",
                size_gb=1.45,
                target_subdir="vae",
                quant_level="BF16",
                category="vae",
            ),
            SuggestedModel(
                id="ltx23-audio-vae",
                filename="vae/LTX23_audio_vae_bf16.safetensors",
                repo_id="Kijai/LTX2.3_comfy",
                description="Split audio VAE/vocoder for local GGUF decode",
                size_gb=0.36,
                target_subdir="vae",
                quant_level="BF16",
                category="vae",
            ),
        ]

    @with_state_lock
    def get_required_model_types(self, skip_text_encoder: bool = False) -> list[ModelFileType]:
        settings = self.state.app_settings
        required = resolve_required_model_types(
            self._config.required_model_types,
            has_api_key=False,
            use_local_text_encoder=True,
        )
        text_encoder_satisfied = self._find_text_encoder_variant() is not None or self.state.available_files.get("text_encoder") is not None
        zit_satisfied = self._find_zit_variant() is not None or self.state.available_files.get("zit") is not None
        return [
            model_type
            for model_type in MODEL_FILE_ORDER
            if model_type in required
            and not (skip_text_encoder and model_type == "text_encoder")
            and not (model_type == "text_encoder" and text_encoder_satisfied)
            and not (model_type == "zit" and zit_satisfied)
        ]

    def get_models_status(self, has_api_key: bool | None = None) -> ModelsStatusResponse:
        files = self.refresh_available_files()
        settings = self.state.app_settings.model_copy(deep=True)

        has_api_key = False

        models: list[ModelFileStatus] = []
        total_size = 0
        downloaded_size = 0
        required_types = resolve_required_model_types(
            self.config.required_model_types,
            has_api_key=False,
            use_local_text_encoder=True,
        )
        variant_path = self._find_text_encoder_variant()
        zit_variant_path = self._find_zit_variant()

        for model_type in MODEL_FILE_ORDER:
            spec = self.config.spec_for(model_type)
            path = files[model_type]
            exists = path is not None
            actual_size = self._path_size(path, is_folder=spec.is_folder) if exists else 0
            required = model_type in required_types
            if model_type == "text_encoder" and variant_path is not None:
                exists = True
                actual_size = variant_path.stat().st_size
            if model_type == "zit" and zit_variant_path is not None:
                exists = True
                actual_size = zit_variant_path.stat().st_size
            if required:
                total_size += spec.expected_size_bytes
                if exists:
                    downloaded_size += actual_size

            description = spec.description
            optional_reason: str | None = None
            if model_type == "text_encoder" and variant_path is not None:
                description = f"Quantized text encoder available ({variant_path.name})"
                required = False
                optional_reason = "Satisfied by downloaded quantized text encoder"
            if model_type == "zit" and zit_variant_path is not None:
                description = f"GGUF Z-Image-Turbo available ({zit_variant_path.name})"
                required = False
                optional_reason = "Satisfied by downloaded GGUF Z-Image-Turbo model"

            models.append(
                ModelFileStatus(
                    id=model_type,
                    name=spec.name,
                    description=description,
                    downloaded=exists,
                    size=actual_size if exists else spec.expected_size_bytes,
                    expected_size=spec.expected_size_bytes,
                    required=required,
                    is_folder=spec.is_folder,
                    optional_reason=optional_reason if model_type in {"text_encoder", "zit"} else None,
                )
            )

        all_downloaded = all(model.downloaded for model in models if model.required)

        return ModelsStatusResponse(
            models=models,
            all_downloaded=all_downloaded,
            total_size=total_size,
            downloaded_size=downloaded_size,
            total_size_gb=round(total_size / (1024**3), 1),
            downloaded_size_gb=round(downloaded_size / (1024**3), 1),
            models_path=str(self.models_dir),
            has_api_key=has_api_key,
            text_encoder_status=self.get_text_encoder_status(),
            use_local_text_encoder=settings.use_local_text_encoder,
        )
