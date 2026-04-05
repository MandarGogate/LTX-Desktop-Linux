"""Retake pipeline protocol definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from collections.abc import Callable

if TYPE_CHECKING:
    import torch
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.loader import LoraPathStrengthAndSDOps
    from ltx_core.quantization import QuantizationPolicy
    from services.vram_manager.vram_manager import VRAMManager


class RetakePipeline(Protocol):
    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        device: "torch.device",
        *,
        loras: list["LoraPathStrengthAndSDOps"] | None = None,
        quantization: "QuantizationPolicy | None" = None,
        vram_manager: "VRAMManager | None" = None,
        text_encoder_variant_path: str | None = None,
    ) -> "RetakePipeline": ...

    def generate(
        self,
        *,
        video_path: str,
        prompt: str,
        start_time: float,
        end_time: float,
        seed: int,
        output_path: str,
        negative_prompt: str = "",
        num_inference_steps: int = 40,
        video_guider_params: "MultiModalGuiderParams | None" = None,
        audio_guider_params: "MultiModalGuiderParams | None" = None,
        regenerate_video: bool = True,
        regenerate_audio: bool = True,
        enhance_prompt: bool = False,
        distilled: bool = True,
        progress_callback: "Callable[[int, int], None] | None" = None,
    ) -> None: ...
