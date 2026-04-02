"""A2V (Audio-to-Video) pipeline protocol definitions."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from api_types import ImageConditioningInput

if TYPE_CHECKING:
    import torch


class A2VPipeline(Protocol):
    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        vram_manager: object | None = None,
        *,
        use_sage_attention: bool = True,
        gguf_path: str | None = None,
        lora_path: str | None = None,
        lora_strength: float = 1.0,
        extra_loras: list[tuple[str, float]] | None = None,
        num_inference_steps: int | None = None,
        text_encoder_variant_path: str | None = None,
        use_upscaler: bool = False,
    ) -> "A2VPipeline": ...

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        images: list[ImageConditioningInput],
        audio_path: str,
        audio_start_time: float,
        audio_max_duration: float | None,
        output_path: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None: ...
