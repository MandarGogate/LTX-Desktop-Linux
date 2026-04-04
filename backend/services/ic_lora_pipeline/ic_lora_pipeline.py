"""IC-LoRA pipeline protocol definitions."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from api_types import ImageConditioningInput

if TYPE_CHECKING:
    import torch


IcLoraProgressCallback = Callable[[str, int | None, int | None], None]


class IcLoraPipeline(Protocol):
    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        lora_path: str,
        device: torch.device,
        vram_manager: Any | None = None,
    ) -> "IcLoraPipeline": ...

    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        output_path: str,
        source_audio_path: str | None = None,
        *,
        source_audio_start_time: float = 0.0,
        source_audio_max_duration: float | None = None,
        skip_stage_2: bool = False,
        progress_callback: IcLoraProgressCallback | None = None,
    ) -> None: ...
