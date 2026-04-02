"""Image generation pipeline protocol definitions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from services.services_utils import ImagePipelineOutputLike


class ImageGenerationPipeline(Protocol):
    @staticmethod
    def create(
        model_path: str,
        device: str | None = None,
    ) -> "ImageGenerationPipeline":
        ...

    def generate(
        self,
        prompt: str,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ImagePipelineOutputLike:
        ...

    def to(self, device: str) -> None:
        ...
