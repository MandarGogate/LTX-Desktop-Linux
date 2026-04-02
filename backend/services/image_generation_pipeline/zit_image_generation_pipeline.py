"""Z-Image-Turbo image generation pipeline wrapper.

Supports loading from:
- A pretrained model directory (safetensors)
- A single GGUF file (transformer loaded via GGUFQuantizationConfig,
  remaining components fetched from the HuggingFace repo)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from diffusers.pipelines.auto_pipeline import ZImagePipeline  # type: ignore[reportUnknownVariableType]
from PIL.Image import Image as PILImage

from services.services_utils import ImagePipelineOutputLike, PILImageType, get_device_type

logger = logging.getLogger(__name__)

# HuggingFace repo used to fetch non-transformer components when loading from GGUF.
_ZIT_BASE_REPO = "Tongyi-MAI/Z-Image-Turbo"


@dataclass(slots=True)
class _ZImageOutput:
    images: Sequence[PILImageType]


def _is_gguf_path(model_path: str) -> bool:
    return model_path.lower().endswith(".gguf")


def _load_pipeline_from_gguf(gguf_path: str) -> Any:
    """Load ZImagePipeline with transformer weights from a GGUF file."""
    from diffusers import GGUFQuantizationConfig  # type: ignore[reportUnknownVariableType]
    from diffusers.models import ZImageTransformer2DModel  # type: ignore[reportUnknownVariableType]

    logger.info("Loading ZIT transformer from GGUF: %s", gguf_path)
    transformer = ZImageTransformer2DModel.from_single_file(  # type: ignore[reportUnknownMemberType]
        gguf_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
    )

    logger.info("Loading ZIT pipeline components from %s", _ZIT_BASE_REPO)
    pipeline = ZImagePipeline.from_pretrained(  # type: ignore[reportUnknownMemberType]
        _ZIT_BASE_REPO,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    return pipeline


class ZitImageGenerationPipeline:
    @staticmethod
    def create(
        model_path: str,
        device: str | None = None,
    ) -> "ZitImageGenerationPipeline":
        return ZitImageGenerationPipeline(model_path=model_path, device=device)

    def __init__(self, model_path: str, device: str | None = None) -> None:
        self._device: str | None = None
        self._cpu_offload_active = False

        if _is_gguf_path(model_path):
            self.pipeline = _load_pipeline_from_gguf(model_path)
        else:
            self.pipeline = ZImagePipeline.from_pretrained(  # type: ignore[reportUnknownMemberType]
                model_path,
                torch_dtype=torch.bfloat16,
            )

        if device is not None:
            self.to(device)

    def _resolve_generator_device(self) -> str:
        if self._cpu_offload_active:
            return "cuda"
        if self._device is not None:
            return self._device

        execution_device = getattr(self.pipeline, "_execution_device", None)
        return get_device_type(execution_device)

    @staticmethod
    def _normalize_output(output: object) -> ImagePipelineOutputLike:
        images = getattr(output, "images", None)
        if not isinstance(images, Sequence):
            raise RuntimeError("Unexpected ZIT pipeline output format: missing images sequence")

        images_list = cast(Sequence[object], images)
        validated_images: list[PILImageType] = []
        for image in images_list:
            if not isinstance(image, PILImage):
                raise RuntimeError("Unexpected ZIT pipeline output format: images must be PIL.Image instances")
            validated_images.append(image)

        return _ZImageOutput(images=validated_images)

    @torch.inference_mode()
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
        # ZImagePipeline ignores guidance_scale, so we drop it explicitly.
        _ = guidance_scale
        generator = torch.Generator(device=self._resolve_generator_device()).manual_seed(seed)
        pipeline = cast(Any, self.pipeline)

        callback_on_step_end = None
        if progress_callback is not None:
            def _on_step_end(pipe: Any, step_index: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
                del pipe, timestep
                progress_callback(step_index + 1, num_inference_steps)
                return callback_kwargs
            callback_on_step_end = _on_step_end

        output = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=0.0,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="pil",
            return_dict=True,
            callback_on_step_end=callback_on_step_end,
        )
        return self._normalize_output(output)

    def to(self, device: str) -> None:
        runtime_device = get_device_type(device)
        if runtime_device in ("cuda", "mps"):
            self.pipeline.enable_model_cpu_offload()  # type: ignore[reportUnknownMemberType]
            self._cpu_offload_active = True
        else:
            self._cpu_offload_active = False
            self.pipeline.to(runtime_device)  # type: ignore[reportUnknownMemberType]
        self._device = runtime_device
