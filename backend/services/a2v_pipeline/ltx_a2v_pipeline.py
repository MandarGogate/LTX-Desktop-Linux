"""LTX A2V (Audio-to-Video) pipeline wrapper.

Keeps the A2V-specific public interface while delegating local execution to the
same low-VRAM/block-swap implementation used by the working T2V/I2V paths.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from api_types import ImageConditioningInput


class LTXa2vPipeline:
    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        vram_manager: Any | None = None,
        *,
        use_sage_attention: bool = True,
        gguf_path: str | None = None,
        lora_path: str | None = None,
        lora_strength: float = 1.0,
        extra_loras: list[tuple[str, float]] | None = None,
        num_inference_steps: int | None = None,
        text_encoder_variant_path: str | None = None,
        use_upscaler: bool = False,
    ) -> "LTXa2vPipeline":
        return LTXa2vPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            vram_manager=vram_manager,
            use_sage_attention=use_sage_attention,
            gguf_path=gguf_path,
            lora_path=lora_path,
            lora_strength=lora_strength,
            extra_loras=extra_loras,
            num_inference_steps=num_inference_steps,
            text_encoder_variant_path=text_encoder_variant_path,
            use_upscaler=use_upscaler,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        vram_manager: Any | None = None,
        *,
        use_sage_attention: bool = True,
        gguf_path: str | None = None,
        lora_path: str | None = None,
        lora_strength: float = 1.0,
        extra_loras: list[tuple[str, float]] | None = None,
        num_inference_steps: int | None = None,
        text_encoder_variant_path: str | None = None,
        use_upscaler: bool = False,
    ) -> None:
        from services.fast_video_pipeline.ltx_low_vram_pipeline import (
            LTXLowVRAMPipeline,
        )

        self.pipeline = LTXLowVRAMPipeline.create(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            vram_manager=vram_manager,
            gguf_path=gguf_path,
            lora_path=lora_path,
            lora_strength=lora_strength,
            extra_loras=extra_loras,
            use_sage_attention=use_sage_attention,
            num_inference_steps=num_inference_steps,
            text_encoder_variant_path=text_encoder_variant_path,
            use_upscaler=use_upscaler,
        )

    @torch.inference_mode()
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
    ) -> None:
        # Respect per-request stage-1 step count for dev A2V schedules.
        self.pipeline._num_inference_steps = int(num_inference_steps)
        self.pipeline.generate_a2v(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            output_path=output_path,
            audio_path=audio_path,
            audio_start_time=audio_start_time,
            audio_max_duration=audio_max_duration,
            progress_callback=progress_callback,
        )
