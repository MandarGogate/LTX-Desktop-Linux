"""LTX IC-LoRA pipeline wrapper."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, cast

import torch

from api_types import ImageConditioningInput
from services.ltx_pipeline_common import default_tiling_config, encode_video_output, video_chunks_number
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8


def _move_value_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        items = cast(tuple[Any, ...], value)
        return tuple(_move_value_to_device(item, device) for item in items)
    if isinstance(value, list):
        items = cast(list[Any], value)
        return [_move_value_to_device(item, device) for item in items]
    if isinstance(value, dict):
        entries = cast(dict[object, Any], value)
        return {key: _move_value_to_device(item, device) for key, item in entries.items()}
    return value


def _force_module_to_device(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Best-effort device fixup for upstream IC-LoRA models.

    Some upstream IC-LoRA builds can retain nested linear weights on CPU even
    after the top-level module is moved to CUDA, which later fails inside the
    transformer patchify projection. Re-apply .to(device) recursively and also
    fix any nested torch.Tensor / nn.Module attributes that are not registered
    as parameters or buffers.
    """
    module = module.to(device)

    for child in module.modules():
        child.to(device)
        for value in vars(child).values():
            if isinstance(value, torch.nn.Module):
                value.to(device)
            elif isinstance(value, torch.Tensor):
                value.data = value.data.to(device)

    return module


def _wrap_model_factory(method: Callable[[], Any], device: torch.device) -> Callable[[], Any]:
    def _wrapped() -> Any:
        model = method()
        if isinstance(model, torch.nn.Module):
            return _force_module_to_device(model, device)
        return model

    return _wrapped


class LTXIcLoraPipeline:
    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        lora_path: str,
        device: torch.device,
    ) -> "LTXIcLoraPipeline":
        return LTXIcLoraPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            lora_path=lora_path,
            device=device,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        lora_path: str,
        device: torch.device,
    ) -> None:
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.ic_lora import ICLoraPipeline

        self.device = device

        lora_entry = LoraPathStrengthAndSDOps(path=lora_path, strength=1.0, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
        self.pipeline = ICLoraPipeline(
            distilled_checkpoint_path=checkpoint_path,
            spatial_upsampler_path=upsampler_path,
            gemma_root=cast(str, gemma_root),
            loras=[lora_entry],
            device=device,
            quantization=QuantizationPolicy.fp8_cast() if device_supports_fp8(device) else None,
        )

        self.pipeline.stage_1_model_ledger.transformer = _wrap_model_factory(
            self.pipeline.stage_1_model_ledger.transformer,
            device,
        )
        self.pipeline.stage_2_model_ledger.transformer = _wrap_model_factory(
            self.pipeline.stage_2_model_ledger.transformer,
            device,
        )

        import ltx_pipelines.ic_lora as ic_lora_module

        original_encode_text = ic_lora_module.encode_text

        def _encode_text_on_device(*args: Any, **kwargs: Any) -> Any:
            return _move_value_to_device(original_encode_text(*args, **kwargs), device)

        ic_lora_module.encode_text = _encode_text_on_device

    def _run_inference(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        tiling_config: TilingConfigType,
        *,
        skip_stage_2: bool = False,
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput

        return self.pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=[_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images],
            video_conditioning=video_conditioning,
            tiling_config=tiling_config,
            skip_stage_2=skip_stage_2,
        )

    @torch.inference_mode()
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
    ) -> None:
        source_audio: AudioOrNone = None
        if source_audio_path is not None:
            from ltx_core.types import Audio
            from ltx_pipelines.utils.media_io import decode_audio_from_file

            decoded_audio = decode_audio_from_file(
                source_audio_path,
                self.device,
                source_audio_start_time,
                source_audio_max_duration,
            )
            if decoded_audio is not None:
                max_samples = round(num_frames / frame_rate * decoded_audio.sampling_rate)
                trimmed_waveform = decoded_audio.waveform.squeeze(0)[..., :max_samples]
                source_audio = Audio(waveform=trimmed_waveform.detach().cpu(), sampling_rate=decoded_audio.sampling_rate)

        tiling_config = default_tiling_config()
        video, generated_audio = self._run_inference(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            video_conditioning=video_conditioning,
            tiling_config=tiling_config,
            skip_stage_2=skip_stage_2,
        )
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(
            video=video,
            audio=source_audio if source_audio is not None else generated_audio,
            fps=int(frame_rate),
            output_path=output_path,
            video_chunks_number_value=chunks,
        )
