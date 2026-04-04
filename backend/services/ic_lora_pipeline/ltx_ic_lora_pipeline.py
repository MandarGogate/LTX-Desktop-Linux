"""LTX IC-LoRA pipeline wrapper with sequential offloading.

Mirrors the sequential-offload + block-swap pattern from
``ltx_low_vram_pipeline.py`` so that IC-LoRA works on 24 GB consumer GPUs.

The upstream ``ICLoraPipeline.__call__`` loads text-encoder, transformer, and
VAE simultaneously which OOMs on ≤24 GB.  We bypass ``__call__`` and drive each
phase (text encode → stage-1 denoise → stage-2 denoise → VAE decode) exactly
like the T2V low-VRAM pipeline: one big model on GPU at a time, with block
swap for the transformer.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import torch

from api_types import ImageConditioningInput
from services.ltx_pipeline_common import (
    default_tiling_config,
    encode_video_output,
    video_chunks_number,
)
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8

from .ic_lora_pipeline import IcLoraProgressCallback

if TYPE_CHECKING:
    from services.block_swap.fast_block_swap import FastBlockSwapWrapper
    from services.vram_manager.vram_manager import VRAMManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        return {
            key: _move_value_to_device(item, device) for key, item in entries.items()
        }
    return value


def _force_module_to_device(
    module: torch.nn.Module, device: torch.device
) -> torch.nn.Module:
    """Recursively move *everything* in a module to *device*."""
    module = module.to(device)
    for child in module.modules():
        child.to(device)
        for value in vars(child).values():
            if isinstance(value, torch.nn.Module):
                value.to(device)
            elif isinstance(value, torch.Tensor):
                value.data = value.data.to(device)
    return module


def _sync_and_cleanup(device: torch.device) -> None:
    """Synchronise GPU, collect garbage, empty CUDA cache."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _move_non_block_parts_to_gpu(
    transformer: torch.nn.Module, device: torch.device
) -> None:
    """Move everything *except* the repeating transformer blocks to GPU.

    Block-swap hooks manage the blocks; we just need the non-block parameters
    (embedding, norm, proj layers …) resident on GPU.
    """
    inner = transformer
    for sub_name in ("velocity_model", "model", "inner_model"):
        sub = getattr(inner, sub_name, None)
        if sub is not None:
            inner = sub
            break

    for _, param in inner.named_parameters(recurse=False):
        param.data = param.data.to(device)
    for _, buf in inner.named_buffers(recurse=False):
        buf.data = buf.data.to(device)

    block_attr_names = {"transformer_blocks", "blocks", "layers", "encoder_layers"}
    for child_name, child in inner.named_children():
        if child_name not in block_attr_names:
            child.to(device)

    if inner is not transformer:
        for _, param in transformer.named_parameters(recurse=False):
            param.data = param.data.to(device)
        for _, buf in transformer.named_buffers(recurse=False):
            buf.data = buf.data.to(device)
        for _, child in transformer.named_children():
            if child is not inner:
                child.to(device)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class LTXIcLoraPipeline:
    """IC-LoRA pipeline with sequential offloading + block swap."""

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        lora_path: str,
        device: torch.device,
        vram_manager: "VRAMManager | None" = None,
    ) -> "LTXIcLoraPipeline":
        return LTXIcLoraPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            lora_path=lora_path,
            device=device,
            vram_manager=vram_manager,
        )

    # ------------------------------------------------------------------
    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        lora_path: str,
        device: torch.device,
        vram_manager: "VRAMManager | None" = None,
    ) -> None:
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.ic_lora import ICLoraPipeline

        self.device = device
        self.vram_manager = vram_manager
        self._block_swap_wrapper: FastBlockSwapWrapper | None = None

        lora_entry = LoraPathStrengthAndSDOps(
            path=lora_path, strength=1.0, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
        )
        # Create the upstream pipeline on CPU so that ModelLedger factory
        # methods don't immediately move multi-GB models onto GPU.
        # We manage GPU placement ourselves phase-by-phase.
        self._cpu_device = torch.device("cpu")
        self.pipeline = ICLoraPipeline(
            distilled_checkpoint_path=checkpoint_path,
            spatial_upsampler_path=upsampler_path,
            gemma_root=cast(str, gemma_root),
            loras=[lora_entry],
            device=self._cpu_device,
            quantization=QuantizationPolicy.fp8_cast()
            if device_supports_fp8(device)
            else None,
        )

        # Monkey-patch encode_text so outputs land on *device*
        import ltx_pipelines.ic_lora as ic_lora_module
        original_encode_text = ic_lora_module.encode_text

        def _encode_text_on_device(*args: Any, **kwargs: Any) -> Any:
            return _move_value_to_device(original_encode_text(*args, **kwargs), device)

        ic_lora_module.encode_text = _encode_text_on_device

    # ------------------------------------------------------------------
    # Block-swap setup (same pattern as ltx_low_vram_pipeline)
    # ------------------------------------------------------------------

    def _setup_block_swap_if_needed(
        self, transformer: torch.nn.Module
    ) -> torch.nn.Module:
        from services.block_swap.fast_block_swap import FastBlockSwapWrapper
        from services.vram_manager.vram_manager import OffloadStrategy

        if self.vram_manager is None:
            return transformer

        if self.vram_manager.offload_strategy not in (
            OffloadStrategy.BLOCK_SWAP,
            OffloadStrategy.BLOCK_SWAP_AGGRESSIVE,
        ):
            return transformer

        if self._block_swap_wrapper is not None:
            self._block_swap_wrapper.restore_gpu_blocks()
            logger.info("[ic-lora] Reusing block swap wrapper")
            return transformer

        blocks_on_gpu = self.vram_manager.block_swap_keep_on_gpu
        logger.info(
            "[ic-lora] Setting up block swap: keeping %d blocks on GPU",
            blocks_on_gpu,
        )

        self._block_swap_wrapper = FastBlockSwapWrapper(
            transformer=transformer,
            device=self.device,
            blocks_to_keep_on_gpu=blocks_on_gpu,
            prefetch_distance=2,
        )
        if self._block_swap_wrapper.block_count == 0:
            logger.warning("[ic-lora] Block swap found 0 blocks; falling back")
            self._block_swap_wrapper = None

        return transformer

    def _prepare_transformer_for_denoise(
        self, transformer: torch.nn.Module
    ) -> torch.nn.Module:
        """Put transformer on GPU, using block swap when available."""
        if self.vram_manager is not None:
            self.vram_manager.cleanup()

        transformer = self._setup_block_swap_if_needed(transformer)

        if self._block_swap_wrapper is not None:
            self._block_swap_wrapper.offload_all()
            _move_non_block_parts_to_gpu(transformer, self.device)
            self._block_swap_wrapper.restore_gpu_blocks()
        else:
            _force_module_to_device(transformer, self.device)

        if self.vram_manager is not None:
            self.vram_manager.cleanup()
        return transformer

    def _offload_transformer(self, transformer: torch.nn.Module) -> None:
        """Move transformer off GPU after denoising."""
        if self._block_swap_wrapper is not None:
            self._block_swap_wrapper.offload_all()
        else:
            transformer.to("cpu")
        _sync_and_cleanup(self.device)

    # ------------------------------------------------------------------
    # Core sequential-offload generation
    # ------------------------------------------------------------------

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
        progress_callback: IcLoraProgressCallback | None = None,
    ) -> None:
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from ltx_core.model.upsampler import upsample_video
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import Audio, VideoPixelShape
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput
        from ltx_pipelines.utils.constants import (
            DISTILLED_SIGMA_VALUES,
            STAGE_2_DISTILLED_SIGMA_VALUES,
        )
        from ltx_pipelines.utils.helpers import (
            cleanup_memory,
            denoise_audio_video,
            image_conditionings_by_replacing_latent,
            simple_denoising_func,
        )
        from ltx_pipelines.utils.samplers import euler_denoising_loop

        from services.services_utils import sync_device

        device = self.device
        dtype = torch.bfloat16

        generator = torch.Generator(device=device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        # The upstream ICLoraPipeline was created with device=cpu to prevent
        # it from loading multi-GB models straight onto GPU.  Point the
        # pipeline_components at the real device so that denoise_audio_video
        # creates its tensors on GPU.
        self.pipeline.pipeline_components.device = device

        stage_1_output_shape = VideoPixelShape(
            batch=1, frames=num_frames,
            width=width // 2, height=height // 2, fps=frame_rate,
        )
        target_output_shape = VideoPixelShape(
            batch=1, frames=num_frames,
            width=width, height=height, fps=frame_rate,
        )

        # Optional source audio (for muxing)
        source_audio: AudioOrNone = None
        if source_audio_path is not None:
            from ltx_pipelines.utils.media_io import decode_audio_from_file

            decoded_audio = decode_audio_from_file(
                source_audio_path, device,
                source_audio_start_time, source_audio_max_duration,
            )
            if decoded_audio is not None:
                max_samples = round(num_frames / frame_rate * decoded_audio.sampling_rate)
                trimmed = decoded_audio.waveform.squeeze(0)[..., :max_samples]
                source_audio = Audio(
                    waveform=trimmed.detach().cpu(),
                    sampling_rate=decoded_audio.sampling_rate,
                )

        # ==============================================================
        # Phase 1: Text encoding
        # ==============================================================
        # The app's TextHandler installs a monkey-patch on encode_text that
        # returns pre-computed embeddings (from API or local cached encoder).
        # We just call encode_text with a dummy text_encoder — the patch
        # intercepts and returns cached results without loading Gemma.
        logger.info("[ic-lora] Phase 1: Text encoding (patched)")
        video_context, audio_context = encode_text(None, prompts=[prompt])[0]  # type: ignore[arg-type]
        # Ensure contexts are on GPU
        video_context = video_context.to(device)
        audio_context = audio_context.to(device)
        logger.info("[ic-lora] Phase 1 done")

        # ==============================================================
        # Phase 2: Image / video conditioning
        # ==============================================================
        logger.info("[ic-lora] Phase 2: Conditioning")
        video_encoder = self.pipeline.stage_1_model_ledger.video_encoder()
        _force_module_to_device(video_encoder, device)

        ltx_images = [
            _LtxImageInput(img.path, img.frame_idx, img.strength) for img in images
        ]
        # _create_conditionings uses self.pipeline.device (=cpu) internally.
        # Temporarily swap it to the real device so conditioning tensors land
        # on GPU where the denoiser expects them.
        self.pipeline.device = device
        stage_1_conditionings = self.pipeline._create_conditionings(
            images=ltx_images,
            video_conditioning=video_conditioning,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            num_frames=num_frames,
        )
        self.pipeline.device = self._cpu_device

        # Keep video_encoder for stage 2 upscaling — offload to CPU for now
        video_encoder.to("cpu")
        _sync_and_cleanup(device)
        logger.info("[ic-lora] Phase 2 done")

        # ==============================================================
        # Phase 3: Stage 1 denoising (half resolution)
        # ==============================================================
        logger.info("[ic-lora] Phase 3: Stage 1 denoise")
        transformer_s1 = self.pipeline.stage_1_model_ledger.transformer()
        transformer_s1 = self._prepare_transformer_for_denoise(transformer_s1)

        stage_1_sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, device=device)
        total_s1_steps = max(int(stage_1_sigmas.shape[0]) - 1, 0)
        step_counter = [0]

        def _stage_1_loop(
            sigmas: torch.Tensor,
            video_state: Any,
            audio_state: Any,
            stepper_: Any,
        ) -> tuple[Any, Any]:
            base = simple_denoising_func(
                video_context=video_context,
                audio_context=audio_context,
                transformer=transformer_s1,
            )

            def _tracked(*a: Any, **kw: Any) -> Any:
                result = base(*a, **kw)
                step_counter[0] += 1
                if progress_callback is not None:
                    progress_callback("denoising_stage_1", step_counter[0], total_s1_steps)
                return result

            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper_,
                denoise_fn=_tracked,
            )

        if progress_callback is not None:
            progress_callback("denoising_stage_1", 0, total_s1_steps)

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=stage_1_sigmas,
            stepper=stepper,
            denoising_loop_fn=_stage_1_loop,
            components=self.pipeline.pipeline_components,
            dtype=dtype,
            device=device,
        )

        self._offload_transformer(transformer_s1)
        del transformer_s1
        cleanup_memory()
        logger.info("[ic-lora] Phase 3 done")

        # ==============================================================
        # Phase 4: Stage 2 upsample + refine (full resolution)
        # ==============================================================
        if skip_stage_2:
            logger.info("[ic-lora] Skipping Stage 2")
        else:
            logger.info("[ic-lora] Phase 4: Stage 2 upsample + denoise")
            # Upsample needs video_encoder + spatial_upsampler on GPU
            _force_module_to_device(video_encoder, device)
            spatial_upsampler = self.pipeline.stage_2_model_ledger.spatial_upsampler()
            _force_module_to_device(spatial_upsampler, device)

            upscaled_video_latent = upsample_video(
                latent=video_state.latent[:1],
                video_encoder=video_encoder,
                upsampler=spatial_upsampler,
            )

            # Stage 2 image conditionings at full resolution
            stage_2_conditionings = image_conditionings_by_replacing_latent(
                images=ltx_images,
                height=target_output_shape.height,
                width=target_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=device,
            )

            # Offload encoder + upsampler
            video_encoder.to("cpu")
            spatial_upsampler.to("cpu")
            _sync_and_cleanup(device)

            # Load stage 2 transformer with block swap
            # Clear the block swap wrapper from stage 1 first
            self._block_swap_wrapper = None
            _sync_and_cleanup(device)

            transformer_s2 = self.pipeline.stage_2_model_ledger.transformer()
            transformer_s2 = self._prepare_transformer_for_denoise(transformer_s2)

            stage_2_sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, device=device)
            total_s2_steps = max(int(stage_2_sigmas.shape[0]) - 1, 0)
            step_counter[0] = 0

            def _stage_2_loop(
                sigmas: torch.Tensor,
                video_state: Any,
                audio_state: Any,
                stepper_: Any,
            ) -> tuple[Any, Any]:
                base = simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer_s2,
                )

                def _tracked(*a: Any, **kw: Any) -> Any:
                    result = base(*a, **kw)
                    step_counter[0] += 1
                    if progress_callback is not None:
                        progress_callback("denoising_stage_2", step_counter[0], total_s2_steps)
                    return result

                return euler_denoising_loop(
                    sigmas=sigmas,
                    video_state=video_state,
                    audio_state=audio_state,
                    stepper=stepper_,
                    denoise_fn=_tracked,
                )

            if progress_callback is not None:
                progress_callback("denoising_stage_2", 0, total_s2_steps)

            video_state, audio_state = denoise_audio_video(
                output_shape=target_output_shape,
                conditionings=stage_2_conditionings,
                noiser=noiser,
                sigmas=stage_2_sigmas,
                stepper=stepper,
                denoising_loop_fn=_stage_2_loop,
                components=self.pipeline.pipeline_components,
                dtype=dtype,
                device=device,
                noise_scale=stage_2_sigmas[0],
                initial_video_latent=upscaled_video_latent,
                initial_audio_latent=audio_state.latent,
            )

            self._offload_transformer(transformer_s2)
            del transformer_s2
            cleanup_memory()
            logger.info("[ic-lora] Phase 4 done")

        # ==============================================================
        # Phase 5: VAE decode + write output
        # ==============================================================
        logger.info("[ic-lora] Phase 5: Decode")
        tiling_config = default_tiling_config()

        if skip_stage_2:
            video_decoder = self.pipeline.stage_1_model_ledger.video_decoder()
            audio_decoder = self.pipeline.stage_1_model_ledger.audio_decoder()
            vocoder = self.pipeline.stage_1_model_ledger.vocoder()
        else:
            video_decoder = self.pipeline.stage_2_model_ledger.video_decoder()
            audio_decoder = self.pipeline.stage_2_model_ledger.audio_decoder()
            vocoder = self.pipeline.stage_2_model_ledger.vocoder()

        _force_module_to_device(audio_decoder, device)
        _force_module_to_device(vocoder, device)
        decoded_audio = vae_decode_audio(audio_state.latent, audio_decoder, vocoder)
        try:
            audio_decoder.to(self._cpu_device)
            vocoder.to(self._cpu_device)
        except Exception:
            logger.debug("[ic-lora] Failed to offload audio decoder/vocoder after decode", exc_info=True)
        del audio_decoder, vocoder
        cleanup_memory()

        _force_module_to_device(video_decoder, device)
        decoded_video = vae_decode_video(
            video_state.latent, video_decoder, tiling_config, generator,
        )

        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(
            video=decoded_video,
            audio=source_audio if source_audio is not None else decoded_audio,
            fps=int(frame_rate),
            output_path=output_path,
            video_chunks_number_value=chunks,
        )

        del video_decoder
        # Clean the block swap wrapper so it doesn't leak between generations
        self._block_swap_wrapper = None
        _sync_and_cleanup(device)
        logger.info("[ic-lora] Phase 5 done — output: %s", output_path)
