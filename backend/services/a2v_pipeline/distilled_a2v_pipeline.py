"""Distilled A2V (Audio-to-Video) pipeline.

This keeps the original A2V generation semantics while adding only:
- explicit device placement for key modules/tensors
- progress callback support
- safe decoder lifetime management during lazy video encode
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import gc
import torch

from services.services_utils import AudioOrNone, TilingConfigType, sync_device

if TYPE_CHECKING:
    from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
    from ltx_core.types import LatentState


class DistilledA2VPipeline:
    """Two-stage distilled audio-to-video pipeline.

    Stage 1 generates video at half resolution with frozen audio conditioning,
    then Stage 2 upsamples by 2x and refines with additional distilled steps.
    Uses a single ModelLedger (no LoRA swap between stages).
    """

    def __init__(
        self,
        distilled_checkpoint_path: str,
        gemma_root: str,
        spatial_upsampler_path: str,
        loras: LoraPathStrengthAndSDOps | None = None,
        device: torch.device | None = None,
        quantization: Any | None = None,
        vram_manager: object | None = None,
    ) -> None:
        from ltx_pipelines.utils import ModelLedger
        from ltx_pipelines.utils.helpers import get_device
        from ltx_pipelines.utils.types import PipelineComponents

        if device is None:
            device = get_device()

        self.device = device
        self.dtype = torch.bfloat16
        self._active_video_decoder: Any | None = None
        self._block_swap_wrapper: Any | None = None

        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=distilled_checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=loras,
            quantization=quantization,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

        self.vram_manager = vram_manager

    @staticmethod
    def _ensure_on_device(module: Any, device: torch.device) -> Any:
        if hasattr(module, "to"):
            module = module.to(device)
        return module

    @staticmethod
    def _offload_module(module: Any) -> None:
        if hasattr(module, "to"):
            module.to("cpu")

    @staticmethod
    def _cleanup_cuda() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()

    def _setup_block_swap_if_needed(self, transformer: torch.nn.Module) -> torch.nn.Module:
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
            return transformer

        self._block_swap_wrapper = FastBlockSwapWrapper(
            transformer=transformer,
            device=self.device,
            blocks_to_keep_on_gpu=self.vram_manager.block_swap_keep_on_gpu,
            prefetch_distance=2,
        )
        if self._block_swap_wrapper.block_count == 0:
            self._block_swap_wrapper = None
        return transformer

    def _move_non_block_parts_to_gpu(self, transformer: torch.nn.Module) -> None:
        inner = transformer
        for sub_name in ("velocity_model", "model", "inner_model"):
            sub = getattr(inner, sub_name, None)
            if sub is not None:
                inner = sub
                break

        for _, param in inner.named_parameters(recurse=False):
            param.data = param.data.to(self.device)
        for _, buf in inner.named_buffers(recurse=False):
            buf.data = buf.data.to(self.device)

        block_attr_names = {"transformer_blocks", "blocks", "layers", "encoder_layers"}
        for child_name, child in inner.named_children():
            if child_name not in block_attr_names:
                child.to(self.device)

        if inner is not transformer:
            for _, param in transformer.named_parameters(recurse=False):
                param.data = param.data.to(self.device)
            for _, buf in transformer.named_buffers(recurse=False):
                buf.data = buf.data.to(self.device)
            for _, child in transformer.named_children():
                if child is not inner:
                    child.to(self.device)

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[tuple[str, int, float]],
        audio_path: str,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        tiling_config: TilingConfigType | None = None,
        progress_callback: Any | None = None,
    ) -> tuple[Iterator[torch.Tensor], AudioOrNone]:
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.components.protocols import DiffusionStepProtocol
        from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
        from ltx_core.model.upsampler import upsample_video
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
        from ltx_pipelines.utils.args import ImageConditioningInput as LtxImageInput
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
        from ltx_pipelines.utils.helpers import (
            assert_resolution,
            cleanup_memory,
            denoise_video_only,
            image_conditionings_by_replacing_latent,
            simple_denoising_func,
        )
        from ltx_pipelines.utils.media_io import decode_audio_from_file
        from ltx_pipelines.utils.samplers import euler_denoising_loop

        assert_resolution(height=height, width=width, is_two_stage=True)

        ltx_images = [LtxImageInput(path, frame_idx, strength) for path, frame_idx, strength in images]
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = self.dtype

        # Text encode (positive only).
        text_encoder = self._ensure_on_device(self.model_ledger.text_encoder(), self.device)
        context_p = encode_text(text_encoder, prompts=[prompt])[0]
        video_context, audio_context = context_p
        video_context = video_context.to(device=self.device, dtype=dtype)
        if audio_context is not None:
            audio_context = audio_context.to(device=self.device, dtype=dtype)

        sync_device(self.device)
        self._offload_module(text_encoder)
        del text_encoder
        self._cleanup_cuda()
        cleanup_memory()

        # Audio encode.
        decoded_audio = decode_audio_from_file(audio_path, self.device, audio_start_time, audio_max_duration)
        assert decoded_audio is not None, "Audio file contains no audio stream"
        audio_encoder = self._ensure_on_device(self.model_ledger.audio_encoder(), self.device)
        encoded_audio_latent = vae_encode_audio(decoded_audio, audio_encoder).to(device=self.device, dtype=dtype)
        audio_shape = AudioLatentShape.from_duration(batch=1, duration=num_frames / frame_rate, channels=8, mel_bins=16)
        target_frames = audio_shape.frames
        if encoded_audio_latent.shape[2] < target_frames:
            pad_size = target_frames - encoded_audio_latent.shape[2]
            encoded_audio_latent = torch.nn.functional.pad(encoded_audio_latent, (0, 0, 0, pad_size))
        else:
            encoded_audio_latent = encoded_audio_latent[:, :, :target_frames]

        # Keep original waveform on CPU for final mux fidelity.
        decoded_audio = Audio(
            waveform=decoded_audio.waveform.detach().cpu(),
            sampling_rate=decoded_audio.sampling_rate,
        )

        sync_device(self.device)
        self._offload_module(audio_encoder)
        del audio_encoder
        self._cleanup_cuda()
        cleanup_memory()

        # Shared denoising closure (simple, no guidance).
        video_encoder = self._ensure_on_device(self.model_ledger.video_encoder(), self.device)
        transformer = self.model_ledger.transformer()
        transformer = self._ensure_on_device(transformer, self.device)

        total_steps = (len(DISTILLED_SIGMA_VALUES) - 1) + (len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1)
        step_counter = [0]

        def denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
        ) -> tuple[LatentState, LatentState]:
            base_denoise = simple_denoising_func(
                video_context=video_context,
                audio_context=audio_context,
                transformer=transformer,
            )

            def tracked_denoise(*args: Any, **kwargs: Any) -> Any:
                result = base_denoise(*args, **kwargs)
                step_counter[0] += 1
                if step_counter[0] <= 2:
                    if args:
                        maybe_state = args[0]
                        latent = getattr(maybe_state, "latent", None)
                        if isinstance(latent, torch.Tensor):
                            self._log_tensor_stats(f"step_{step_counter[0]}_input_latent", latent)
                    if isinstance(result, tuple):
                        for idx, item in enumerate(result):
                            latent = getattr(item, "latent", None)
                            if isinstance(latent, torch.Tensor):
                                self._log_tensor_stats(f"step_{step_counter[0]}_output_latent_{idx}", latent)
                    else:
                        latent = getattr(result, "latent", None)
                        if isinstance(latent, torch.Tensor):
                            self._log_tensor_stats(f"step_{step_counter[0]}_output_latent", latent)
                if progress_callback is not None:
                    progress_callback(step_counter[0], total_steps)
                return result

            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=tracked_denoise,
            )

        # Stage 1: Half-resolution video generation with frozen audio.
        stage_1_sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=self.device)
        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = image_conditionings_by_replacing_latent(
            images=ltx_images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )

        video_state = denoise_video_only(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=stage_1_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,  # type: ignore[arg-type]
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            initial_audio_latent=encoded_audio_latent,
        )
        self._log_tensor_stats("stage_1_video_latent", video_state.latent)
        del stage_1_conditionings
        del stage_1_sigmas
        self._cleanup_cuda()

        # Upsample video 2x.
        spatial_upsampler = self._ensure_on_device(self.model_ledger.spatial_upsampler(), self.device)
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=spatial_upsampler,
        )
        self._log_tensor_stats("upscaled_video_latent", upscaled_video_latent)
        del video_state
        self._offload_module(spatial_upsampler)

        sync_device(self.device)
        self._cleanup_cuda()
        cleanup_memory()

        # Stage 2: Full-resolution refinement with frozen audio.
        stage_2_sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = image_conditionings_by_replacing_latent(
            images=ltx_images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        self._offload_module(video_encoder)
        del spatial_upsampler
        self._cleanup_cuda()

        video_state = denoise_video_only(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=stage_2_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,  # type: ignore[arg-type]
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=stage_2_sigmas[0].item(),
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=encoded_audio_latent,
        )
        self._log_tensor_stats("stage_2_video_latent", video_state.latent)

        sync_device(self.device)
        del stage_2_conditionings
        del stage_2_sigmas
        if self._block_swap_wrapper is not None:
            self._block_swap_wrapper.offload_all()
        else:
            self._offload_module(transformer)
        del transformer
        del video_encoder
        cleanup_memory()
        self._cleanup_cuda()

        # Decode video; keep decoder alive until output encoding consumes iterator.
        video_decoder = self._ensure_on_device(self.model_ledger.video_decoder(), self.device)
        self._active_video_decoder = video_decoder
        decoded_video = vae_decode_video(video_state.latent, video_decoder, tiling_config, generator)
        sync_device(self.device)

        # Trim waveform to target video duration so the muxed output doesn't
        # extend beyond the generated video frames.
        max_samples = round(num_frames / frame_rate * decoded_audio.sampling_rate)
        trimmed_waveform = decoded_audio.waveform.squeeze(0)[..., :max_samples]
        original_audio = Audio(waveform=trimmed_waveform, sampling_rate=decoded_audio.sampling_rate)

        return decoded_video, original_audio

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[tuple[str, int, float]],
        audio_path: str,
        output_path: str,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        tiling_config: TilingConfigType | None = None,
        progress_callback: Any | None = None,
    ) -> None:
        from services.ltx_pipeline_common import encode_video_output, video_chunks_number

        video, audio = self(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            audio_path=audio_path,
            audio_start_time=audio_start_time,
            audio_max_duration=audio_max_duration,
            tiling_config=tiling_config,
            progress_callback=progress_callback,
        )
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(
            video=video,
            audio=audio,
            fps=int(frame_rate),
            output_path=output_path,
            video_chunks_number_value=chunks,
        )
        if self._active_video_decoder is not None:
            self._active_video_decoder.to("cpu")
            self._active_video_decoder = None
            self._cleanup_cuda()
