"""LTX retake pipeline wrapper.

Forked orchestration of the retake pipeline flow from ``ltx_pipelines.retake``
with the following fixes applied in-line (no monkey-patching required):

* ``@torch.no_grad()`` instead of ``@torch.inference_mode()`` — the
  transformer checkpoint uses custom autograd functions incompatible with
  inference-mode tensors.
* Tiled video encoding via ``VideoEncoder.tiled_encode`` — the original
  encodes all frames in a single pass which OOMs on most GPUs.
* Tiled video decoding via ``decode_video(..., tiling_config)`` — the
  original omits the tiling argument.
* Lazy ``denoising_loop`` closure (same pattern as ``DistilledA2VPipeline``)
  so that ``del transformer`` actually frees the model before the decoder
  loads.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import logging
from typing import TYPE_CHECKING, Any, cast

import torch

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio
from ltx_pipelines.utils.media_io import encode_video, get_videostream_metadata

from services.retake_pipeline.retake_pipeline import RetakePipeline
from services.services_utils import sync_device

if TYPE_CHECKING:
    from services.block_swap.fast_block_swap import FastBlockSwapWrapper
    from services.vram_manager.vram_manager import VRAMManager

if TYPE_CHECKING:
    from ltx_core.types import LatentState

logger = logging.getLogger(__name__)


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


class LTXRetakePipeline:
    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        device: torch.device,
        *,
        loras: list[LoraPathStrengthAndSDOps] | None = None,
        quantization: QuantizationPolicy | None = None,
        vram_manager: VRAMManager | None = None,
    ) -> RetakePipeline:
        return LTXRetakePipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            device=device,
            loras=loras or [],
            quantization=quantization,
            vram_manager=vram_manager,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        device: torch.device,
        *,
        loras: list[LoraPathStrengthAndSDOps],
        quantization: QuantizationPolicy | None,
        vram_manager: VRAMManager | None,
    ) -> None:
        from ltx_pipelines.utils import ModelLedger
        from ltx_pipelines.utils.types import PipelineComponents

        from services.vram_manager.vram_manager import VRAMManager

        self.device = device
        self.dtype = torch.bfloat16
        self.vram_manager = vram_manager or VRAMManager(device, 0)
        self._block_swap_wrapper: FastBlockSwapWrapper | None = None

        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            loras=loras,  # type: ignore[arg-type]  # upstream ModelLedger accepts list at runtime
            quantization=quantization,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

        self.model_ledger.transformer = _wrap_model_factory(self.model_ledger.transformer, device)

        import ltx_core.text_encoders.gemma as gemma_module
        import ltx_pipelines.retake as retake_module

        original_encode_text = gemma_module.encode_text

        def _encode_text_on_device(*args: Any, **kwargs: Any) -> Any:
            return _move_value_to_device(original_encode_text(*args, **kwargs), device)

        gemma_module.encode_text = _encode_text_on_device
        retake_module.encode_text = _encode_text_on_device
        try:
            import ltx_pipelines.retake_pipeline as legacy_retake_module

            legacy_retake_module.encode_text = _encode_text_on_device
        except ModuleNotFoundError:
            pass

    def _setup_block_swap_if_needed(self, transformer: torch.nn.Module) -> torch.nn.Module:
        """Wrap transformer with block swap if strategy requires it.

        Reuses the existing wrapper if available to avoid accumulating
        duplicate forward hooks on every generation call.
        """
        from services.block_swap.fast_block_swap import FastBlockSwapWrapper
        from services.vram_manager.vram_manager import OffloadStrategy

        if self.vram_manager.offload_strategy not in (
            OffloadStrategy.BLOCK_SWAP,
            OffloadStrategy.BLOCK_SWAP_AGGRESSIVE,
        ):
            logger.info("Retake: block swap not needed for offload strategy %s", self.vram_manager.offload_strategy.value)
            return transformer

        # Reuse existing wrapper — hooks are already installed on the modules
        if self._block_swap_wrapper is not None:
            self._block_swap_wrapper.restore_gpu_blocks()
            logger.info("Retake: reusing existing block swap wrapper")
            return transformer

        blocks_on_gpu = self.vram_manager.block_swap_keep_on_gpu
        logger.info("Retake: setting up block swap, keeping %d blocks on GPU", blocks_on_gpu)

        self._block_swap_wrapper = FastBlockSwapWrapper(
            transformer=transformer,
            device=self.device,
            blocks_to_keep_on_gpu=blocks_on_gpu,
            prefetch_distance=2,
        )
        self._block_swap_wrapper._default_blocks_to_keep_on_gpu = blocks_on_gpu  # type: ignore[attr-defined]
        self._block_swap_wrapper._default_prefetch_distance = 2  # type: ignore[attr-defined]

        if self._block_swap_wrapper.block_count == 0:
            logger.warning("Retake: block swap found 0 blocks — falling back to sequential offloading")
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

    @torch.no_grad()
    def _run(  # noqa: PLR0913, PLR0915
        self,
        video_path: str,
        prompt: str,
        start_time: float,
        end_time: float,
        seed: int,
        *,
        negative_prompt: str = "",
        num_inference_steps: int = 40,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
        regenerate_video: bool = True,
        regenerate_audio: bool = True,
        enhance_prompt: bool = False,
        distilled: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.guiders import MultiModalGuider
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.components.protocols import DiffusionStepProtocol
        from ltx_core.components.schedulers import LTX2Scheduler
        from ltx_core.conditioning import ConditioningItem
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import AudioLatentShape, VideoPixelShape
        from ltx_pipelines.utils.helpers import (
            cleanup_memory,
            multi_modal_guider_denoising_func,
            noise_audio_state,
            noise_video_state,
            simple_denoising_func,
        )
        from ltx_pipelines.utils.media_io import (
            decode_audio_from_file,
            load_video_conditioning,
        )
        from ltx_pipelines.utils.samplers import euler_denoising_loop

        try:
            from ltx_pipelines.retake import TemporalRegionMask
        except ImportError:
            from ltx_pipelines.retake_pipeline import TemporalRegionMask  # type: ignore[no-redef]

        try:
            from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES as _distilled_sigmas
        except ImportError:
            _distilled_sigmas = [1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.04, 0.0]

        if start_time >= end_time:
            raise ValueError(f"start_time ({start_time}) must be less than end_time ({end_time})")

        effective_seed = int(torch.randint(0, 2**31, (1,)).item()) if seed < 0 else seed
        generator = torch.Generator(device=self.device).manual_seed(effective_seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = self.dtype
        tiling = TilingConfig.default()

        # --- Encode source video (tiled) ---
        video_encoder = self.model_ledger.video_encoder()
        fps, num_pixel_frames, src_width, src_height = get_videostream_metadata(video_path)
        output_shape = VideoPixelShape(
            batch=1, frames=num_pixel_frames, width=src_width, height=src_height, fps=fps,
        )
        pixel_video = load_video_conditioning(
            video_path=video_path,
            height=output_shape.height,
            width=output_shape.width,
            frame_cap=output_shape.frames,
            dtype=dtype,
            device=self.device,
        )
        initial_video_latent = video_encoder.tiled_encode(pixel_video, tiling)
        del pixel_video

        video_conditionings: list[ConditioningItem] = [
            TemporalRegionMask(
                start_time=start_time if regenerate_video else 0.0,
                end_time=end_time if regenerate_video else 0.0,
                fps=fps,
            )
        ]
        del video_encoder
        cleanup_memory()

        # --- Encode source audio ---
        initial_audio_latent: torch.Tensor | None = None
        audio_conditionings: list[ConditioningItem] = []
        audio_in = decode_audio_from_file(video_path, self.device)
        audio_encoder = self.model_ledger.audio_encoder()

        if audio_in is not None:
            waveform = audio_in.waveform.squeeze(0)
            waveform_sr = audio_in.sampling_rate
        else:
            waveform, waveform_sr = None, None

        if waveform is not None and waveform_sr is not None:
            waveform_batch = waveform.unsqueeze(0) if waveform.dim() == 2 else waveform
            initial_audio_latent = vae_encode_audio(
                Audio(waveform=waveform_batch.to(dtype), sampling_rate=waveform_sr),
                audio_encoder,
                None,
            )
            expected = AudioLatentShape.from_video_pixel_shape(output_shape).frames
            actual = initial_audio_latent.shape[2]
            if actual > expected:
                initial_audio_latent = initial_audio_latent[:, :, :expected, :]
            elif actual < expected:
                pad = torch.zeros(
                    initial_audio_latent.shape[0],
                    initial_audio_latent.shape[1],
                    expected - actual,
                    initial_audio_latent.shape[3],
                    device=initial_audio_latent.device,
                    dtype=initial_audio_latent.dtype,
                )
                initial_audio_latent = torch.cat([initial_audio_latent, pad], dim=2)

            audio_conditionings = [
                TemporalRegionMask(
                    start_time=start_time if regenerate_audio else 0.0,
                    end_time=end_time if regenerate_audio else 0.0,
                    fps=fps,
                )
            ]

        del audio_encoder
        cleanup_memory()

        # --- Text encoding ---
        text_encoder = self.model_ledger.text_encoder()
        text_encoder.to(self.device)

        v_context_n: torch.Tensor | None = None
        a_context_n: torch.Tensor | None = None

        if distilled:
            context_p = encode_text(text_encoder, prompts=[prompt])[0]
            v_context_p, a_context_p = context_p
        else:
            context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
            v_context_p, a_context_p = context_p
            v_context_n, a_context_n = context_n

        sync_device(self.device)
        del text_encoder
        cleanup_memory()

        # --- Denoising ---
        self.vram_manager.cleanup()
        transformer = self.model_ledger.transformer()
        transformer = self._setup_block_swap_if_needed(transformer)
        if self._block_swap_wrapper is not None:
            self._block_swap_wrapper.offload_all()
            self._move_non_block_parts_to_gpu(transformer)
            self._block_swap_wrapper.restore_gpu_blocks()
        else:
            self.vram_manager.ensure_on_gpu("retake_transformer", transformer)

        raw_sigmas: torch.Tensor = (
            torch.tensor(_distilled_sigmas)
            if distilled
            else LTX2Scheduler().execute(steps=num_inference_steps)  # type: ignore[no-untyped-call]
        )
        sigmas = raw_sigmas.to(dtype=torch.float32, device=self.device)
        total_steps = max(1, len(sigmas) - 1)
        step_counter = [0]

        if distilled:
            def denoising_loop(
                sigmas: torch.Tensor,
                video_state: LatentState,
                audio_state: LatentState,
                stepper: DiffusionStepProtocol,
            ) -> tuple[LatentState, LatentState]:
                base_denoise = simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=cast(Any, transformer),
                )

                def tracked_denoise(*args: Any, **kwargs: Any) -> Any:
                    result = base_denoise(*args, **kwargs)
                    step_counter[0] += 1
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
        else:
            assert video_guider_params is not None, "video_guider_params required for non-distilled"
            assert audio_guider_params is not None, "audio_guider_params required for non-distilled"
            assert v_context_n is not None, "v_context_n required for non-distilled"
            assert a_context_n is not None, "a_context_n required for non-distilled"
            video_guider = MultiModalGuider(params=video_guider_params, negative_context=v_context_n)
            audio_guider = MultiModalGuider(params=audio_guider_params, negative_context=a_context_n)

            def denoising_loop(
                sigmas: torch.Tensor,
                video_state: LatentState,
                audio_state: LatentState,
                stepper: DiffusionStepProtocol,
            ) -> tuple[LatentState, LatentState]:
                base_denoise = multi_modal_guider_denoising_func(
                    video_guider,
                    audio_guider,
                    v_context=v_context_p,
                    a_context=a_context_p,
                    transformer=cast(Any, transformer),
                )

                def tracked_denoise(*args: Any, **kwargs: Any) -> Any:
                    result = base_denoise(*args, **kwargs)
                    step_counter[0] += 1
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

        video_state, video_tools = noise_video_state(
            output_shape=output_shape,
            noiser=noiser,
            conditionings=video_conditionings,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            initial_latent=initial_video_latent,
        )
        audio_state, audio_tools = noise_audio_state(
            output_shape=output_shape,
            noiser=noiser,
            conditionings=audio_conditionings,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            initial_latent=initial_audio_latent,
        )

        video_state, audio_state = denoising_loop(sigmas, video_state, audio_state, stepper)

        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

        sync_device(self.device)
        if self._block_swap_wrapper is not None:
            self._block_swap_wrapper.offload_all()
        else:
            self.vram_manager.offload_to_cpu("retake_transformer", transformer)
        del transformer, denoising_loop
        cleanup_memory()

        # --- Decode audio first (eager, small) then free those models ---
        decoded_audio = vae_decode_audio(
            audio_state.latent, self.model_ledger.audio_decoder(), self.model_ledger.vocoder(),
        )
        del audio_state
        cleanup_memory()

        # --- Decode video (lazy generator, tiled) ---
        decoded_video = vae_decode_video(
            video_state.latent, self.model_ledger.video_decoder(), tiling, generator,
        )

        return decoded_video, decoded_audio

    @torch.no_grad()
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
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
        regenerate_video: bool = True,
        regenerate_audio: bool = True,
        enhance_prompt: bool = False,
        distilled: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        fps, num_frames, _, _ = get_videostream_metadata(video_path)
        video_iter, audio = self._run(
            video_path=video_path,
            prompt=prompt,
            start_time=start_time,
            end_time=end_time,
            seed=seed,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            regenerate_video=regenerate_video,
            regenerate_audio=regenerate_audio,
            enhance_prompt=enhance_prompt,
            distilled=distilled,
            progress_callback=progress_callback,
        )
        audio_out: Audio | None = audio
        tiling_config = TilingConfig.default()
        video_chunks = get_video_chunks_number(num_frames, tiling_config)
        encode_video(
            video=video_iter,
            fps=int(fps),
            audio=audio_out,
            output_path=output_path,
            video_chunks_number=video_chunks,
        )
