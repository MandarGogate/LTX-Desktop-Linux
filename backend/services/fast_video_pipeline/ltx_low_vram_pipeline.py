"""Low-VRAM pipeline using sequential offloading, GGUF, and block swap.

This pipeline is designed for consumer GPUs (8-24GB VRAM) and implements
three key techniques from ComfyUI:

1. **Sequential Model Offloading**: Only one major model on GPU at a time.
   Text encoder → evict → Transformer → evict → VAE decoder.

2. **GGUF Quantized Models**: Load transformer from GGUF format (Q4/Q5/Q8)
   reducing VRAM from ~22GB to 5-12GB for the transformer alone.

3. **Block Swap**: For the transformer, swap individual transformer blocks
   between CPU and GPU during the forward pass. Only a few blocks need to
   be on GPU at once since they execute sequentially.

Memory flow for a 12GB GPU at 540p:
- Phase 1: Text encoder on GPU (~3GB) → encode → offload
- Phase 2: Transformer blocks swap through GPU (~2-4GB active) → denoise → offload
- Phase 3: VAE decoder on GPU (~2GB) → decode → offload
- Peak: ~4GB (with block swap) vs ~22GB (without)
"""

from __future__ import annotations

import gc
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

if TYPE_CHECKING:
    import torch

    from api_types import ImageConditioningInput
    from services.block_swap.block_swap import BlockSwapTransformerWrapper
    from services.gguf_loader.gguf_loader import GGUFModelLoader
    from services.vram_manager.vram_manager import VRAMManager

logger = logging.getLogger(__name__)


class LTXLowVRAMPipeline:
    """Low-VRAM pipeline with sequential offloading + GGUF + block swap.

    Conforms to the FastVideoPipeline protocol so it can be used as a
    drop-in replacement for LTXFastVideoPipeline.
    """

    pipeline_kind: Final = "fast"

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        *,
        vram_manager: VRAMManager | None = None,
        gguf_path: str | None = None,
    ) -> "LTXLowVRAMPipeline":
        return LTXLowVRAMPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            vram_manager=vram_manager,
            gguf_path=gguf_path,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        *,
        vram_manager: VRAMManager | None = None,
        gguf_path: str | None = None,
    ) -> None:
        import torch as _torch

        from services.vram_manager.vram_manager import VRAMManager

        self._torch = _torch
        self.device = device
        self.dtype = _torch.bfloat16
        self._checkpoint_path = checkpoint_path
        self._gemma_root = gemma_root
        self._upsampler_path = upsampler_path
        self._gguf_path = gguf_path

        # Create default VRAMManager if not provided
        if vram_manager is None:
            vram_gb = 24
            if _torch.cuda.is_available():
                vram_gb = int(
                    _torch.cuda.get_device_properties(0).total_memory  # type: ignore[union-attr]
                    // (1024**3)
                )
            vram_manager = VRAMManager(device, vram_gb)

        self.vram_manager = vram_manager
        self._block_swap_wrapper: BlockSwapTransformerWrapper | None = None
        self._use_gguf = False

        # Load model components to CPU for sequential offloading
        logger.info(
            "Initializing LTXLowVRAMPipeline: tier=%s strategy=%s gguf=%s",
            vram_manager.tier.value,
            vram_manager.offload_strategy.value,
            gguf_path is not None,
        )

        self._init_model_ledger()

    def _init_model_ledger(self) -> None:
        """Initialize the model ledger with models on CPU."""
        from services.services_utils import device_supports_fp8

        # Use GGUF if available and recommended
        if self._gguf_path is not None and Path(self._gguf_path).exists():
            logger.info("Will use GGUF model: %s", self._gguf_path)
            self._use_gguf = True

        # Initialize model ledger — load to CPU first (key for low VRAM)
        from ltx_pipelines.utils import ModelLedger
        from ltx_pipelines.utils.types import PipelineComponents

        # Determine quantization
        quantization = None
        if not self._use_gguf and device_supports_fp8(self.device):
            from ltx_core.quantization import QuantizationPolicy

            quantization = QuantizationPolicy.fp8_cast()

        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=self._torch.device("cpu"),  # Load to CPU first!
            checkpoint_path=self._checkpoint_path,
            gemma_root_path=self._gemma_root,
            loras=None,
            quantization=quantization,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=self.device,
        )

    def _setup_block_swap_if_needed(self, transformer: torch.nn.Module) -> torch.nn.Module:
        """Wrap transformer with block swap if strategy requires it."""
        from services.block_swap.block_swap import BlockSwapTransformerWrapper
        from services.vram_manager.vram_manager import OffloadStrategy

        if self.vram_manager.offload_strategy != OffloadStrategy.BLOCK_SWAP:
            return transformer

        blocks_on_gpu = self.vram_manager.block_swap_keep_on_gpu
        logger.info("Setting up block swap: keeping %d blocks on GPU", blocks_on_gpu)

        self._block_swap_wrapper = BlockSwapTransformerWrapper(
            transformer=transformer,
            device=self.device,
            blocks_to_keep_on_gpu=blocks_on_gpu,
            use_async_prefetch=True,
        )

        return transformer

    def _load_gguf_transformer(self) -> Any:
        """Load transformer from GGUF file instead of safetensors."""
        if self._gguf_path is None:
            raise RuntimeError("No GGUF path configured")

        from services.gguf_loader.gguf_loader import GGUFModelLoader

        logger.info("Loading GGUF transformer from %s", self._gguf_path)
        state_dict = GGUFModelLoader.load_gguf_sd_for_diffusers(
            Path(self._gguf_path),
            device="cpu",
        )

        # Get the transformer architecture and load the GGUF weights
        transformer = self.model_ledger.transformer()
        try:
            transformer.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.warning("GGUF state dict load had issues: %s", e)

        logger.info("GGUF transformer loaded successfully")
        return transformer

    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[Any],
        output_path: str,
    ) -> None:
        """Generate a video with sequential model offloading.

        Memory flow:
        1. Text encoder to GPU → encode prompt → offload to CPU
        2. Video encoder to GPU (if i2v) → encode images → offload
        3. Transformer to GPU (with block swap) → denoise → offload
        4. Video decoder to GPU → decode latents → offload
        5. Audio decoder to GPU → decode audio → offload
        6. Encode final video to disk
        """
        import torch

        with torch.inference_mode():
            self._generate_impl(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=images,
                output_path=output_path,
            )

    def _generate_impl(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[Any],
        output_path: str,
    ) -> None:
        import torch

        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
        from ltx_pipelines.utils.helpers import (
            cleanup_memory,
            denoise_audio_video,
            image_conditionings_by_replacing_latent,
            simple_denoising_func,
        )
        from ltx_pipelines.utils.samplers import euler_denoising_loop

        from services.ltx_pipeline_common import default_tiling_config, encode_video_output, video_chunks_number

        logger.info(
            "[low-vram] Starting generation: %dx%d, %d frames, tier=%s",
            width, height, num_frames, self.vram_manager.tier.value,
        )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        # ============================================================
        # Phase 1: Text encoding (text encoder on GPU)
        # ============================================================
        logger.info("[low-vram] Phase 1: Text encoding")
        text_encoder = self.model_ledger.text_encoder()
        self.vram_manager.ensure_on_gpu("text_encoder", text_encoder)

        context_p = encode_text(text_encoder, prompts=[prompt])[0]
        video_context, audio_context = context_p

        self.vram_manager.offload_to_cpu("text_encoder", text_encoder)
        del text_encoder
        self.vram_manager.cleanup()

        # ============================================================
        # Phase 2: Image conditioning (if i2v)
        # ============================================================
        output_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width, height=height, fps=frame_rate,
        )

        conditionings = None
        if images:
            logger.info("[low-vram] Phase 2: Image conditioning")
            video_encoder = self.model_ledger.video_encoder()
            self.vram_manager.ensure_on_gpu("video_encoder", video_encoder)

            ltx_images = [_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images]
            conditionings = image_conditionings_by_replacing_latent(
                images=ltx_images,
                height=output_shape.height,
                width=output_shape.width,
                video_encoder=video_encoder,
                dtype=self.dtype,
                device=self.device,
            )

            self.vram_manager.offload_to_cpu("video_encoder", video_encoder)
            del video_encoder
            self.vram_manager.cleanup()

        # ============================================================
        # Phase 3: Denoising (transformer on GPU, with block swap)
        # ============================================================
        logger.info("[low-vram] Phase 3: Denoising (transformer)")

        # Load transformer — GGUF or standard
        if self._use_gguf:
            transformer = self._load_gguf_transformer()
        else:
            transformer = self.model_ledger.transformer()

        # Setup block swap for low-VRAM tiers
        transformer = self._setup_block_swap_if_needed(transformer)

        # Move non-block-swapped parts to GPU
        if self._block_swap_wrapper is None:
            self.vram_manager.ensure_on_gpu("transformer", transformer)

        sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)

        def denoising_loop(
            sigmas: torch.Tensor,
            video_state: Any,
            audio_state: Any,
            stepper: EulerDiffusionStep,
        ) -> tuple[Any, Any]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer,
                ),
            )

        video_state, audio_state = denoise_audio_video(
            output_shape=output_shape,
            conditionings=conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=cast(Any, denoising_loop),
            components=self.pipeline_components,
            dtype=self.dtype,
            device=self.device,
        )

        # Offload transformer
        if self._block_swap_wrapper is not None:
            self._block_swap_wrapper.offload_all()
        else:
            self.vram_manager.offload_to_cpu("transformer", transformer)
        del transformer
        self.vram_manager.cleanup()

        # ============================================================
        # Phase 4: Video VAE decode
        # ============================================================
        logger.info("[low-vram] Phase 4: VAE video decode")
        tiling_config = self._get_adaptive_tiling_config()

        video_decoder = self.model_ledger.video_decoder()
        self.vram_manager.ensure_on_gpu("video_decoder", video_decoder)

        decoded_video = vae_decode_video(
            video_state.latent, video_decoder, tiling_config,
        )

        self.vram_manager.offload_to_cpu("video_decoder", video_decoder)
        del video_decoder
        self.vram_manager.cleanup()

        # ============================================================
        # Phase 5: Audio VAE decode
        # ============================================================
        logger.info("[low-vram] Phase 5: Audio decode")
        audio_decoder = self.model_ledger.audio_decoder()
        vocoder = self.model_ledger.vocoder()
        self.vram_manager.ensure_on_gpu("audio_decoder", audio_decoder)

        decoded_audio = vae_decode_audio(
            audio_state.latent, audio_decoder, vocoder,
        )

        self.vram_manager.offload_to_cpu("audio_decoder", audio_decoder)
        del audio_decoder, vocoder
        self.vram_manager.cleanup()

        # ============================================================
        # Phase 6: Encode to video file
        # ============================================================
        logger.info("[low-vram] Phase 6: Encoding video output")
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(
            video=decoded_video,
            audio=decoded_audio,
            fps=int(frame_rate),
            output_path=output_path,
            video_chunks_number_value=chunks,
        )

        logger.info("[low-vram] Generation complete: %s", output_path)

    def _get_adaptive_tiling_config(self) -> Any:
        """Return tiling config adapted to VRAM tier.

        TilingConfig API (from ltx_core.model.video_vae):
          TilingConfig(
              spatial_config=SpatialTilingConfig(tile_size_in_pixels, tile_overlap_in_pixels),
              temporal_config=TemporalTilingConfig(tile_size_in_frames, tile_overlap_in_frames),
          )
        Constraints:
          - spatial tile_size_in_pixels >= 64, divisible by 32
          - spatial tile_overlap_in_pixels divisible by 32, < tile_size
          - temporal tile_size_in_frames >= 16, divisible by 8
          - temporal tile_overlap_in_frames divisible by 8, < tile_size
        """
        from ltx_core.model.video_vae import (
            SpatialTilingConfig,
            TemporalTilingConfig,
            TilingConfig,
        )

        from services.vram_manager.vram_manager import VRAMTier

        match self.vram_manager.tier:
            case VRAMTier.HIGH:
                # Default: 512px spatial tiles, 64-frame temporal tiles
                return TilingConfig.default()
            case VRAMTier.MEDIUM:
                # Smaller spatial tiles to reduce peak VRAM
                return TilingConfig(
                    spatial_config=SpatialTilingConfig(
                        tile_size_in_pixels=256,
                        tile_overlap_in_pixels=64,
                    ),
                    temporal_config=TemporalTilingConfig(
                        tile_size_in_frames=32,
                        tile_overlap_in_frames=8,
                    ),
                )
            case _:
                # Aggressive tiling for LOW / VERY_LOW
                return TilingConfig(
                    spatial_config=SpatialTilingConfig(
                        tile_size_in_pixels=128,
                        tile_overlap_in_pixels=32,
                    ),
                    temporal_config=TemporalTilingConfig(
                        tile_size_in_frames=16,
                        tile_overlap_in_frames=8,
                    ),
                )

    def warmup(self, output_path: str) -> None:
        """Minimal warmup at low resolution."""
        import torch

        warmup_frames = 9
        try:
            with torch.inference_mode():
                self._generate_impl(
                    prompt="test warmup",
                    seed=42,
                    height=256,
                    width=384,
                    num_frames=warmup_frames,
                    frame_rate=8,
                    images=[],
                    output_path=output_path,
                )
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def compile_transformer(self) -> None:
        """Compile transformer (skipped for low-VRAM — block swap is incompatible)."""
        logger.info(
            "Skipping torch.compile for low-VRAM pipeline "
            "(incompatible with block swap / sequential offloading)"
        )
