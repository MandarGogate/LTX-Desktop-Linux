"""Low-VRAM pipeline using sequential offloading, GGUF, block swap, and SageAttention.

This pipeline is designed for consumer GPUs (8-24GB VRAM) and implements:

1. **Sequential Model Offloading**: Only one major model on GPU at a time.
2. **GGUF Quantized Models**: Load transformer from GGUF format (Q4/Q8).
3. **Block Swap**: Swap transformer blocks between CPU and GPU.
4. **SageAttention**: Replace PyTorch SDPA with SageAttention for 2-3× speed.
5. **Layerwise Text Encoder**: Block-swap the Gemma text encoder layers too.
6. **Custom LoRA**: Support multiple LoRAs with per-LoRA strength.
7. **Non-distilled (dev) model**: Custom sigma schedules for the dev checkpoint.
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

# ---------------------------------------------------------------------------
# Non-distilled (dev) sigma schedule — 50 steps flow-matching
# ---------------------------------------------------------------------------
_DEV_SIGMA_VALUES_50: list[float] = [
    float(1.0 - i / 50) for i in range(51)
]
# 20-step schedule for faster dev inference
_DEV_SIGMA_VALUES_20: list[float] = [
    float(1.0 - i / 20) for i in range(21)
]


def _make_dev_sigmas(steps: int) -> list[float]:
    """Create a linear sigma schedule for the non-distilled (dev) model."""
    return [float(1.0 - i / steps) for i in range(steps + 1)]


# ---------------------------------------------------------------------------
# SageAttention monkey-patch
# ---------------------------------------------------------------------------
_sage_attention_installed = False


def install_sage_attention() -> bool:
    """Replace PyTorch SDPA with SageAttention in the LTX attention module.

    SageAttention uses INT8 quantised Q·K matmuls with FP8/FP16 accumulation
    giving ~2-3× speed-up over cuDNN SDPA while using less VRAM.
    """
    global _sage_attention_installed
    if _sage_attention_installed:
        return True

    try:
        from sageattention import sageattn
    except ImportError:
        logger.info("SageAttention not installed — using default attention")
        return False

    try:
        from ltx_core.model.transformer import attention as attn_mod

        class SageAttn(attn_mod.AttentionCallable):
            """Drop-in replacement using SageAttention."""

            def __call__(
                self,
                q: Any,
                k: Any,
                v: Any,
                heads: int,
                mask: Any | None = None,
            ) -> Any:
                import torch as _t

                b, _, dim_head = q.shape
                dim_head //= heads
                q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

                # SageAttention is strict about dtype equality and mask support.
                # Fall back to SDPA when:
                # - a mask is provided
                # - q/k/v dtypes differ
                # - the tensors are not CUDA tensors
                # - SageAttention raises for an unsupported dtype/layout
                use_sdpa = False
                if mask is not None:
                    use_sdpa = True
                if q.dtype != k.dtype or k.dtype != v.dtype:
                    logger.debug(
                        "SageAttention fallback to SDPA due to mixed dtypes: q=%s k=%s v=%s",
                        q.dtype,
                        k.dtype,
                        v.dtype,
                    )
                    use_sdpa = True
                if not (q.is_cuda and k.is_cuda and v.is_cuda):
                    use_sdpa = True

                if use_sdpa:
                    if mask is not None:
                        if mask.ndim == 2:
                            mask = mask.unsqueeze(0)
                        if mask.ndim == 3:
                            mask = mask.unsqueeze(1)
                    out = _t.nn.functional.scaled_dot_product_attention(
                        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False,
                    )
                else:
                    try:
                        out = sageattn(q, k, v, is_causal=False)
                    except Exception as exc:
                        logger.debug("SageAttention failed (%s); falling back to SDPA", exc)
                        out = _t.nn.functional.scaled_dot_product_attention(
                            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                        )

                return out.transpose(1, 2).reshape(b, -1, heads * dim_head)

        _original_default = attn_mod.AttentionFunction.DEFAULT

        # Monkey-patch the DEFAULT to prefer SageAttention
        def _sage_default(
            self: Any, q: Any, k: Any, v: Any, heads: int, mask: Any | None = None,
        ) -> Any:
            return SageAttn()(q, k, v, heads, mask)

        attn_mod.AttentionFunction.__call__ = _sage_default  # type: ignore[assignment]
        _sage_attention_installed = True
        logger.info("SageAttention installed as default attention backend")
        return True
    except Exception as exc:
        logger.warning("Failed to install SageAttention: %s", exc)
        return False


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
        lora_path: str | None = None,
        lora_strength: float = 1.0,
        extra_loras: list[tuple[str, float]] | None = None,
        use_sage_attention: bool = True,
        num_inference_steps: int | None = None,
        text_encoder_variant_path: str | None = None,
    ) -> "LTXLowVRAMPipeline":
        return LTXLowVRAMPipeline(
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
        lora_path: str | None = None,
        lora_strength: float = 1.0,
        extra_loras: list[tuple[str, float]] | None = None,
        use_sage_attention: bool = True,
        num_inference_steps: int | None = None,
        text_encoder_variant_path: str | None = None,
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
        self._lora_path = lora_path
        self._lora_strength = lora_strength
        self._extra_loras = extra_loras or []
        self._num_inference_steps = num_inference_steps
        self._text_encoder_variant_path = text_encoder_variant_path

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

        # Cached model references — avoid reloading from disk each generation
        self._cached_text_encoder: Any = None
        self._cached_transformer: Any = None
        self._cached_video_decoder: Any = None
        self._cached_audio_decoder: Any = None
        self._cached_vocoder: Any = None
        self._cached_video_encoder: Any = None

        # Install SageAttention if requested
        if use_sage_attention:
            install_sage_attention()

        logger.info(
            "Initializing LTXLowVRAMPipeline: tier=%s strategy=%s gguf=%s sage=%s",
            vram_manager.tier.value,
            vram_manager.offload_strategy.value,
            gguf_path is not None,
            _sage_attention_installed,
        )

        self._init_model_ledger()

    def _init_model_ledger(self) -> None:
        """Initialize the model ledger with models on CPU."""
        from services.services_utils import device_supports_fp8

        # Use GGUF if available
        if self._gguf_path is not None and Path(self._gguf_path).exists():
            logger.info("Will use GGUF model: %s", self._gguf_path)
            self._use_gguf = True

        from ltx_pipelines.utils import ModelLedger
        from ltx_pipelines.utils.types import PipelineComponents

        # Always use FP8 for the transformer when supported
        quantization = None
        if device_supports_fp8(self.device):
            from ltx_core.quantization import QuantizationPolicy

            quantization = QuantizationPolicy.fp8_cast()
            logger.info("FP8 quantization enabled")

        # Build LoRA list from primary + extras.
        # Note: LoRAs must be compatible with the model architecture.
        # The LTX 2.3 (22B) model uses LoRAs trained for 22B dimensions,
        # while LTX 2.0 (19B) LoRAs have different tensor sizes and will fail.
        loras = None
        all_lora_entries = self._collect_loras()
        if all_lora_entries:
            loras = all_lora_entries
            for l in all_lora_entries:
                logger.info("LoRA: %s (strength=%.2f)", l.path, l.strength)

        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=self._torch.device("cpu"),
            checkpoint_path=self._checkpoint_path,
            gemma_root_path=self._gemma_root,
            loras=loras,
            quantization=quantization,
        )

        # Optional override for a pre-quantized text encoder safetensors.
        # We still use the base gemma root for tokenizer + processor config,
        # but swap the model weight file used by the text encoder builder.
        if self._text_encoder_variant_path and self._gemma_root and Path(self._text_encoder_variant_path).exists():
            try:
                from ltx_core.loader.registry import DummyRegistry
                from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
                from ltx_core.text_encoders.gemma import (
                    AV_GEMMA_TEXT_ENCODER_KEY_OPS,
                    GEMMA_MODEL_OPS,
                    GemmaTextEncoderConfigurator,
                    module_ops_from_gemma_root,
                )

                variant_path = str(Path(self._text_encoder_variant_path))
                module_ops = module_ops_from_gemma_root(self._gemma_root)
                if not hasattr(self.model_ledger, "_default_text_encoder_builder"):
                    setattr(
                        self.model_ledger,
                        "_default_text_encoder_builder",
                        self.model_ledger.text_encoder_builder,
                    )
                self.model_ledger.text_encoder_builder = Builder(
                    model_path=(str(self._checkpoint_path), variant_path),
                    model_class_configurator=GemmaTextEncoderConfigurator,
                    model_sd_ops=AV_GEMMA_TEXT_ENCODER_KEY_OPS,
                    registry=DummyRegistry(),
                    module_ops=(GEMMA_MODEL_OPS, *module_ops),
                )
                logger.info("Using text encoder variant: %s", variant_path)
            except Exception as exc:
                logger.warning("Failed to configure text encoder variant %s: %s", self._text_encoder_variant_path, exc)

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=self.device,
        )

    def _collect_loras(self) -> list[Any] | None:
        """Build a list of LoRA entries from primary lora_path + extra_loras."""
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP

        entries: list[Any] = []
        if self._lora_path and Path(self._lora_path).exists():
            entries.append(LoraPathStrengthAndSDOps(
                path=self._lora_path,
                strength=self._lora_strength,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            ))
        for path, strength in self._extra_loras:
            if Path(path).exists():
                entries.append(LoraPathStrengthAndSDOps(
                    path=path,
                    strength=strength,
                    sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                ))
        return entries if entries else None

    # ------------------------------------------------------------------
    # Block swap
    # ------------------------------------------------------------------

    def _setup_block_swap_if_needed(self, transformer: torch.nn.Module) -> torch.nn.Module:
        """Wrap transformer with block swap if strategy requires it."""
        from services.block_swap.block_swap import BlockSwapTransformerWrapper
        from services.vram_manager.vram_manager import OffloadStrategy

        if self.vram_manager.offload_strategy not in (
            OffloadStrategy.BLOCK_SWAP,
            OffloadStrategy.BLOCK_SWAP_AGGRESSIVE,
        ):
            return transformer

        blocks_on_gpu = self.vram_manager.block_swap_keep_on_gpu
        logger.info("Setting up block swap: keeping %d blocks on GPU", blocks_on_gpu)

        self._block_swap_wrapper = BlockSwapTransformerWrapper(
            transformer=transformer,
            device=self.device,
            blocks_to_keep_on_gpu=blocks_on_gpu,
            use_async_prefetch=True,
        )

        if self._block_swap_wrapper.block_count == 0:
            logger.warning("Block swap found 0 blocks — falling back to sequential offloading")
            self._block_swap_wrapper = None

        return transformer

    def _load_gguf_transformer(self) -> Any:
        """Load transformer from GGUF file instead of safetensors.

        We can't rely on ``ModelLedger.transformer()`` here because the checkpoint
        shim used for GGUF mode may yield a meta-backed model that fails on
        ``.to(device)``. Build the architecture safely on CPU, materialize any
        meta tensors with ``to_empty()``, then load the GGUF state dict.
        """
        if self._gguf_path is None:
            raise RuntimeError("No GGUF path configured")

        import torch
        from dataclasses import replace
        from ltx_core.loader import SDOps
        from ltx_core.model.transformer import X0Model
        from services.gguf_loader.gguf_loader import GGUFModelLoader

        logger.info("Loading GGUF transformer from %s", self._gguf_path)
        state_dict = GGUFModelLoader.load_gguf_sd_for_diffusers(
            Path(self._gguf_path),
            device="cpu",
        )

        builder = self.model_ledger.transformer_builder
        if self.model_ledger.quantization is not None:
            sd_ops = builder.model_sd_ops
            if self.model_ledger.quantization.sd_ops is not None:
                sd_ops = SDOps(
                    name=f"sd_ops_chain_{sd_ops.name}+{self.model_ledger.quantization.sd_ops.name}",
                    mapping=(*sd_ops.mapping, *self.model_ledger.quantization.sd_ops.mapping),
                )
            builder = replace(
                builder,
                module_ops=(*builder.module_ops, *self.model_ledger.quantization.module_ops),
                model_sd_ops=sd_ops,
            )

        base_model = builder.build(device=torch.device("cpu"))
        has_meta = any(
            str(t.device) == "meta"
            for t in list(base_model.parameters()) + list(base_model.buffers())
        )
        if has_meta:
            logger.info("GGUF transformer base model has meta tensors, materializing with to_empty()")
            base_model = base_model.to_empty(device=torch.device("cpu"))

        transformer = X0Model(base_model).eval()
        try:
            transformer.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.warning("GGUF state dict load had issues: %s", e)

        logger.info("GGUF transformer loaded successfully")
        return transformer

    # ------------------------------------------------------------------
    # Sigma schedule
    # ------------------------------------------------------------------

    def _get_sigma_schedule(self) -> list[float]:
        """Return the sigma schedule for denoising.

        - Distilled model (default): 8-step distilled schedule.
        - Non-distilled (dev) model: linear schedule with configurable steps.
        - ``num_inference_steps`` overrides the step count for dev models.
        """
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES

        # Heuristic: if a distilled LoRA is loaded, or the checkpoint name
        # contains 'distilled', use the 8-step distilled schedule.
        is_distilled = "distilled" in self._checkpoint_path.lower()
        if self._lora_path and "distilled" in self._lora_path.lower():
            is_distilled = True

        # GGUF files may be dev or distilled
        if self._gguf_path and "dev" in self._gguf_path.lower():
            # Dev base model: use distilled schedule ONLY if distilled LoRA is also loaded
            if self._lora_path and "distilled" in self._lora_path.lower():
                is_distilled = True
            else:
                is_distilled = False

        if is_distilled and self._num_inference_steps is None:
            return list(DISTILLED_SIGMA_VALUES)

        # Non-distilled or custom step count
        steps = self._num_inference_steps or 20
        schedule = _make_dev_sigmas(steps)
        logger.info("Using %d-step linear sigma schedule (dev/non-distilled)", steps)
        return schedule

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

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
        progress_callback: Any = None,
    ) -> None:
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
                progress_callback=progress_callback,
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
        progress_callback: Any = None,
    ) -> None:
        import torch

        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput
        from ltx_pipelines.utils.helpers import (
            cleanup_memory,
            denoise_audio_video,
            image_conditionings_by_replacing_latent,
            simple_denoising_func,
        )
        from ltx_pipelines.utils.samplers import euler_denoising_loop

        from services.ltx_pipeline_common import encode_video_output, video_chunks_number

        logger.info(
            "[low-vram] Starting generation: %dx%d, %d frames, tier=%s",
            width, height, num_frames, self.vram_manager.tier.value,
        )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        # ============================================================
        # Phase 1: Text encoding (with caching)
        # ============================================================
        import time as _time
        t_phase1 = _time.perf_counter()
        logger.info("[low-vram] Phase 1: Text encoding")

        if self._cached_text_encoder is None:
            logger.info("[low-vram] Loading text encoder from disk (first run)")
            text_encoder = self.model_ledger.text_encoder()
            self._quantize_text_encoder_fp8(text_encoder)
            self._cached_text_encoder = text_encoder
        else:
            logger.info("[low-vram] Using cached text encoder")
            text_encoder = self._cached_text_encoder

        te_block_swap = self._setup_text_encoder_block_swap(text_encoder)
        if te_block_swap is None:
            # Without block swapping, a cached encoder may have been fully offloaded
            # after the previous run. Move the whole module back before encoding.
            text_encoder.to(self.device)
        else:
            self._move_text_encoder_non_layers_to_gpu(text_encoder)

        # Patch device property so input_ids are created on GPU
        gemma = getattr(text_encoder, "model", None)
        if gemma is not None:
            _device = self.device

            class _DeviceOverride(type(gemma)):  # type: ignore[misc]
                @property
                def device(self_inner: Any) -> Any:  # type: ignore[override]
                    return _device

            gemma.__class__ = _DeviceOverride  # type: ignore[assignment]

        context_p = encode_text(text_encoder, prompts=[prompt])[0]
        video_context, audio_context = self._normalize_text_contexts(*context_p)

        # Offload text encoder to CPU (keep cached reference)
        if te_block_swap is not None:
            te_block_swap.offload_all()
        text_encoder.to("cpu")
        self.vram_manager.cleanup()
        logger.info("[low-vram] Phase 1 done: %.2fs", _time.perf_counter() - t_phase1)

        # ============================================================
        # Phase 2: Image conditioning (if i2v)
        # ============================================================
        output_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width, height=height, fps=frame_rate,
        )

        conditionings: list[Any] = []
        if images:
            t_phase2 = _time.perf_counter()
            logger.info("[low-vram] Phase 2: Image conditioning")
            if self._cached_video_encoder is None:
                video_encoder = self.model_ledger.video_encoder()
                self._cached_video_encoder = video_encoder
            else:
                video_encoder = self._cached_video_encoder
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
            self.vram_manager.cleanup()
            logger.info("[low-vram] Phase 2 done: %.2fs", _time.perf_counter() - t_phase2)

        # ============================================================
        # Phase 3: Denoising (transformer on GPU, with block swap)
        # ============================================================
        t_phase3 = _time.perf_counter()
        logger.info("[low-vram] Phase 3: Denoising (transformer)")

        if self._cached_transformer is None:
            logger.info("[low-vram] Loading transformer from disk (first run)")
            t_load = _time.perf_counter()
            if self._use_gguf:
                transformer = self._load_gguf_transformer()
            else:
                transformer = self.model_ledger.transformer()
            logger.info("[low-vram] Transformer loaded: %.2fs", _time.perf_counter() - t_load)
            self._cached_transformer = transformer
        else:
            logger.info("[low-vram] Using cached transformer")
            transformer = self._cached_transformer

        transformer = self._setup_block_swap_if_needed(transformer)

        if self._block_swap_wrapper is not None:
            self._move_non_block_parts_to_gpu(transformer)
        else:
            self.vram_manager.ensure_on_gpu("transformer", transformer)

        sigma_values = self._get_sigma_schedule()
        total_steps = len(sigma_values) - 1  # number of denoising steps
        sigmas = torch.Tensor(sigma_values).to(self.device)

        # Track denoising step for progress callback
        step_counter = [0]

        def denoising_loop(
            sigmas: torch.Tensor,
            video_state: Any,
            audio_state: Any,
            stepper: EulerDiffusionStep,
        ) -> tuple[Any, Any]:
            # Wrap denoise_fn to count steps
            base_denoise = simple_denoising_func(
                video_context=video_context,
                audio_context=audio_context,
                transformer=transformer,
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

        logger.info("[low-vram] Denoising done: %.2fs", _time.perf_counter() - t_phase3)

        # Offload transformer to CPU (keep cached reference)
        if self._block_swap_wrapper is not None:
            self._block_swap_wrapper.offload_all()
        else:
            self.vram_manager.offload_to_cpu("transformer", transformer)
        self._block_swap_wrapper = None
        self.vram_manager.cleanup()

        # ============================================================
        # Phase 4-6: VAE decode + encode to file
        # ============================================================
        t_phase4 = _time.perf_counter()
        logger.info("[low-vram] Phase 4: VAE video decode")
        tiling_config = self._get_adaptive_tiling_config()

        if self._cached_video_decoder is None:
            video_decoder = self.model_ledger.video_decoder()
            self._cached_video_decoder = video_decoder
        else:
            video_decoder = self._cached_video_decoder
        self.vram_manager.ensure_on_gpu("video_decoder", video_decoder)

        decoded_video = vae_decode_video(
            video_state.latent, video_decoder, tiling_config,
        )

        logger.info("[low-vram] Phase 5: Audio decode")
        if self._cached_audio_decoder is None:
            audio_decoder = self.model_ledger.audio_decoder()
            self._cached_audio_decoder = audio_decoder
        else:
            audio_decoder = self._cached_audio_decoder
        if self._cached_vocoder is None:
            vocoder = self.model_ledger.vocoder()
            self._cached_vocoder = vocoder
        else:
            vocoder = self._cached_vocoder
        self.vram_manager.ensure_on_gpu("audio_decoder", audio_decoder)
        self.vram_manager.ensure_on_gpu("vocoder", vocoder)

        decoded_audio = vae_decode_audio(
            audio_state.latent, audio_decoder, vocoder,
        )

        self.vram_manager.offload_to_cpu("audio_decoder", audio_decoder)
        self.vram_manager.offload_to_cpu("vocoder", vocoder)

        logger.info("[low-vram] Phase 6: Encoding video output")
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(
            video=decoded_video,
            audio=decoded_audio,
            fps=int(frame_rate),
            output_path=output_path,
            video_chunks_number_value=chunks,
        )

        self.vram_manager.offload_to_cpu("video_decoder", video_decoder)
        self.vram_manager.cleanup()
        logger.info("[low-vram] Phase 4-6 done: %.2fs", _time.perf_counter() - t_phase4)

        logger.info("[low-vram] Generation complete: %s", output_path)

    # ------------------------------------------------------------------
    # Tiling configuration
    # ------------------------------------------------------------------

    def _get_adaptive_tiling_config(self) -> Any:
        from ltx_core.model.video_vae import (
            SpatialTilingConfig,
            TemporalTilingConfig,
            TilingConfig,
        )

        from services.vram_manager.vram_manager import VRAMTier

        match self.vram_manager.tier:
            case VRAMTier.HIGH:
                return TilingConfig(
                    spatial_config=SpatialTilingConfig(
                        tile_size_in_pixels=384,
                        tile_overlap_in_pixels=64,
                    ),
                    temporal_config=TemporalTilingConfig(
                        tile_size_in_frames=48,
                        tile_overlap_in_frames=8,
                    ),
                )
            case VRAMTier.MEDIUM:
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

    # ------------------------------------------------------------------
    # Transformer non-block GPU placement
    # ------------------------------------------------------------------

    def _move_non_block_parts_to_gpu(self, transformer: torch.nn.Module) -> None:
        """Move non-transformer-block parts of the model to GPU."""
        inner = transformer
        for sub_name in ("velocity_model", "model", "inner_model"):
            sub = getattr(inner, sub_name, None)
            if sub is not None:
                inner = sub
                break

        for name, param in inner.named_parameters(recurse=False):
            param.data = param.data.to(self.device)
        for name, buf in inner.named_buffers(recurse=False):
            buf.data = buf.data.to(self.device)

        block_attr_names = {"transformer_blocks", "blocks", "layers", "encoder_layers"}
        for child_name, child in inner.named_children():
            if child_name not in block_attr_names:
                child.to(self.device)

        if inner is not transformer:
            for name, param in transformer.named_parameters(recurse=False):
                param.data = param.data.to(self.device)
            for name, buf in transformer.named_buffers(recurse=False):
                buf.data = buf.data.to(self.device)
            for child_name, child in transformer.named_children():
                if child is not inner:
                    child.to(self.device)

        logger.info("Moved non-block transformer parts to GPU")

    # ------------------------------------------------------------------
    # Text encoder FP8 quantisation + layerwise offload
    # ------------------------------------------------------------------

    @staticmethod
    def _quantize_text_encoder_fp8(text_encoder: torch.nn.Module) -> None:
        """Quantize all Linear layers in the text encoder to FP8.

        Skips layers that are already in FP8 (e.g. from a pre-quantized model).
        """
        import torch as _torch

        count = 0
        skipped = 0
        for child in text_encoder.modules():
            if not isinstance(child, _torch.nn.Linear):
                continue
            # Skip if already quantized
            if child.weight.dtype == _torch.float8_e4m3fn:
                skipped += 1
                continue
            child.weight.data = child.weight.data.to(_torch.float8_e4m3fn)
            if child.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                child.bias.data = child.bias.data.to(_torch.float8_e4m3fn)

            def _make_upcast_forward(lin: _torch.nn.Linear) -> Any:
                def _fwd(x: _torch.Tensor, **kw: Any) -> _torch.Tensor:
                    w = lin.weight.to(x.dtype)
                    b = lin.bias.to(x.dtype) if lin.bias is not None else None  # pyright: ignore[reportUnnecessaryComparison]
                    return _torch.nn.functional.linear(x, w, b)
                return _fwd

            child.forward = _make_upcast_forward(child)  # type: ignore[assignment]
            count += 1

        if skipped > 0:
            logger.info("Text encoder: %d layers already FP8, quantized %d more", skipped, count)
        else:
            logger.info("Quantized %d Linear layers to FP8 in text encoder", count)

    def _setup_text_encoder_block_swap(
        self, text_encoder: torch.nn.Module,
    ) -> "BlockSwapTransformerWrapper | None":
        """Apply block swap to Gemma language model layers."""
        from services.block_swap.block_swap import BlockSwapTransformerWrapper
        from services.vram_manager.vram_manager import VRAMTier

        gemma_model = getattr(text_encoder, "model", None)
        if gemma_model is None:
            return None

        lang_model = getattr(gemma_model, "language_model", None)
        if lang_model is None:
            return None

        layers = self._find_text_encoder_layers(lang_model)
        if layers is None or len(layers) < 2:
            return None

        # Text encoder VRAM policy: never keep the full Gemma stack on GPU.
        # Even on 24 GB cards, lm/attention activations and temporary upcasts can OOM.
        # Scale conservatively by tier and current free VRAM.
        free_vram_mb = self.vram_manager.get_free_vram_mb()
        match self.vram_manager.tier:
            case VRAMTier.HIGH:
                keep_on_gpu = 6
            case VRAMTier.MEDIUM:
                keep_on_gpu = 4
            case VRAMTier.LOW:
                keep_on_gpu = 2
            case _:
                keep_on_gpu = 1

        # Tighten further if currently little VRAM is free.
        if free_vram_mb < 10_000:
            keep_on_gpu = min(keep_on_gpu, 2)
        if free_vram_mb < 6_000:
            keep_on_gpu = 1

        keep_on_gpu = max(1, min(keep_on_gpu, len(layers) - 1))

        wrapper = BlockSwapTransformerWrapper(
            transformer=lang_model,
            device=self.device,
            blocks_to_keep_on_gpu=keep_on_gpu,
            use_async_prefetch=True,
        )

        if wrapper.block_count == 0:
            return None

        logger.info("Text encoder block swap: %d layers, %d on GPU", wrapper.block_count, keep_on_gpu)
        return wrapper

    def _find_text_encoder_layers(self, lang_model: torch.nn.Module) -> Any:
        """Find Gemma decoder layers across direct and nested layouts."""
        for target in (lang_model, getattr(lang_model, "model", None), getattr(lang_model, "inner_model", None)):
            if target is None:
                continue
            layers = getattr(target, "layers", None)
            if layers is not None and hasattr(layers, "__len__"):
                return layers
        return None

    def _move_text_encoder_non_layers_to_gpu(self, text_encoder: torch.nn.Module) -> None:
        """Move text encoder non-layer components to GPU."""
        gemma_model = getattr(text_encoder, "model", None)
        if gemma_model is None:
            text_encoder.to(self.device)
            return

        lang_model = getattr(gemma_model, "language_model", None)
        if lang_model is None:
            lang_model = getattr(getattr(gemma_model, "model", None), "language_model", None)

        if lang_model is not None:
            for child_name, child in lang_model.named_children():
                if child_name != "layers":
                    child.to(self.device)
            for name, param in lang_model.named_parameters(recurse=False):
                param.data = param.data.to(self.device)
            for name, buf in lang_model.named_buffers(recurse=False):
                buf.data = buf.data.to(self.device)

        # Keep lm_head off GPU. Prompt encoding only needs hidden_states, and moving
        # lm_head to GPU can trigger multi-GB temporary allocations during upcast.
        lm_head = getattr(gemma_model, "lm_head", None)
        if lm_head is not None:
            lm_head.to("cpu")

        inner_model = getattr(gemma_model, "model", None)
        if inner_model is not None:
            for child_name, child in inner_model.named_children():
                if child_name in ("language_model", "text_model", "vision_tower"):
                    continue
                child.to(self.device)

        for child_name, child in text_encoder.named_children():
            if child_name != "model":
                child.to(self.device)

        for name, param in text_encoder.named_parameters(recurse=False):
            param.data = param.data.to(self.device)
        for name, buf in text_encoder.named_buffers(recurse=False):
            buf.data = buf.data.to(self.device)

        logger.info("Moved text encoder non-layer parts to GPU (skipping vision_tower)")

    def _normalize_text_contexts(
        self,
        video_context: torch.Tensor,
        audio_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Match text-conditioning tensors to transformer compute dtype/device."""
        video_context = video_context.to(device=self.device, dtype=self.dtype)
        if audio_context is not None:
            audio_context = audio_context.to(device=self.device, dtype=self.dtype)
        return video_context, audio_context

    # ------------------------------------------------------------------
    # Warmup & compile stubs
    # ------------------------------------------------------------------

    def warmup(self, output_path: str) -> None:
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
        logger.info(
            "Skipping torch.compile for low-VRAM pipeline "
            "(incompatible with block swap / sequential offloading)"
        )
