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
    from services.block_swap.fast_block_swap import FastBlockSwapWrapper
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


def _build_split_video_vae_sd_ops(kind: str) -> Any:
    """Accept both monolithic and split-file key layouts for video VAE weights."""
    from ltx_core.loader.sd_ops import SDOps

    if kind not in {"encoder", "decoder"}:
        raise ValueError(f"Unsupported split video VAE kind: {kind}")

    base_prefix = f"{kind}."
    nested_prefix = f"vae.{kind}."
    return (
        SDOps(f"SPLIT_VIDEO_VAE_{kind.upper()}_SD_OPS")
        .with_matching(prefix=base_prefix)
        .with_matching(prefix=nested_prefix)
        .with_matching(prefix="per_channel_statistics.")
        .with_matching(prefix="vae.per_channel_statistics.")
        .with_replacement(nested_prefix, "")
        .with_replacement(base_prefix, "")
        .with_replacement("vae.per_channel_statistics.", "per_channel_statistics.")
    )


def _make_dev_sigmas(steps: int) -> list[float]:
    """Create the correct sigma schedule for the non-distilled (dev) model.

    Uses LTX2Scheduler which applies token-count-dependent shifting and
    stretching. A simple linear schedule produces noise with dev models.
    """
    from ltx_pipelines.ti2vid_one_stage import LTX2Scheduler
    scheduler = LTX2Scheduler()
    sigmas = scheduler.execute(steps=steps)
    return sigmas.tolist()


def _get_official_dev_guidance_defaults() -> tuple[str, Any, Any]:
    """Return the official LTX 2.3 dev negative prompt and guider params."""
    from ltx_core.components.guiders import MultiModalGuiderParams

    try:
        from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, LTX_2_3_PARAMS

        return (
            DEFAULT_NEGATIVE_PROMPT,
            LTX_2_3_PARAMS.video_guider_params,
            LTX_2_3_PARAMS.audio_guider_params,
        )
    except Exception:
        return (
            "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
            "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
            "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
            "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
            "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
            "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
            "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
            "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
            "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
            "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
            "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.",
            MultiModalGuiderParams(
                cfg_scale=3.0,
                stg_scale=1.0,
                rescale_scale=0.7,
                modality_scale=3.0,
                skip_step=0,
                stg_blocks=[28],
            ),
            MultiModalGuiderParams(
                cfg_scale=7.0,
                stg_scale=1.0,
                rescale_scale=0.7,
                modality_scale=3.0,
                skip_step=0,
                stg_blocks=[28],
            ),
        )


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
                    # Cast to common dtype to prevent SDPA dtype mismatch errors
                    # (GGUF models can produce mixed bfloat16/float32 tensors)
                    if q.dtype != k.dtype or k.dtype != v.dtype:
                        common_dtype = q.dtype
                        k = k.to(common_dtype)
                        v = v.to(common_dtype)
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
        use_upscaler: bool = False,
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
            use_upscaler=use_upscaler,
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
        use_upscaler: bool = False,
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
        self._use_upscaler = use_upscaler

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
        self._block_swap_wrapper: FastBlockSwapWrapper | None = None
        self._te_block_swap_wrapper: FastBlockSwapWrapper | None = None
        self._use_gguf = False

        # Cached model references — avoid reloading from disk each generation
        self._cached_text_encoder: Any = None
        self._cached_transformer: Any = None
        self._cached_video_decoder: Any = None
        self._cached_audio_decoder: Any = None
        self._cached_vocoder: Any = None
        self._cached_video_encoder: Any = None
        self._cached_spatial_upsampler: Any = None

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

        # GGUF is already quantized. Applying the FP8 builder transforms on top
        # can corrupt the loaded transformer and produce garbage/noisy output.
        # Keep FP8 only for safetensors checkpoints.
        quantization = None
        if device_supports_fp8(self.device) and not self._use_gguf:
            from ltx_core.quantization import QuantizationPolicy

            quantization = QuantizationPolicy.fp8_cast()
            logger.info("FP8 quantization enabled")
        elif self._use_gguf:
            logger.info("Skipping FP8 quantization for GGUF transformer")

        # Build LoRA list from primary + extras.
        # Note: LoRAs must be compatible with the model architecture.
        # The LTX 2.3 (22B) model uses LoRAs trained for 22B dimensions,
        # while LTX 2.0 (19B) LoRAs have different tensor sizes and will fail.
        #
        # IMPORTANT: Only pre-fuse LoRAs for safetensors mode. For GGUF mode,
        # LoRAs are applied at inference time via forward hooks in
        # _install_lora_hooks(). Passing them to ModelLedger would corrupt
        # the transformer builder's module_ops, causing the GGUF weights to
        # be loaded into a LoRA-modified skeleton → noise/garbage output.
        loras = None
        if not self._use_gguf:
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
            spatial_upsampler_path=self._upsampler_path,
            loras=loras,
            quantization=quantization,
        )
        self._maybe_configure_split_component_builders()

        # Optional override for a quantized text encoder variant.
        # We still use the base gemma root for tokenizer + processor config.
        if self._text_encoder_variant_path and self._gemma_root and Path(self._text_encoder_variant_path).exists():
            try:
                variant_path = str(Path(self._text_encoder_variant_path))
                if not hasattr(self.model_ledger, "_default_text_encoder_builder"):
                    setattr(
                        self.model_ledger,
                        "_default_text_encoder_builder",
                        self.model_ledger.text_encoder_builder,
                    )
                if variant_path.lower().endswith(".gguf"):
                    from services.text_encoder.gguf_text_encoder_builder import (
                        GGUFGemmaTextEncoderBuilder,
                    )

                    self.model_ledger.text_encoder_builder = GGUFGemmaTextEncoderBuilder(
                        base_builder=self.model_ledger.text_encoder_builder,
                        checkpoint_path=getattr(self.model_ledger.text_encoder_builder, "model_path", str(self._checkpoint_path)),
                        gguf_path=variant_path,
                    )
                else:
                    from ltx_core.loader.registry import DummyRegistry
                    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
                    from ltx_core.text_encoders.gemma import (
                        AV_GEMMA_TEXT_ENCODER_KEY_OPS,
                        GEMMA_MODEL_OPS,
                        GemmaTextEncoderConfigurator,
                        module_ops_from_gemma_root,
                    )

                    variant_model_path: tuple[str, ...]
                    variant_path_obj = Path(variant_path)
                    if variant_path_obj.is_dir():
                        shard_paths = sorted(str(path) for path in variant_path_obj.glob("model-*.safetensors"))
                        if not shard_paths:
                            raise ValueError(
                                f"Text encoder variant directory contains no model shards: {variant_path}"
                            )
                        variant_model_path = (str(self._checkpoint_path), *shard_paths)
                        logger.info(
                            "Using sharded text encoder variant directory: %s (%d shards)",
                            variant_path,
                            len(shard_paths),
                        )
                    else:
                        variant_model_path = (str(self._checkpoint_path), variant_path)

                    module_ops = module_ops_from_gemma_root(self._gemma_root)
                    self.model_ledger.text_encoder_builder = Builder(
                        model_path=variant_model_path,
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

    def _maybe_configure_split_component_builders(self) -> None:
        """Use split LTX component files when the checkpoint is transformer-only."""
        checkpoint_path = Path(self._checkpoint_path)
        if not checkpoint_path.exists():
            return
        if checkpoint_path.suffix.lower() != ".safetensors":
            return
        if checkpoint_path.stat().st_size > 10_000_000_000:
            return

        models_dir = checkpoint_path.parent.parent
        video_candidates = (
            models_dir / "vae" / "LTX23_video_vae_bf16.safetensors",
            models_dir / "vae" / "LTX2_video_vae_bf16.safetensors",
        )
        audio_candidates = (
            models_dir / "vae" / "LTX23_audio_vae_bf16.safetensors",
            models_dir / "vae" / "LTX2_audio_vae_bf16.safetensors",
        )
        text_projection_candidates = (
            models_dir / "text_encoders" / "ltx-2.3_text_projection_bf16.safetensors",
            models_dir / "text_encoders" / "ltx-2-19b-embeddings_connector_dev_bf16.safetensors",
        )

        video_split = next((path for path in video_candidates if path.exists()), None)
        audio_split = next((path for path in audio_candidates if path.exists()), None)
        text_projection = next((path for path in text_projection_candidates if path.exists()), None)
        if video_split is None or audio_split is None:
            return

        from ltx_core.loader.registry import DummyRegistry
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
        from ltx_core.model.audio_vae.model_configurator import (
            AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
            AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            AudioDecoderConfigurator,
            AudioEncoderConfigurator,
            VOCODER_COMFY_KEYS_FILTER,
            VocoderConfigurator,
        )
        from ltx_core.model.video_vae.model_configurator import (
            VideoDecoderConfigurator,
            VideoEncoderConfigurator,
        )

        registry = getattr(self.model_ledger, "registry", DummyRegistry())
        self.model_ledger.vae_decoder_builder = Builder(
            model_path=str(video_split),
            model_class_configurator=VideoDecoderConfigurator,
            model_sd_ops=_build_split_video_vae_sd_ops("decoder"),
            registry=registry,
        )
        self.model_ledger.vae_encoder_builder = Builder(
            model_path=str(video_split),
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=_build_split_video_vae_sd_ops("encoder"),
            registry=registry,
        )
        self.model_ledger.audio_encoder_builder = Builder(
            model_path=str(audio_split),
            model_class_configurator=AudioEncoderConfigurator,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry,
        )
        self.model_ledger.audio_decoder_builder = Builder(
            model_path=str(audio_split),
            model_class_configurator=AudioDecoderConfigurator,
            model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
            registry=registry,
        )
        self.model_ledger.vocoder_builder = Builder(
            model_path=str(audio_split),
            model_class_configurator=VocoderConfigurator,
            model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
            registry=registry,
        )

        if text_projection is not None and self._gemma_root:
            try:
                from ltx_core.loader.registry import DummyRegistry as _DummyRegistry
                from ltx_core.text_encoders.gemma import (
                    AV_GEMMA_TEXT_ENCODER_KEY_OPS,
                    GEMMA_MODEL_OPS,
                    GemmaTextEncoderConfigurator,
                    module_ops_from_gemma_root,
                )

                model_folder = next(path.parent for path in Path(self._gemma_root).rglob("model*.safetensors"))
                weight_paths = [str(path) for path in model_folder.rglob("*.safetensors")]
                module_ops = module_ops_from_gemma_root(self._gemma_root)
                self.model_ledger.text_encoder_builder = Builder(
                    model_path=(str(text_projection), *weight_paths),
                    model_class_configurator=GemmaTextEncoderConfigurator,
                    model_sd_ops=AV_GEMMA_TEXT_ENCODER_KEY_OPS,
                    registry=_DummyRegistry(),
                    module_ops=(GEMMA_MODEL_OPS, *module_ops),
                )
            except Exception:
                logger.warning("Failed to configure split text projection builder", exc_info=True)

        logger.warning(
            "Using split LTX component weights with transformer-only checkpoint: video=%s audio=%s text_projection=%s",
            video_split,
            audio_split,
            text_projection,
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
            return transformer

        # Reuse existing wrapper — hooks are already installed on the modules
        if self._block_swap_wrapper is not None:
            self._block_swap_wrapper.restore_gpu_blocks()
            logger.info("Reusing existing block swap wrapper")
            return transformer

        blocks_on_gpu = self.vram_manager.block_swap_keep_on_gpu
        logger.info("Setting up block swap: keeping %d blocks on GPU", blocks_on_gpu)

        self._block_swap_wrapper = FastBlockSwapWrapper(
            transformer=transformer,
            device=self.device,
            blocks_to_keep_on_gpu=blocks_on_gpu,
            prefetch_distance=2,
        )

        if self._block_swap_wrapper.block_count == 0:
            logger.warning("Block swap found 0 blocks — falling back to sequential offloading")
            self._block_swap_wrapper = None

        return transformer

    def _prepare_a2v_transformer_for_denoise(
        self, transformer: torch.nn.Module
    ) -> tuple[torch.nn.Module, bool]:
        """Prepare the A2V transformer for denoising.

        Returns the possibly wrapped transformer plus whether block swap is
        actively in use for subsequent offload / bring-back decisions.
        """
        self.vram_manager.cleanup()
        transformer = self._setup_block_swap_if_needed(transformer)
        using_block_swap = self._block_swap_wrapper is not None

        try:
            # Prefer full placement when block swap is not active. If block
            # swap is active, only keep the non-block pieces resident and let
            # the wrapper manage block residency.
            if using_block_swap:
                self._move_non_block_parts_to_gpu(transformer)
            else:
                self.vram_manager.ensure_on_gpu("transformer", transformer)
        except Exception as exc:
            if "out of memory" not in str(exc).lower():
                raise
            logger.warning(
                "[low-vram-a2v] Full transformer GPU placement OOM; falling back to block swap"
            )
            self.vram_manager.cleanup()
            transformer = self._setup_block_swap_if_needed(transformer)
            if self._block_swap_wrapper is None:
                raise
            self._move_non_block_parts_to_gpu(transformer)
            using_block_swap = True
        else:
            if using_block_swap and self._block_swap_wrapper is not None:
                # Ensure hooks state doesn't keep stale residency assumptions
                # before the actual denoising loop starts.
                self._block_swap_wrapper.offload_all()

        self.vram_manager.cleanup()
        return transformer, using_block_swap

    def _load_gguf_transformer(self) -> Any:
        """Load transformer from GGUF file instead of safetensors.

        We can't rely on ``ModelLedger.transformer()`` here because the checkpoint
        shim used for GGUF mode may not have transformer config metadata at all.
        Build the architecture from the GGUF's embedded config, materialize any
        meta tensors with ``to_empty()``, then load the GGUF state dict.
        """
        if self._gguf_path is None:
            raise RuntimeError("No GGUF path configured")

        import time as _time
        import torch
        from dataclasses import replace
        from ltx_core.loader import SDOps
        from ltx_core.model.transformer import X0Model
        from services.gguf_loader.gguf_lazy_loader import (
            assign_gguf_linear_weights,
            load_gguf_lazy_state_dict,
            remap_gguf_keys,
            replace_linear_with_gguf,
        )

        logger.info("Loading GGUF transformer from %s", self._gguf_path)
        t_load = _time.perf_counter()
        state_dict = load_gguf_lazy_state_dict(
            self._gguf_path,
            device="cpu",
        )
        state_dict = remap_gguf_keys(state_dict)
        logger.info("GGUF state dict loaded lazily in %.2fs", _time.perf_counter() - t_load)

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

        config = builder.model_config()
        base_model = builder.meta_model(config, builder.module_ops)
        replace_linear_with_gguf(base_model, state_dict, compute_dtype=self.dtype)
        base_model, remaining_state_dict = assign_gguf_linear_weights(base_model, state_dict)
        base_model.load_state_dict(remaining_state_dict, strict=False, assign=True)

        # Cast all non-GGUF parameters (norms, biases, embeddings) to the
        # pipeline compute dtype.  Without this, norm layers stay in float32
        # (the GGUF default for non-quantized tensors) while GGUF linears
        # compute in bfloat16, causing dtype mismatches in attention.
        from services.gguf_loader.gguf_lazy_loader import GGUFParameter
        for param in base_model.parameters():
            if isinstance(param, GGUFParameter):
                continue  # Skip quantized GGUF weights
            if param.dtype != self.dtype and param.dtype.is_floating_point:
                param.data = param.data.to(self.dtype)
        for buf in base_model.buffers():
            if buf.dtype.is_floating_point and buf.dtype != self.dtype:
                buf.data = buf.data.to(self.dtype)

        transformer = X0Model(base_model).eval()

        # Install LoRA hooks for GGUF mode — LoRAs can't be pre-fused into
        # GGUF weights, so we apply them at inference time via forward hooks.
        # Skip LoRA when the GGUF is already a distilled model (applying the
        # distilled LoRA on top of an already-distilled model corrupts output).
        gguf_name = Path(self._gguf_path).name.lower()
        gguf_is_distilled = "distilled" in gguf_name
        if gguf_is_distilled and self._lora_path and "distilled" in Path(self._lora_path).name.lower():
            logger.info("Skipping distilled LoRA — GGUF model is already distilled: %s", gguf_name)
        else:
            self._install_lora_hooks(transformer)

        logger.info("GGUF transformer loaded successfully")
        return transformer

    # ------------------------------------------------------------------
    # LoRA hooks for GGUF mode (apply at inference, not pre-fuse)
    # ------------------------------------------------------------------

    def _install_lora_hooks(self, transformer: Any) -> None:
        """Install inference-time LoRA hooks on the transformer.

        Instead of pre-fusing LoRA into weights (which requires dequant + fuse + requant),
        we apply LoRA as: output += (input @ A) @ B * strength during forward.

        Guards against duplicate installation when the transformer is cached.
        """
        import torch as _torch

        # Guard against duplicate hook installation on cached transformer
        if getattr(transformer, '_lora_hooks_installed', False):
            logger.info("LoRA hooks already installed on cached transformer — skipping")
            return

        lora_entries = self._collect_loras()
        if not lora_entries:
            return

        import safetensors

        for lora_entry in lora_entries:
            lora_path = lora_entry.path
            lora_strength = lora_entry.strength

            if lora_strength == 0:
                continue

            import time as _time
            t0 = _time.perf_counter()

            # Load LoRA state dict
            lora_sd: dict[str, _torch.Tensor] = {}
            with safetensors.safe_open(lora_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    mapped_key = key
                    if "diffusion_model." in mapped_key:
                        mapped_key = mapped_key.replace("diffusion_model.", "")
                    lora_sd[mapped_key] = f.get_tensor(key)

            # Find A/B pairs and install hooks
            hook_count = 0
            pairs: dict[str, dict[str, _torch.Tensor]] = {}
            for key, tensor in lora_sd.items():
                if ".lora_A.weight" in key:
                    base = key.replace(".lora_A.weight", "")
                    pairs.setdefault(base, {})["A"] = tensor
                elif ".lora_B.weight" in key:
                    base = key.replace(".lora_B.weight", "")
                    pairs.setdefault(base, {})["B"] = tensor

            inner_model = transformer
            for attr in ("velocity_model", "model", "inner_model"):
                sub = getattr(inner_model, attr, None)
                if sub is not None:
                    inner_model = sub
                    break

            for base_key, ab in pairs.items():
                if "A" not in ab or "B" not in ab:
                    continue

                # Find the target module
                parts = base_key.split(".")
                module = inner_model
                found = True
                for part in parts[:-1]:
                    module = getattr(module, part, None)
                    if module is None:
                        found = False
                        break
                if not found or module is None:
                    continue

                target_name = parts[-1]
                target = getattr(module, target_name, None)
                if target is None or not isinstance(target, _torch.nn.Module):
                    continue

                lora_A = ab["A"].to(dtype=self.dtype)
                lora_B = ab["B"].to(dtype=self.dtype)
                strength = lora_strength

                def make_hook(A: _torch.Tensor, B: _torch.Tensor, s: float) -> Any:
                    def hook(mod: Any, inp: Any, out: _torch.Tensor) -> _torch.Tensor:
                        x = inp[0] if isinstance(inp, tuple) else inp
                        # LoRA: out += (x @ A^T) @ B^T * strength
                        device = out.device
                        a = A.to(device)
                        b = B.to(device)
                        lora_out = (x.to(self.dtype) @ a.T) @ b.T * s
                        return out + lora_out.to(out.dtype)
                    return hook

                target.register_forward_hook(make_hook(lora_A, lora_B, strength))
                hook_count += 1

            logger.info(
                "LoRA hooks installed: %s (%d hooks, strength=%.2f) in %.2fs",
                Path(lora_path).name, hook_count, lora_strength,
                _time.perf_counter() - t0,
            )

        transformer._lora_hooks_installed = True  # type: ignore[attr-defined]

    def _is_distilled_mode(self) -> bool:
        """Return True if the current configuration should use distilled denoising.

        Distilled mode means:
        - The base model is a distilled checkpoint/GGUF, OR
        - A distilled LoRA is loaded on top of a dev model.

        In distilled mode, guidance is baked into the model and no CFG is needed.
        In dev mode, classifier-free guidance (CFG) must be applied explicitly.
        """
        is_distilled = "distilled" in self._checkpoint_path.lower()
        if self._gguf_path and "distilled" in self._gguf_path.lower():
            is_distilled = True
        if self._lora_path and "distilled" in self._lora_path.lower():
            is_distilled = True
        if self._gguf_path and "dev" in self._gguf_path.lower():
            if self._lora_path and "distilled" in self._lora_path.lower():
                is_distilled = True
            else:
                is_distilled = False
        return is_distilled

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

        # Heuristic: if a distilled GGUF/LoRA/checkpoint is selected, use the
        # distilled sigma table by default. Distilled runs should not silently
        # fall back to the dev linear schedule just because settings happen to
        # store an explicit step count of 8.
        is_distilled = "distilled" in self._checkpoint_path.lower()
        if self._gguf_path and "distilled" in self._gguf_path.lower():
            is_distilled = True
        if self._lora_path and "distilled" in self._lora_path.lower():
            is_distilled = True

        # GGUF files may be dev or distilled
        if self._gguf_path and "dev" in self._gguf_path.lower():
            # Dev base model: use distilled schedule ONLY if distilled LoRA is also loaded
            if self._lora_path and "distilled" in self._lora_path.lower():
                is_distilled = True
            else:
                is_distilled = False

        distilled_default_steps = len(DISTILLED_SIGMA_VALUES) - 1
        if is_distilled:
            # Distilled models MUST use the distilled sigma schedule. Using a
            # linear dev schedule with a distilled model produces noise. The
            # number of steps is fixed by the distilled training and cannot be
            # overridden by the user's step-count setting.
            logger.info("Using %d-step distilled sigma schedule", distilled_default_steps)
            return list(DISTILLED_SIGMA_VALUES)

        # Non-distilled (dev) model — use configurable step count
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
        negative_prompt: str = "",
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
                negative_prompt=negative_prompt,
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
        negative_prompt: str = "",
    ) -> None:
        import torch

        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from ltx_core.model.upsampler import upsample_video
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput
        from ltx_pipelines.utils.helpers import (
            cleanup_memory,
            denoise_audio_video,
            image_conditionings_by_replacing_latent,
            multi_modal_guider_factory_denoising_func,
            simple_denoising_func,
        )
        from ltx_core.components.guiders import MultiModalGuiderFactory, MultiModalGuiderParams
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
            if not getattr(text_encoder, "_ltx_gguf_text_encoder", False):
                self._quantize_text_encoder_fp8(text_encoder)
            else:
                logger.info("Skipping FP8 quantization for GGUF text encoder")
            self._cached_text_encoder = text_encoder
        else:
            logger.info("[low-vram] Using cached text encoder")
            text_encoder = self._cached_text_encoder

        te_block_swap = self._setup_text_encoder_block_swap_cached(text_encoder)
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
            # Avoid re-wrapping the class on every generation (would create
            # an ever-deepening class hierarchy). Only patch once.
            if not getattr(gemma, "_device_override_applied", False):
                class _DeviceOverride(type(gemma)):  # type: ignore[misc]
                    @property
                    def device(self_inner: Any) -> Any:  # type: ignore[override]
                        return _device

                gemma.__class__ = _DeviceOverride  # type: ignore[assignment]
                gemma._device_override_applied = True  # type: ignore[attr-defined]

        context_p = encode_text(text_encoder, prompts=[prompt])[0]
        video_context, audio_context = self._normalize_text_contexts(*context_p)
        logger.info(
            "[low-vram] Text encoding result: video_context shape=%s dtype=%s "
            "mean=%.6f std=%.6f min=%.6f max=%.6f device=%s",
            video_context.shape, video_context.dtype,
            video_context.float().mean().item(), video_context.float().std().item(),
            video_context.float().min().item(), video_context.float().max().item(),
            video_context.device,
        )

        # Dev (non-distilled) models need the official LTX 2.3 CFG/STG setup.
        use_cfg = not self._is_distilled_mode()
        neg_video_context: torch.Tensor | None = None
        neg_audio_context: torch.Tensor | None = None
        dev_negative_prompt, video_guider_defaults, audio_guider_defaults = _get_official_dev_guidance_defaults()
        if use_cfg:
            resolved_negative_prompt = negative_prompt.strip() or dev_negative_prompt
            neg_context_p = encode_text(text_encoder, prompts=[resolved_negative_prompt])[0]
            neg_video_context, neg_audio_context = self._normalize_text_contexts(*neg_context_p)
            logger.info(
                "[low-vram] Dev mode: encoded official negative prompt for CFG/STG "
                "(cfg=%.2f stg=%.2f stg_blocks=%s)",
                video_guider_defaults.cfg_scale,
                video_guider_defaults.stg_scale,
                list(video_guider_defaults.stg_blocks),
            )

        # Offload text encoder to CPU (keep cached reference)
        if te_block_swap is not None:
            te_block_swap.offload_all()
        else:
            text_encoder.to("cpu")
        self.vram_manager.cleanup()
        self._log_vram_usage("after Phase 1")
        logger.info("[low-vram] Phase 1 done: %.2fs", _time.perf_counter() - t_phase1)

        # ============================================================
        # Phase 2: Image conditioning (if i2v)
        # ============================================================
        target_output_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width, height=height, fps=frame_rate,
        )
        stage_1_output_shape = target_output_shape
        if self._use_upscaler:
            stage_1_output_shape = VideoPixelShape(
                batch=1,
                frames=num_frames,
                width=width // 2,
                height=height // 2,
                fps=frame_rate,
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
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=video_encoder,
                dtype=self.dtype,
                device=self.device,
            )

            self.vram_manager.offload_to_cpu("video_encoder", video_encoder)
            self.vram_manager.cleanup()
            self._log_vram_usage("after Phase 2")
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

        self.vram_manager.cleanup()
        self._log_vram_usage("before Phase 3 block swap")
        transformer = self._setup_block_swap_if_needed(transformer)

        if self._block_swap_wrapper is not None:
            self._move_non_block_parts_to_gpu(transformer)
        else:
            self.vram_manager.ensure_on_gpu("transformer", transformer)
        self.vram_manager.cleanup()
        self._log_vram_usage("before Phase 3 denoise")

        sigma_values = self._get_sigma_schedule()
        total_steps = len(sigma_values) - 1  # number of denoising steps
        sigmas = torch.tensor(sigma_values, dtype=torch.float32, device=self.device)
        logger.info(
            "[low-vram] Sigma schedule: %d steps, values=%s",
            total_steps, [f'{s:.6f}' for s in sigma_values],
        )

        # Track denoising step for progress callback
        step_counter = [0]

        def denoising_loop(
            sigmas: torch.Tensor,
            video_state: Any,
            audio_state: Any,
            stepper: EulerDiffusionStep,
        ) -> tuple[Any, Any]:
            # Use CFG for dev (non-distilled) models, matching TI2VidOneStagePipeline
            if use_cfg and neg_video_context is not None:
                video_guider_factory = MultiModalGuiderFactory.constant(
                    video_guider_defaults,
                    negative_context=neg_video_context,
                )
                audio_guider_factory = MultiModalGuiderFactory.constant(
                    audio_guider_defaults,
                    negative_context=neg_audio_context,
                )
                base_denoise = multi_modal_guider_factory_denoising_func(
                    video_guider_factory=video_guider_factory,
                    audio_guider_factory=audio_guider_factory,
                    v_context=video_context,
                    a_context=audio_context,
                    transformer=transformer,
                )
            else:
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
            output_shape=stage_1_output_shape,
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

        if self._use_upscaler:
            t_phase3b = _time.perf_counter()
            logger.info("[low-vram] Phase 3b: 2x upscaler refinement")

            if self._cached_video_encoder is None:
                video_encoder = self.model_ledger.video_encoder()
                self._cached_video_encoder = video_encoder
            else:
                video_encoder = self._cached_video_encoder
            self.vram_manager.ensure_on_gpu("video_encoder", video_encoder)

            if self._cached_spatial_upsampler is None:
                spatial_upsampler = self.model_ledger.spatial_upsampler()
                self._cached_spatial_upsampler = spatial_upsampler
            else:
                spatial_upsampler = self._cached_spatial_upsampler
            self.vram_manager.ensure_on_gpu("spatial_upsampler", spatial_upsampler)

            upscaled_video_latent = upsample_video(
                latent=video_state.latent[:1],
                video_encoder=video_encoder,
                upsampler=spatial_upsampler,
            )

            stage_2_conditionings: list[Any] = []
            if images:
                ltx_images = [_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images]
                stage_2_conditionings = image_conditionings_by_replacing_latent(
                    images=ltx_images,
                    height=target_output_shape.height,
                    width=target_output_shape.width,
                    video_encoder=video_encoder,
                    dtype=self.dtype,
                    device=self.device,
                )

            stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
            total_steps += len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1
            video_state, audio_state = denoise_audio_video(
                output_shape=target_output_shape,
                conditionings=stage_2_conditionings,
                noiser=noiser,
                sigmas=stage_2_sigmas,
                stepper=stepper,
                denoising_loop_fn=cast(Any, denoising_loop),
                components=self.pipeline_components,
                dtype=self.dtype,
                device=self.device,
                noise_scale=stage_2_sigmas[0],
                initial_video_latent=upscaled_video_latent,
                initial_audio_latent=audio_state.latent,
            )

            self.vram_manager.offload_to_cpu("video_encoder", video_encoder)
            self.vram_manager.offload_to_cpu("spatial_upsampler", spatial_upsampler)
            self.vram_manager.cleanup()
            self._log_vram_usage("after Phase 3b")
            logger.info("[low-vram] Phase 3b done: %.2fs", _time.perf_counter() - t_phase3b)

        # Offload transformer to CPU (keep cached reference)
        if self._block_swap_wrapper is not None:
            self._block_swap_wrapper.offload_all()
        else:
            self.vram_manager.offload_to_cpu("transformer", transformer)
        # Don't clear self._block_swap_wrapper — reuse it next generation
        self.vram_manager.cleanup()
        self._log_vram_usage("after Phase 3")

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

        # ``vae_decode_video`` returns a lazy iterator. Keep the decoder resident
        # until Phase 6 consumes it during encode_video_output().
        self._log_vram_usage("after Phase 4 setup")

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
        self.vram_manager.cleanup()
        self._log_vram_usage("after Phase 5")

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
        self._log_vram_usage("after Phase 6")
        logger.info("[low-vram] Phase 4-6 done: %.2fs", _time.perf_counter() - t_phase4)

        logger.info("[low-vram] Generation complete: %s", output_path)

    # ------------------------------------------------------------------
    # A2V generation (reuses all low-VRAM infrastructure)
    # ------------------------------------------------------------------

    def generate_a2v(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[Any],
        output_path: str,
        audio_path: str,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        progress_callback: Any = None,
        negative_prompt: str = "",
    ) -> None:
        import torch

        with torch.inference_mode():
            self._generate_a2v_impl(
                prompt=prompt,
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
                negative_prompt=negative_prompt,
            )

    def _generate_a2v_impl(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[Any],
        output_path: str,
        audio_path: str,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        progress_callback: Any = None,
        negative_prompt: str = "",
    ) -> None:
        """A2V generation using the same low-VRAM infrastructure as T2V/I2V.

        Key differences from normal generation:
        - Encodes audio input as frozen latent conditioning
        - Uses denoise_video_only (audio is frozen, not jointly denoised)
        - Returns original audio (not VAE-decoded) for fidelity
        - Always uses two-stage (half-res + refinement) like the original A2V
        """
        import torch
        import time as _time

        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
        from ltx_core.model.upsampler import upsample_video
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
        from ltx_pipelines.utils.helpers import (
            cleanup_memory,
            denoise_video_only,
            image_conditionings_by_replacing_latent,
            multi_modal_guider_factory_denoising_func,
            simple_denoising_func,
        )
        from ltx_pipelines.utils.media_io import decode_audio_from_file
        from ltx_pipelines.utils.samplers import euler_denoising_loop
        from services.ltx_pipeline_common import encode_video_output, video_chunks_number

        logger.info(
            "[low-vram-a2v] Starting A2V generation: %dx%d, %d frames, tier=%s",
            width, height, num_frames, self.vram_manager.tier.value,
        )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        # ============================================================
        # Phase 1: Text encoding (reuses cached text encoder)
        # ============================================================
        t_phase1 = _time.perf_counter()
        logger.info("[low-vram-a2v] Phase 1: Text encoding")

        if self._cached_text_encoder is None:
            logger.info("[low-vram-a2v] Loading text encoder from disk")
            text_encoder = self.model_ledger.text_encoder()
            if not getattr(text_encoder, "_ltx_gguf_text_encoder", False):
                self._quantize_text_encoder_fp8(text_encoder)
            self._cached_text_encoder = text_encoder
        else:
            text_encoder = self._cached_text_encoder

        te_block_swap = self._setup_text_encoder_block_swap_cached(text_encoder)
        if te_block_swap is None:
            text_encoder.to(self.device)
        else:
            self._move_text_encoder_non_layers_to_gpu(text_encoder)

        gemma = getattr(text_encoder, "model", None)
        if gemma is not None:
            _device = self.device
            if not getattr(gemma, "_device_override_applied", False):
                class _DeviceOverride(type(gemma)):  # type: ignore[misc]
                    @property
                    def device(self_inner: Any) -> Any:  # type: ignore[override]
                        return _device
                gemma.__class__ = _DeviceOverride  # type: ignore[assignment]
                gemma._device_override_applied = True  # type: ignore[attr-defined]

        context_p = encode_text(text_encoder, prompts=[prompt])[0]
        video_context, audio_context = self._normalize_text_contexts(*context_p)

        # IMPORTANT: for dev A2V, encode negative prompt while text encoder is
        # already resident to avoid reloading it after transformer placement.
        # Reloading text encoder later can OOM on 24GB cards.
        neg_video_context: torch.Tensor | None = None
        neg_audio_context: torch.Tensor | None = None
        if not self._is_distilled_mode():
            dev_negative_prompt, _, _ = _get_official_dev_guidance_defaults()
            resolved_neg = negative_prompt.strip() or dev_negative_prompt
            neg_context_p = encode_text(text_encoder, prompts=[resolved_neg])[0]
            neg_video_context, neg_audio_context = self._normalize_text_contexts(*neg_context_p)

        if te_block_swap is not None:
            te_block_swap.offload_all()
        else:
            text_encoder.to("cpu")
        self.vram_manager.cleanup()
        logger.info("[low-vram-a2v] Phase 1 done: %.2fs", _time.perf_counter() - t_phase1)

        # ============================================================
        # Phase 1b: Audio encoding
        # ============================================================
        t_audio = _time.perf_counter()
        logger.info("[low-vram-a2v] Phase 1b: Audio encoding")

        decoded_audio = decode_audio_from_file(
            audio_path, self.device, audio_start_time, audio_max_duration,
        )
        assert decoded_audio is not None, "Audio file contains no audio stream"

        audio_encoder = self.model_ledger.audio_encoder()
        self.vram_manager.ensure_on_gpu("audio_encoder", audio_encoder)
        encoded_audio_latent = vae_encode_audio(decoded_audio, audio_encoder).to(
            device=self.device, dtype=self.dtype,
        )
        # Keep the original waveform on CPU only; we only need it again at final mux.
        decoded_audio = Audio(
            waveform=decoded_audio.waveform.detach().cpu(),
            sampling_rate=decoded_audio.sampling_rate,
        )
        audio_shape = AudioLatentShape.from_duration(
            batch=1, duration=num_frames / frame_rate, channels=8, mel_bins=16,
        )
        target_frames = audio_shape.frames
        if encoded_audio_latent.shape[2] < target_frames:
            pad_size = target_frames - encoded_audio_latent.shape[2]
            encoded_audio_latent = torch.nn.functional.pad(
                encoded_audio_latent, (0, 0, 0, pad_size),
            )
        else:
            encoded_audio_latent = encoded_audio_latent[:, :, :target_frames]

        self.vram_manager.offload_to_cpu("audio_encoder", audio_encoder)
        self.vram_manager.cleanup()
        logger.info("[low-vram-a2v] Phase 1b done: %.2fs", _time.perf_counter() - t_audio)

        # ============================================================
        # Phase 2: Image conditioning (if i2v+a2v combo)
        # ============================================================
        target_output_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width, height=height, fps=frame_rate,
        )
        stage_1_output_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=frame_rate,
        )

        conditionings: list[Any] = []
        if images:
            t_phase2 = _time.perf_counter()
            logger.info("[low-vram-a2v] Phase 2: Image conditioning")
            if self._cached_video_encoder is None:
                video_encoder = self.model_ledger.video_encoder()
                self._cached_video_encoder = video_encoder
            else:
                video_encoder = self._cached_video_encoder
            self.vram_manager.ensure_on_gpu("video_encoder", video_encoder)

            ltx_images = [_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images]
            conditionings = image_conditionings_by_replacing_latent(
                images=ltx_images,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=video_encoder,
                dtype=self.dtype,
                device=self.device,
            )
            self.vram_manager.offload_to_cpu("video_encoder", video_encoder)
            self.vram_manager.cleanup()
            logger.info("[low-vram-a2v] Phase 2 done: %.2fs", _time.perf_counter() - t_phase2)

        # ============================================================
        # Phase 3: Stage 1 denoising (half-resolution, frozen audio)
        # ============================================================
        t_phase3 = _time.perf_counter()
        logger.info("[low-vram-a2v] Phase 3: Stage 1 denoising (half-res)")

        if self._cached_transformer is None:
            logger.info("[low-vram-a2v] Loading transformer from disk")
            t_load = _time.perf_counter()
            if self._use_gguf:
                transformer = self._load_gguf_transformer()
            else:
                transformer = self.model_ledger.transformer()
            logger.info("[low-vram-a2v] Transformer loaded: %.2fs", _time.perf_counter() - t_load)
            self._cached_transformer = transformer
        else:
            transformer = self._cached_transformer

        transformer, using_block_swap = self._prepare_a2v_transformer_for_denoise(transformer)
        self._log_vram_usage("before Phase 3 denoise (a2v)")

        # Respect the model type: distilled models use the baked-in
        # distilled sigma schedule with no guidance.  Dev (non-distilled)
        # models need the LTX2Scheduler sigmas and CFG/STG guidance —
        # without this, dev models produce pure noise.
        use_cfg = not self._is_distilled_mode()
        if use_cfg:
            from ltx_core.components.guiders import MultiModalGuiderFactory

            dev_negative_prompt, video_guider_defaults, audio_guider_defaults = (
                _get_official_dev_guidance_defaults()
            )
            # Re-encode negative prompt for CFG (text encoder was offloaded,
            # but contexts are still on CPU from Phase 1).
            stage_1_sigma_values = _make_dev_sigmas(self._num_inference_steps or 20)
            stage_1_sigmas = torch.tensor(stage_1_sigma_values, dtype=torch.float32, device=self.device)
            # Stage 2 always uses the distilled refinement schedule
            stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)

            logger.info(
                "[low-vram-a2v] Dev mode: %d-step Stage 1 + %d-step Stage 2, CFG=%.1f STG=%.1f",
                len(stage_1_sigma_values) - 1,
                len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1,
                video_guider_defaults.cfg_scale,
                video_guider_defaults.stg_scale,
            )
        else:
            stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)
            stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)

        total_steps = (len(stage_1_sigmas) - 1) + (len(stage_2_sigmas) - 1)
        step_counter = [0]

        # Negative prompt context is prepared in Phase 1 for dev mode to avoid
        # reloading text encoder after transformer allocation.

        def denoising_loop(
            sigmas: torch.Tensor,
            video_state: Any,
            audio_state: Any,
            stepper: EulerDiffusionStep,
        ) -> tuple[Any, Any]:
            if use_cfg and neg_video_context is not None:
                video_guider_factory = MultiModalGuiderFactory.constant(
                    video_guider_defaults,
                    negative_context=neg_video_context,
                )
                audio_guider_factory = MultiModalGuiderFactory.constant(
                    audio_guider_defaults,
                    negative_context=neg_audio_context,
                )
                base_denoise = multi_modal_guider_factory_denoising_func(
                    video_guider_factory=video_guider_factory,
                    audio_guider_factory=audio_guider_factory,
                    v_context=video_context,
                    a_context=audio_context,
                    transformer=transformer,
                )
            else:
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

        video_state = denoise_video_only(
            output_shape=stage_1_output_shape,
            conditionings=conditionings,
            noiser=noiser,
            sigmas=stage_1_sigmas,
            stepper=stepper,
            denoising_loop_fn=cast(Any, denoising_loop),
            components=self.pipeline_components,
            dtype=self.dtype,
            device=self.device,
            initial_audio_latent=encoded_audio_latent,
        )
        logger.info("[low-vram-a2v] Stage 1 done: %.2fs", _time.perf_counter() - t_phase3)

        # Free stage-1-only data before the upscale pass.
        del conditionings
        video_context = video_context.to("cpu")
        if audio_context is not None:
            audio_context = audio_context.to("cpu")
        encoded_audio_latent = encoded_audio_latent.to("cpu")
        self.vram_manager.cleanup()

        # ============================================================
        # Phase 3b: Upsample + Stage 2 refinement (full-res)
        # ============================================================
        t_phase3b = _time.perf_counter()
        logger.info("[low-vram-a2v] Phase 3b: Upscale + Stage 2 refinement")

        # Offload transformer for upscale
        if using_block_swap and self._block_swap_wrapper is not None:
            self._block_swap_wrapper.offload_all()
        else:
            self.vram_manager.offload_to_cpu("transformer", transformer)
        self.vram_manager.cleanup()

        # Upscale
        if self._cached_video_encoder is None:
            video_encoder = self.model_ledger.video_encoder()
            self._cached_video_encoder = video_encoder
        else:
            video_encoder = self._cached_video_encoder
        self.vram_manager.ensure_on_gpu("video_encoder", video_encoder)

        if self._cached_spatial_upsampler is None:
            spatial_upsampler = self.model_ledger.spatial_upsampler()
            self._cached_spatial_upsampler = spatial_upsampler
        else:
            spatial_upsampler = self._cached_spatial_upsampler
        self.vram_manager.ensure_on_gpu("spatial_upsampler", spatial_upsampler)

        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=spatial_upsampler,
        )
        del video_state
        self.vram_manager.cleanup()

        # Image conditioning at full resolution
        stage_2_conditionings: list[Any] = []
        if images:
            ltx_images = [_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images]
            stage_2_conditionings = image_conditionings_by_replacing_latent(
                images=ltx_images,
                height=target_output_shape.height,
                width=target_output_shape.width,
                video_encoder=video_encoder,
                dtype=self.dtype,
                device=self.device,
            )

        self.vram_manager.offload_to_cpu("video_encoder", video_encoder)
        self.vram_manager.offload_to_cpu("spatial_upsampler", spatial_upsampler)
        self.vram_manager.cleanup()

        # Bring frozen contexts/latents back only when refinement is ready to start.
        video_context = video_context.to(device=self.device, dtype=self.dtype)
        if audio_context is not None:
            audio_context = audio_context.to(device=self.device, dtype=self.dtype)
        encoded_audio_latent = encoded_audio_latent.to(device=self.device, dtype=self.dtype)
        self.vram_manager.cleanup()

        # Bring transformer back for stage 2
        if using_block_swap and self._block_swap_wrapper is not None:
            transformer = self._setup_block_swap_if_needed(transformer)
            self._move_non_block_parts_to_gpu(transformer)
        else:
            self.vram_manager.ensure_on_gpu("transformer", transformer)
        self.vram_manager.cleanup()

        video_state = denoise_video_only(
            output_shape=target_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=stage_2_sigmas,
            stepper=stepper,
            denoising_loop_fn=cast(Any, denoising_loop),
            components=self.pipeline_components,
            dtype=self.dtype,
            device=self.device,
            noise_scale=stage_2_sigmas[0].item(),
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=encoded_audio_latent,
        )
        logger.info("[low-vram-a2v] Phase 3b done: %.2fs", _time.perf_counter() - t_phase3b)

        # Offload transformer
        if using_block_swap and self._block_swap_wrapper is not None:
            self._block_swap_wrapper.offload_all()
        else:
            self.vram_manager.offload_to_cpu("transformer", transformer)
        self.vram_manager.cleanup()

        # ============================================================
        # Phase 4: VAE video decode + encode output
        # ============================================================
        t_phase4 = _time.perf_counter()
        logger.info("[low-vram-a2v] Phase 4: VAE decode + output")
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

        # Use original audio (not VAE-decoded) for fidelity
        max_samples = round(num_frames / frame_rate * decoded_audio.sampling_rate)
        trimmed_waveform = decoded_audio.waveform.squeeze(0)[..., :max_samples]
        original_audio = Audio(waveform=trimmed_waveform, sampling_rate=decoded_audio.sampling_rate)

        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(
            video=decoded_video,
            audio=original_audio,
            fps=int(frame_rate),
            output_path=output_path,
            video_chunks_number_value=chunks,
        )
        self.vram_manager.offload_to_cpu("video_decoder", video_decoder)
        self.vram_manager.cleanup()
        logger.info("[low-vram-a2v] Phase 4 done: %.2fs", _time.perf_counter() - t_phase4)
        logger.info("[low-vram-a2v] A2V generation complete: %s", output_path)

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
        from services.gguf_loader.gguf_lazy_loader import GGUFLinear

        count = 0
        skipped = 0
        for child in text_encoder.modules():
            if not isinstance(child, _torch.nn.Linear):
                continue
            if isinstance(child, GGUFLinear):
                skipped += 1
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
                    w = lin.weight.to(device=x.device, dtype=x.dtype)
                    b = lin.bias.to(device=x.device, dtype=x.dtype) if lin.bias is not None else None  # pyright: ignore[reportUnnecessaryComparison]
                    return _torch.nn.functional.linear(x, w, b)
                return _fwd

            child.forward = _make_upcast_forward(child)  # type: ignore[assignment]
            count += 1

        if skipped > 0:
            logger.info("Text encoder: %d layers already FP8, quantized %d more", skipped, count)
        else:
            logger.info("Quantized %d Linear layers to FP8 in text encoder", count)

    def _setup_text_encoder_block_swap_cached(
        self, text_encoder: torch.nn.Module,
    ) -> "FastBlockSwapWrapper | None":
        """Apply block swap to Gemma language model layers.

        Reuses the existing wrapper if available to avoid accumulating
        duplicate forward hooks on every generation call.
        """
        if self._te_block_swap_wrapper is not None:
            self._te_block_swap_wrapper.restore_gpu_blocks()
            logger.info("Reusing existing text encoder block swap wrapper")
            return self._te_block_swap_wrapper

        wrapper = self._setup_text_encoder_block_swap(text_encoder)
        if wrapper is not None:
            self._te_block_swap_wrapper = wrapper
        return wrapper

    def _setup_text_encoder_block_swap(
        self, text_encoder: torch.nn.Module,
    ) -> "FastBlockSwapWrapper | None":
        """Apply block swap to Gemma language model layers."""
        from services.block_swap.fast_block_swap import FastBlockSwapWrapper
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

        wrapper = FastBlockSwapWrapper(
            transformer=lang_model,
            device=self.device,
            blocks_to_keep_on_gpu=keep_on_gpu,
            prefetch_distance=2,
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

    def _log_vram_usage(self, label: str) -> None:
        """Log current allocated VRAM after a phase boundary."""
        stats = self.vram_manager.get_vram_debug_stats_mb()
        logger.info(
            "[low-vram] VRAM %s: allocated=%dMB reserved=%dMB driver_used=%dMB",
            label,
            stats["memory_allocated_mb"],
            stats["memory_reserved_mb"],
            stats["driver_used_mb"],
        )

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
