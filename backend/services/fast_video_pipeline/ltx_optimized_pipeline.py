"""Optimized low-VRAM pipeline with lazy GGUF, fast block swap, and registry caching.

Key improvements over ltx_low_vram_pipeline.py:
1. Lazy GGUF: keeps weights quantized in memory, dequantizes per-layer during forward
2. Fast block swap: pinned memory + double-buffered async prefetch
3. StateDictRegistry: caches parsed safetensors across builders (text encoder, VAE, etc.)
4. Skips redundant safetensors load when using GGUF (no more double-load)
5. LoRA applied at inference time via hooks (no pre-fusion for GGUF mode)
"""

from __future__ import annotations

import gc
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

if TYPE_CHECKING:
    import torch

    from api_types import ImageConditioningInput
    from services.vram_manager.vram_manager import VRAMManager

logger = logging.getLogger(__name__)


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
# SageAttention (reuse from original)
# ---------------------------------------------------------------------------
_sage_attention_installed = False


def _install_sage_attention() -> bool:
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
            def __call__(self, q: Any, k: Any, v: Any, heads: int, mask: Any | None = None) -> Any:
                import torch as _t
                b, _, dim_head = q.shape
                dim_head //= heads
                q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))
                if mask is not None:
                    if mask.ndim == 2:
                        mask = mask.unsqueeze(0)
                    if mask.ndim == 3:
                        mask = mask.unsqueeze(1)
                    out = _t.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
                else:
                    out = sageattn(q, k, v, is_causal=False)
                return out.transpose(1, 2).reshape(b, -1, heads * dim_head)

        def _sage_default(self: Any, q: Any, k: Any, v: Any, heads: int, mask: Any | None = None) -> Any:
            return SageAttn()(q, k, v, heads, mask)

        attn_mod.AttentionFunction.__call__ = _sage_default  # type: ignore[assignment]
        _sage_attention_installed = True
        logger.info("SageAttention installed as default attention backend")
        return True
    except Exception as exc:
        logger.warning("Failed to install SageAttention: %s", exc)
        return False


class LTXOptimizedPipeline:
    """Optimized low-VRAM pipeline with lazy GGUF and fast block swap.

    Drop-in replacement for LTXLowVRAMPipeline with identical API.
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
    ) -> "LTXOptimizedPipeline":
        return LTXOptimizedPipeline(
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

        if vram_manager is None:
            vram_gb = 24
            if _torch.cuda.is_available():
                vram_gb = int(_torch.cuda.get_device_properties(0).total_memory // (1024**3))
            vram_manager = VRAMManager(device, vram_gb)

        self.vram_manager = vram_manager
        self._use_gguf = gguf_path is not None and Path(gguf_path).exists()
        self._block_swap_wrapper: Any = None
        self._te_block_swap_wrapper: Any = None

        # Cached model references
        self._cached_text_encoder: Any = None
        self._cached_transformer: Any = None
        self._cached_video_decoder: Any = None
        self._cached_audio_decoder: Any = None
        self._cached_vocoder: Any = None
        self._cached_video_encoder: Any = None

        # Prompt embedding cache: prompt -> (video_context, audio_context)
        self._prompt_cache: dict[str, tuple[_torch.Tensor, Any]] = {}
        self._prompt_cache_max_size: int = 64

        # Lazy GGUF state dict (kept quantized in memory)
        self._gguf_state_dict: dict[str, _torch.Tensor] | None = None

        if use_sage_attention:
            _install_sage_attention()

        logger.info(
            "LTXOptimizedPipeline: tier=%s gguf=%s lazy_gguf=%s sage=%s",
            vram_manager.tier.value,
            gguf_path is not None,
            self._use_gguf,
            _sage_attention_installed,
        )

        self._init_model_ledger()

    def _init_model_ledger(self) -> None:
        from ltx_core.loader.registry import StateDictRegistry
        from ltx_pipelines.utils import ModelLedger
        from ltx_pipelines.utils.types import PipelineComponents

        from services.services_utils import device_supports_fp8

        quantization = None
        if device_supports_fp8(self.device) and not self._use_gguf:
            from ltx_core.quantization import QuantizationPolicy
            quantization = QuantizationPolicy.fp8_cast()
            logger.info("FP8 quantization enabled")
        elif self._use_gguf:
            logger.info("Skipping FP8 quantization for GGUF transformer")

        # Use StateDictRegistry for caching across builders
        registry = StateDictRegistry()

        loras = None
        if not self._use_gguf:
            # Only pre-fuse LoRA for safetensors mode
            all_lora_entries = self._collect_loras()
            if all_lora_entries:
                loras = all_lora_entries

        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=self._torch.device("cpu"),
            checkpoint_path=self._checkpoint_path,
            gemma_root_path=self._gemma_root,
            loras=loras,
            quantization=quantization,
            registry=registry,
        )

        # Text encoder variant override — use separate (non-shared) registry
        # to avoid poisoning the shared registry with variant-specific state dicts
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
                from services.text_encoder.safetensors_text_encoder_builder import (
                    SafetensorsGemmaTextEncoderBuilder,
                )
                from services.text_encoder.text_encoder_variant_utils import (
                    variant_uses_wrapped_gemma_text_encoder_keys,
                )
                variant_path = str(Path(self._text_encoder_variant_path))
                module_ops = module_ops_from_gemma_root(self._gemma_root)
                # Use DummyRegistry for the variant builder to avoid meta-device
                # issues when the shared StateDictRegistry caches the main
                # checkpoint's state dict
                if not hasattr(self.model_ledger, "_default_text_encoder_builder"):
                    setattr(
                        self.model_ledger,
                        "_default_text_encoder_builder",
                        self.model_ledger.text_encoder_builder,
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

                variant_module_ops = (
                    (*module_ops,)
                    if variant_uses_wrapped_gemma_text_encoder_keys(variant_path)
                    else (GEMMA_MODEL_OPS, *module_ops)
                )
                if variant_module_ops == (*module_ops,):
                    logger.info(
                        "Text encoder variant already uses wrapped keys; skipping GEMMA_MODEL_OPS prefix: %s",
                        variant_path,
                    )
                base_variant_builder = Builder(
                    model_path=variant_model_path,
                    model_class_configurator=GemmaTextEncoderConfigurator,
                    model_sd_ops=AV_GEMMA_TEXT_ENCODER_KEY_OPS,
                    registry=DummyRegistry(),
                    module_ops=variant_module_ops,
                )
                self.model_ledger.text_encoder_builder = SafetensorsGemmaTextEncoderBuilder(
                    base_builder=base_variant_builder,
                    checkpoint_sources=variant_model_path,
                    module_ops=variant_module_ops,
                    variant_path=variant_path,
                )
                logger.info("Using text encoder variant: %s", variant_path)
            except Exception as exc:
                logger.warning("Failed to configure text encoder variant: %s", exc)

        self.pipeline_components = PipelineComponents(dtype=self.dtype, device=self.device)

        # Pre-load GGUF state dict (lazy — stays quantized)
        if self._use_gguf and self._gguf_path:
            self._preload_gguf()
        else:
            self._gguf_state_dict = None

    def _preload_gguf(self) -> None:
        """Load GGUF with parallel dequantization (threaded, ~2-3x faster)."""
        from services.gguf_loader.gguf_fast_loader import load_gguf_fast

        t0 = time.perf_counter()
        self._gguf_state_dict = load_gguf_fast(
            self._gguf_path,  # type: ignore[arg-type]
            device="cpu",
            num_workers=6,
            remap_keys=True,
        )
        logger.info("GGUF pre-loaded (parallel dequant) in %.2fs", time.perf_counter() - t0)

    def _collect_loras(self) -> list[Any] | None:
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP

        entries: list[Any] = []
        if self._lora_path and Path(self._lora_path).exists():
            entries.append(LoraPathStrengthAndSDOps(
                path=self._lora_path, strength=self._lora_strength,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            ))
        for path, strength in self._extra_loras:
            if Path(path).exists():
                entries.append(LoraPathStrengthAndSDOps(
                    path=path, strength=strength,
                    sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                ))
        return entries if entries else None

    # ------------------------------------------------------------------
    # Prompt embedding cache
    # ------------------------------------------------------------------

    def _cache_prompt_embedding(
        self,
        prompt: str,
        video_context_cpu: Any,
        audio_context_cpu: Any,
    ) -> None:
        """Store prompt embedding on CPU. Evict oldest if over limit."""
        if len(self._prompt_cache) >= self._prompt_cache_max_size:
            oldest = next(iter(self._prompt_cache))
            del self._prompt_cache[oldest]
        self._prompt_cache[prompt] = (video_context_cpu, audio_context_cpu)
        logger.info(
            "Cached prompt embedding (%d/%d): '%s...'",
            len(self._prompt_cache),
            self._prompt_cache_max_size,
            prompt[:40],
        )

    def clear_prompt_cache(self) -> None:
        """Clear all cached prompt embeddings."""
        self._prompt_cache.clear()
        logger.info("Prompt cache cleared")

    def _normalize_text_contexts(
        self,
        video_context: Any,
        audio_context: Any,
    ) -> tuple[Any, Any]:
        """Match text-conditioning tensors to transformer compute dtype/device."""
        video_context = video_context.to(device=self.device, dtype=self.dtype)
        if audio_context is not None:
            audio_context = audio_context.to(device=self.device, dtype=self.dtype)
        return video_context, audio_context

    # ------------------------------------------------------------------
    # GGUF transformer loading (lazy — no dequantization at load time)
    # ------------------------------------------------------------------

    def _load_gguf_transformer(self) -> Any:
        """Build transformer skeleton + load GGUF weights (skip safetensors)."""
        if self._gguf_state_dict is None:
            raise RuntimeError("GGUF state dict not loaded")

        t0 = time.perf_counter()

        # Build only the model skeleton (on meta device — no memory allocation)
        # This avoids loading the 46GB safetensors checkpoint entirely.
        builder = self.model_ledger.transformer_builder
        config = builder.model_config()
        meta_model = builder.meta_model(config, builder.module_ops)

        # Load GGUF weights directly into the skeleton (assign=True replaces
        # meta tensors with actual data without shape mismatch errors)
        meta_model.load_state_dict(self._gguf_state_dict, strict=False, assign=True)

        # Cast all non-GGUF parameters (norms, biases, embeddings) to the
        # pipeline compute dtype to prevent mixed-dtype attention errors.
        from services.gguf_loader.gguf_lazy_loader import GGUFParameter as _GGUFParam
        for param in meta_model.parameters():
            if isinstance(param, _GGUFParam):
                continue  # Skip quantized GGUF weights
            if param.dtype != self.dtype and param.dtype.is_floating_point:
                param.data = param.data.to(self.dtype)
        for buf in meta_model.buffers():
            if buf.dtype.is_floating_point and buf.dtype != self.dtype:
                buf.data = buf.data.to(self.dtype)

        # Free the GGUF state dict from RAM — weights are now in the model
        self._gguf_state_dict = None
        gc.collect()

        # Move to CPU and wrap in X0Model
        from ltx_core.model.transformer import X0Model
        transformer = X0Model(meta_model).eval()

        logger.info("GGUF transformer loaded in %.2fs (skeleton + assign), freed GGUF cache", time.perf_counter() - t0)
        return transformer

    # ------------------------------------------------------------------
    # LoRA hooks for GGUF mode (apply at inference, not pre-fuse)
    # ------------------------------------------------------------------

    def _install_lora_hooks(self, transformer: Any) -> None:
        """Install inference-time LoRA hooks on the transformer.

        Instead of pre-fusing LoRA into weights (which requires dequant + fuse + requant),
        we apply LoRA as: output += (input @ A) @ B * strength during forward.
        """
        import torch as _torch

        lora_entries = self._collect_loras()
        if not lora_entries:
            return

        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        import safetensors

        for lora_entry in lora_entries:
            lora_path = lora_entry.path
            lora_strength = lora_entry.strength

            if lora_strength == 0:
                continue

            t0 = time.perf_counter()

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
                time.perf_counter() - t0,
            )

    # ------------------------------------------------------------------
    # Block swap
    # ------------------------------------------------------------------

    def _setup_block_swap(self, transformer: Any) -> Any:
        """Wrap transformer with block swap.

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
        logger.info("FastBlockSwap: keeping %d blocks on GPU", blocks_on_gpu)

        self._block_swap_wrapper = FastBlockSwapWrapper(
            transformer=transformer,
            device=self.device,
            blocks_to_keep_on_gpu=blocks_on_gpu,
            prefetch_distance=2,
        )

        if self._block_swap_wrapper.block_count == 0:
            logger.warning("Block swap found 0 blocks — falling back")
            self._block_swap_wrapper = None

        return transformer

    # ------------------------------------------------------------------
    # Text encoder helpers (same as original)
    # ------------------------------------------------------------------

    @staticmethod
    def _quantize_text_encoder_fp8(text_encoder: Any) -> None:
        import torch as _torch
        count = 0
        for child in text_encoder.modules():
            if not isinstance(child, _torch.nn.Linear):
                continue
            if child.weight.dtype == _torch.float8_e4m3fn:
                continue
            child.weight.data = child.weight.data.to(_torch.float8_e4m3fn)
            if child.bias is not None:
                child.bias.data = child.bias.data.to(_torch.float8_e4m3fn)

            def _make_upcast_forward(lin: _torch.nn.Linear) -> Any:
                def _fwd(x: _torch.Tensor, **kw: Any) -> _torch.Tensor:
                    w = lin.weight.to(device=x.device, dtype=x.dtype)
                    b = lin.bias.to(device=x.device, dtype=x.dtype) if lin.bias is not None else None
                    return _torch.nn.functional.linear(x, w, b)
                return _fwd

            child.forward = _make_upcast_forward(child)  # type: ignore[assignment]
            count += 1
        logger.info("Quantized %d Linear layers to FP8 in text encoder", count)

    def _setup_text_encoder_block_swap(self, text_encoder: Any) -> Any:
        """Apply block swap to Gemma language model layers.

        Reuses the existing wrapper if available to avoid accumulating
        duplicate forward hooks on every generation call.
        """
        if self._te_block_swap_wrapper is not None:
            self._te_block_swap_wrapper.restore_gpu_blocks()
            logger.info("Reusing existing text encoder block swap wrapper")
            return self._te_block_swap_wrapper

        wrapper = self._create_text_encoder_block_swap(text_encoder)
        if wrapper is not None:
            self._te_block_swap_wrapper = wrapper
        return wrapper

    def _create_text_encoder_block_swap(self, text_encoder: Any) -> Any:
        from services.block_swap.fast_block_swap import FastBlockSwapWrapper
        from services.vram_manager.vram_manager import VRAMTier

        gemma_model = getattr(text_encoder, "model", None)
        if gemma_model is None:
            return None
        lang_model = getattr(gemma_model, "language_model", None)
        if lang_model is None:
            return None

        layers = None
        for target in (lang_model, getattr(lang_model, "model", None)):
            if target is None:
                continue
            layers = getattr(target, "layers", None)
            if layers is not None and hasattr(layers, "__len__"):
                break

        if layers is None or len(layers) < 2:
            return None

        match self.vram_manager.tier:
            case VRAMTier.HIGH:
                keep = 6
            case VRAMTier.MEDIUM:
                keep = 4
            case VRAMTier.LOW:
                keep = 3
            case _:
                keep = 2

        # Tighten if VRAM is low
        free_vram_mb = self.vram_manager.get_free_vram_mb()
        if free_vram_mb < 10_000:
            keep = min(keep, 3)
        if free_vram_mb < 6_000:
            keep = min(keep, 2)

        if keep >= len(layers):
            return None

        wrapper = FastBlockSwapWrapper(
            transformer=lang_model,
            device=self.device,
            blocks_to_keep_on_gpu=keep,
            prefetch_distance=2,
        )
        if wrapper.block_count == 0:
            return None
        logger.info("Text encoder block swap: %d layers, %d on GPU", wrapper.block_count, keep)
        return wrapper

    def _move_text_encoder_non_layers_to_gpu(self, text_encoder: Any) -> None:
        gemma_model = getattr(text_encoder, "model", None)
        if gemma_model is None:
            text_encoder.to(self.device)
            return
        lang_model = getattr(gemma_model, "language_model", None)
        if lang_model is not None:
            for child_name, child in lang_model.named_children():
                if child_name != "layers":
                    child.to(self.device)
            for name, param in lang_model.named_parameters(recurse=False):
                param.data = param.data.to(self.device)
            for name, buf in lang_model.named_buffers(recurse=False):
                buf.data = buf.data.to(self.device)
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

    def _move_non_block_parts_to_gpu(self, transformer: Any) -> None:
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

    # ------------------------------------------------------------------
    # Distilled mode detection
    # ------------------------------------------------------------------

    def _is_distilled_mode(self) -> bool:
        """Return True if the configuration uses distilled denoising (no CFG needed)."""
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
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
        is_distilled = "distilled" in self._checkpoint_path.lower()
        if self._lora_path and "distilled" in self._lora_path.lower():
            is_distilled = True
        if self._gguf_path and "dev" in self._gguf_path.lower():
            if self._lora_path and "distilled" in self._lora_path.lower():
                is_distilled = True
            else:
                is_distilled = False
        if is_distilled:
            # Distilled models MUST use the distilled sigma schedule. Using a
            # linear dev schedule with a distilled model produces noise. The
            # number of steps is fixed by the distilled training and cannot be
            # overridden by the user's step-count setting.
            logger.info("Using %d-step distilled sigma schedule", len(DISTILLED_SIGMA_VALUES) - 1)
            return list(DISTILLED_SIGMA_VALUES)
        steps = self._num_inference_steps or 20
        return _make_dev_sigmas(steps)

    # ------------------------------------------------------------------
    # Tiling config
    # ------------------------------------------------------------------

    def _get_tiling_config(self) -> Any:
        from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
        from services.vram_manager.vram_manager import VRAMTier

        match self.vram_manager.tier:
            case VRAMTier.HIGH:
                return TilingConfig(
                    spatial_config=SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=64),
                    temporal_config=TemporalTilingConfig(tile_size_in_frames=64, tile_overlap_in_frames=24),
                )
            case VRAMTier.MEDIUM:
                return TilingConfig(
                    spatial_config=SpatialTilingConfig(tile_size_in_pixels=384, tile_overlap_in_pixels=64),
                    temporal_config=TemporalTilingConfig(tile_size_in_frames=48, tile_overlap_in_frames=16),
                )
            case VRAMTier.LOW:
                return TilingConfig(
                    spatial_config=SpatialTilingConfig(tile_size_in_pixels=256, tile_overlap_in_pixels=64),
                    temporal_config=TemporalTilingConfig(tile_size_in_frames=32, tile_overlap_in_frames=16),
                )
            case _:
                return TilingConfig(
                    spatial_config=SpatialTilingConfig(tile_size_in_pixels=128, tile_overlap_in_pixels=64),
                    temporal_config=TemporalTilingConfig(tile_size_in_frames=16, tile_overlap_in_frames=8),
                )

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
        advanced_mode: str = "standard",
    ) -> None:
        del advanced_mode
        import torch
        with torch.inference_mode():
            self._generate_impl(
                prompt=prompt, seed=seed, height=height, width=width,
                num_frames=num_frames, frame_rate=frame_rate, images=images,
                output_path=output_path, progress_callback=progress_callback,
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
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput
        from ltx_pipelines.utils.helpers import (
            denoise_audio_video,
            image_conditionings_by_replacing_latent,
            multi_modal_guider_factory_denoising_func,
            simple_denoising_func,
        )
        from ltx_core.components.guiders import MultiModalGuiderFactory, MultiModalGuiderParams
        from ltx_pipelines.utils.samplers import euler_denoising_loop

        from services.ltx_pipeline_common import encode_video_output, video_chunks_number

        logger.info("[optimized] %dx%d, %d frames, tier=%s", width, height, num_frames, self.vram_manager.tier.value)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        # Phase 1: Text encoding (with prompt-level caching)
        t1 = time.perf_counter()
        cache_key = prompt.strip()
        cached = self._prompt_cache.get(cache_key)
        neg_video_context = None
        neg_audio_context = None
        dev_negative_prompt, video_guider_defaults, audio_guider_defaults = _get_official_dev_guidance_defaults()
        if cached is not None:
            video_context, audio_context = cached
            video_context, audio_context = self._normalize_text_contexts(video_context, audio_context)
            # Also check for cached negative prompt
            resolved_negative_prompt = negative_prompt.strip() or dev_negative_prompt
            neg_cached = self._prompt_cache.get(f"__neg__::{resolved_negative_prompt}")
            if neg_cached is not None:
                neg_video_context, neg_audio_context = neg_cached
                neg_video_context, neg_audio_context = self._normalize_text_contexts(neg_video_context, neg_audio_context)
            logger.info("[optimized] Phase 1 (text encode): CACHED %.4fs", time.perf_counter() - t1)
        else:
            if self._cached_text_encoder is None:
                text_encoder = self.model_ledger.text_encoder()
                self._quantize_text_encoder_fp8(text_encoder)
                self._cached_text_encoder = text_encoder
            else:
                text_encoder = self._cached_text_encoder

            te_swap = self._setup_text_encoder_block_swap(text_encoder)
            if te_swap is None:
                # Without block swapping, a cached encoder may have been fully
                # offloaded after the previous run. Move it back as one module.
                text_encoder.to(self.device)
            else:
                self._move_text_encoder_non_layers_to_gpu(text_encoder)

            gemma = getattr(text_encoder, "model", None)
            if gemma is not None:
                from services.text_encoder.ltx_text_encoder import _set_text_encoder_runtime_device

                _set_text_encoder_runtime_device(text_encoder, self.device)

            context_p = encode_text(text_encoder, prompts=[prompt])[0]
            video_context, audio_context = self._normalize_text_contexts(*context_p)

            # Dev (non-distilled) models need CFG — encode negative prompt
            use_cfg = not self._is_distilled_mode()
            neg_video_context = None
            neg_audio_context = None
            if use_cfg:
                resolved_negative_prompt = negative_prompt.strip() or dev_negative_prompt
                neg_cache_key = f"__neg__::{resolved_negative_prompt}"
                neg_context_p = encode_text(text_encoder, prompts=[resolved_negative_prompt])[0]
                neg_video_context, neg_audio_context = self._normalize_text_contexts(*neg_context_p)
                # Cache negative embeddings
                self._cache_prompt_embedding(
                    neg_cache_key,
                    neg_video_context.detach().cpu(),
                    neg_audio_context.detach().cpu() if neg_audio_context is not None else None,
                )
                logger.info(
                    "[optimized] Dev mode: encoded official negative prompt for CFG/STG "
                    "(cfg=%.2f stg=%.2f stg_blocks=%s)",
                    video_guider_defaults.cfg_scale,
                    video_guider_defaults.stg_scale,
                    list(video_guider_defaults.stg_blocks),
                )

            # Cache the result on CPU for reuse
            self._cache_prompt_embedding(
                cache_key,
                video_context.detach().cpu(),
                audio_context.detach().cpu() if audio_context is not None else None,
            )

            if te_swap is not None:
                te_swap.offload_all()
            text_encoder.to("cpu")
            self.vram_manager.cleanup()
            logger.info("[optimized] Phase 1 (text encode): %.2fs", time.perf_counter() - t1)

        # Phase 2: Image conditioning
        output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        conditionings: list[Any] = []
        if images:
            t2 = time.perf_counter()
            if self._cached_video_encoder is None:
                self._cached_video_encoder = self.model_ledger.video_encoder()
            video_encoder = self._cached_video_encoder
            self.vram_manager.ensure_on_gpu("video_encoder", video_encoder)
            ltx_images = [_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images]
            conditionings = image_conditionings_by_replacing_latent(
                images=ltx_images, height=output_shape.height, width=output_shape.width,
                video_encoder=video_encoder, dtype=self.dtype, device=self.device,
            )
            self.vram_manager.offload_to_cpu("video_encoder", video_encoder)
            self.vram_manager.cleanup()
            logger.info("[optimized] Phase 2 (image cond): %.2fs", time.perf_counter() - t2)

        # Phase 3: Denoising
        t3 = time.perf_counter()
        if self._cached_transformer is None:
            t_load = time.perf_counter()
            if self._use_gguf:
                transformer = self._load_gguf_transformer()
                # Install LoRA hooks for GGUF mode
                self._install_lora_hooks(transformer)
            else:
                transformer = self.model_ledger.transformer()
            logger.info("[optimized] Transformer loaded: %.2fs", time.perf_counter() - t_load)
            self._cached_transformer = transformer
        else:
            transformer = self._cached_transformer

        transformer = self._setup_block_swap(transformer)

        if self._block_swap_wrapper is not None:
            self._move_non_block_parts_to_gpu(transformer)
        else:
            self.vram_manager.ensure_on_gpu("transformer", transformer)

        sigma_values = self._get_sigma_schedule()
        total_steps = len(sigma_values) - 1
        sigmas = torch.tensor(sigma_values, dtype=torch.float32, device=self.device)
        step_counter = [0]

        def denoising_loop(
            sigmas: torch.Tensor, video_state: Any, audio_state: Any, stepper: EulerDiffusionStep,
        ) -> tuple[Any, Any]:
            use_cfg = not self._is_distilled_mode()
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
                    video_context=video_context, audio_context=audio_context, transformer=transformer,
                )
            def tracked_denoise(*args: Any, **kwargs: Any) -> Any:
                result = base_denoise(*args, **kwargs)
                step_counter[0] += 1
                if progress_callback is not None:
                    progress_callback(step_counter[0], total_steps)
                return result
            return euler_denoising_loop(
                sigmas=sigmas, video_state=video_state, audio_state=audio_state,
                stepper=stepper, denoise_fn=tracked_denoise,
            )

        video_state, audio_state = denoise_audio_video(
            output_shape=output_shape, conditionings=conditionings, noiser=noiser,
            sigmas=sigmas, stepper=stepper, denoising_loop_fn=cast(Any, denoising_loop),
            components=self.pipeline_components, dtype=self.dtype, device=self.device,
        )
        logger.info("[optimized] Phase 3 (denoise): %.2fs", time.perf_counter() - t3)

        if self._block_swap_wrapper is not None:
            self._block_swap_wrapper.offload_all()
        else:
            self.vram_manager.offload_to_cpu("transformer", transformer)
        # Don't clear self._block_swap_wrapper — reuse it next generation
        self.vram_manager.cleanup()

        # Phase 4-6: VAE decode + encode
        t4 = time.perf_counter()
        tiling_config = self._get_tiling_config()

        if self._cached_video_decoder is None:
            self._cached_video_decoder = self.model_ledger.video_decoder()
        video_decoder = self._cached_video_decoder
        self.vram_manager.ensure_on_gpu("video_decoder", video_decoder)
        decoded_video = vae_decode_video(video_state.latent, video_decoder, tiling_config)

        # Offload video decoder BEFORE loading audio models to free VRAM
        self.vram_manager.offload_to_cpu("video_decoder", video_decoder)
        self.vram_manager.cleanup()

        if self._cached_audio_decoder is None:
            self._cached_audio_decoder = self.model_ledger.audio_decoder()
        if self._cached_vocoder is None:
            self._cached_vocoder = self.model_ledger.vocoder()
        audio_decoder = self._cached_audio_decoder
        vocoder = self._cached_vocoder
        self.vram_manager.ensure_on_gpu("audio_decoder", audio_decoder)
        self.vram_manager.ensure_on_gpu("vocoder", vocoder)
        decoded_audio = vae_decode_audio(audio_state.latent, audio_decoder, vocoder)

        self.vram_manager.offload_to_cpu("audio_decoder", audio_decoder)
        self.vram_manager.offload_to_cpu("vocoder", vocoder)
        self.vram_manager.cleanup()

        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(
            video=decoded_video, audio=decoded_audio, fps=int(frame_rate),
            output_path=output_path, video_chunks_number_value=chunks,
        )
        logger.info("[optimized] Phase 4-6 (decode+encode): %.2fs", time.perf_counter() - t4)
        logger.info("[optimized] Generation complete: %s", output_path)

    # ------------------------------------------------------------------
    # Warmup & compile stubs
    # ------------------------------------------------------------------

    def warmup(self, output_path: str) -> None:
        import torch
        try:
            with torch.inference_mode():
                self._generate_impl(
                    prompt="test warmup", seed=42, height=256, width=384,
                    num_frames=9, frame_rate=8, images=[], output_path=output_path,
                )
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def compile_transformer(self) -> None:
        logger.info("Skipping torch.compile for optimized pipeline (block swap mode)")
