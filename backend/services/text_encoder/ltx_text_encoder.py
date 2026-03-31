"""Text encoder patching and API embedding service."""

from __future__ import annotations

import logging
import pickle
import time
from collections.abc import Callable
from dataclasses import replace
from types import MethodType, SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import torch

from services.http_client.http_client import HTTPClient
from services.services_utils import PromptInput, TensorOrNone, sync_device
from state.app_state_types import CachedTextEncoder, TextEncodingResult

if TYPE_CHECKING:
    from state.app_state_types import AppState

logger = logging.getLogger(__name__)


class LTXTextEncoder:
    """Stateless text encoding operations with idempotent monkey-patching."""

    def __init__(self, device: torch.device, http: HTTPClient, ltx_api_base_url: str) -> None:
        self.device = device
        self.http = http
        self.ltx_api_base_url = ltx_api_base_url
        self._model_ledger_patched = False
        self._encode_text_patched = False

    def install_patches(self, state_getter: Callable[[], AppState]) -> None:
        self._install_model_ledger_patch(state_getter)
        self._install_encode_text_patch(state_getter)

    def _install_model_ledger_patch(self, state_getter: Callable[[], AppState]) -> None:
        if self._model_ledger_patched:
            return

        try:
            from ltx_pipelines.utils import ModelLedger
            from ltx_pipelines.utils import helpers as ltx_utils
            from ltx_core.loader import SDOps
            from ltx_core.model.transformer import X0Model

            original_text_encoder = ModelLedger.text_encoder
            original_cleanup_memory = ltx_utils.cleanup_memory

            def _patch_gemma_forward_hidden_states_only(module: object) -> None:
                """Skip lm_head during prompt encoding to avoid huge transient VRAM spikes.

                Gemma's standard forward computes logits via lm_head even though LTX only
                needs hidden_states for feature extraction. On 24/16/12 GB cards this can
                allocate multi-GB temporary tensors. Patch the model forward to return an
                object exposing only hidden_states.
                """
                model = getattr(module, "model", None)
                if model is None or not hasattr(model, "model"):
                    return
                if getattr(model, "_ltx_hidden_states_only_patched", False):
                    return

                def _forward_hidden_states_only(self_model: object, *args: object, **kwargs: object) -> object:
                    kwargs["return_dict"] = True
                    outputs = getattr(self_model, "model")(*args, **kwargs)
                    hidden_states = getattr(outputs, "hidden_states", None)
                    return SimpleNamespace(hidden_states=hidden_states)

                model.forward = MethodType(_forward_hidden_states_only, model)  # type: ignore[assignment]
                setattr(model, "_ltx_hidden_states_only_patched", True)

            def _quantize_linear_weights_fp8(module: object) -> None:
                """Cast Linear weights to float8_e4m3fn and patch forward to upcast.

                Large output layers like lm_head are intentionally skipped because they
                create huge temporary allocations when upcast on GPU.
                """
                for name, child in module.named_modules():  # type: ignore[union-attr]
                    if not isinstance(child, torch.nn.Linear):
                        continue
                    if name.endswith("lm_head") or name == "lm_head":
                        continue
                    child.weight.data = child.weight.data.to(torch.float8_e4m3fn)
                    if child.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                        child.bias.data = child.bias.data.to(torch.float8_e4m3fn)

                    def _make_upcast_forward(lin: torch.nn.Linear) -> Callable[..., torch.Tensor]:
                        def _fwd(x: torch.Tensor, **kw: object) -> torch.Tensor:
                            w = lin.weight.to(x.dtype)
                            b = lin.bias.to(x.dtype) if lin.bias is not None else None  # pyright: ignore[reportUnnecessaryComparison]
                            return torch.nn.functional.linear(x, w, b)
                        return _fwd

                    child.forward = _make_upcast_forward(child)  # type: ignore[assignment]

            def _build_text_encoder_safe(self_model_ledger: ModelLedger) -> CachedTextEncoder:
                """Build text encoder, handling meta-backed models via to_empty()."""
                builder = getattr(self_model_ledger, "text_encoder_builder", None)
                if builder is None:
                    raise ValueError(
                        "Text encoder not initialized. Please provide a checkpoint path "
                        "and gemma root path to the ModelLedger constructor."
                    )
                model = builder.build(device=torch.device("cpu"), dtype=self_model_ledger.dtype)
                has_meta = any(
                    str(p.device) == "meta"
                    for p in list(model.parameters()) + list(model.buffers())
                )
                if has_meta:
                    logger.info("Text encoder has meta tensors, using to_empty() to materialize on CPU")
                    model = model.to_empty(device=torch.device("cpu"))
                    # Zero-initialize any still-uninitialized params
                    for param in model.parameters():
                        if not param.is_contiguous() or param.data.nelement() == 0:
                            param.data = torch.zeros_like(param, device=torch.device("cpu"))
                    for buf in model.buffers():
                        if not buf.is_contiguous() or buf.data.nelement() == 0:
                            buf.data = torch.zeros_like(buf, device=torch.device("cpu"))
                model.eval()
                return cast(CachedTextEncoder, model)

            def _materialize_meta_module_on_cpu(module: torch.nn.Module, label: str) -> torch.nn.Module:
                has_meta = any(
                    str(p.device) == "meta"
                    for p in list(module.parameters()) + list(module.buffers())
                )
                if not has_meta:
                    return module

                logger.info("%s has meta tensors, using to_empty() to materialize on CPU", label)
                module = module.to_empty(device=torch.device("cpu"))
                for param in module.parameters():
                    if str(param.device) == "meta":
                        param.data = torch.zeros_like(param, device=torch.device("cpu"))
                for buf in module.buffers():
                    if str(buf.device) == "meta":
                        buf.data = torch.zeros_like(buf, device=torch.device("cpu"))
                return module

            def _build_component_safe(
                self_model_ledger: ModelLedger,
                *,
                builder_attr: str,
                missing_message: str,
                label: str,
            ) -> torch.nn.Module:
                builder = getattr(self_model_ledger, builder_attr, None)
                if builder is None:
                    raise ValueError(missing_message)
                model = builder.build(device=torch.device("cpu"), dtype=self_model_ledger.dtype)
                model = _materialize_meta_module_on_cpu(model, label)
                return model.eval()

            def _build_transformer_safe(self_model_ledger: ModelLedger) -> torch.nn.Module:
                if not hasattr(self_model_ledger, "transformer_builder"):
                    raise ValueError(
                        "Transformer not initialized. Please provide a checkpoint path "
                        "to the ModelLedger constructor."
                    )

                if self_model_ledger.quantization is None:
                    builder = self_model_ledger.transformer_builder
                    model = builder.build(device=torch.device("cpu"), dtype=self_model_ledger.dtype)
                else:
                    sd_ops = self_model_ledger.transformer_builder.model_sd_ops
                    if self_model_ledger.quantization.sd_ops is not None:
                        sd_ops = SDOps(
                            name=(
                                f"sd_ops_chain_{sd_ops.name}"
                                f"+{self_model_ledger.quantization.sd_ops.name}"
                            ),
                            mapping=(
                                *sd_ops.mapping,
                                *self_model_ledger.quantization.sd_ops.mapping,
                            ),
                        )
                    builder = replace(
                        self_model_ledger.transformer_builder,
                        module_ops=(
                            *self_model_ledger.transformer_builder.module_ops,
                            *self_model_ledger.quantization.module_ops,
                        ),
                        model_sd_ops=sd_ops,
                    )
                    model = builder.build(device=torch.device("cpu"))

                model = _materialize_meta_module_on_cpu(model, "Transformer")
                return X0Model(model).eval()

            def patched_text_encoder(self_model_ledger: ModelLedger) -> object:
                state = state_getter()
                te_state = state.text_encoder
                if te_state is None:
                    return original_text_encoder(self_model_ledger)

                if te_state.api_embeddings is not None:
                    return DummyTextEncoder()

                if te_state.cached_encoder is not None:
                    return te_state.cached_encoder

                try:
                    te_state.cached_encoder = _build_text_encoder_safe(self_model_ledger)
                except Exception:
                    fallback_builder = getattr(self_model_ledger, "_default_text_encoder_builder", None)
                    if fallback_builder is not None:
                        logger.warning(
                            "Text encoder variant builder failed; "
                            "falling back to the default text encoder builder",
                            exc_info=True,
                        )
                        original_builder = getattr(self_model_ledger, "text_encoder_builder", None)
                        setattr(self_model_ledger, "text_encoder_builder", fallback_builder)
                        try:
                            te_state.cached_encoder = _build_text_encoder_safe(self_model_ledger)
                        finally:
                            if original_builder is not None:
                                setattr(self_model_ledger, "text_encoder_builder", original_builder)
                    else:
                        raise

                _patch_gemma_forward_hidden_states_only(te_state.cached_encoder)
                _quantize_linear_weights_fp8(te_state.cached_encoder)
                return te_state.cached_encoder

            def patched_cleanup_memory() -> None:
                state = state_getter()
                te_state = state.text_encoder
                if te_state is not None and te_state.cached_encoder is not None:
                    try:
                        te_state.cached_encoder.to(torch.device("cpu"))
                    except Exception:
                        logger.warning("Failed to move cached text encoder to CPU", exc_info=True)
                original_cleanup_memory()

            def patched_video_encoder(self_model_ledger: ModelLedger) -> object:
                return _build_component_safe(
                    self_model_ledger,
                    builder_attr="vae_encoder_builder",
                    missing_message=(
                        "Video encoder not initialized. Please provide a checkpoint path "
                        "to the ModelLedger constructor."
                    ),
                    label="Video encoder",
                ).to(self.device)

            def patched_video_decoder(self_model_ledger: ModelLedger) -> object:
                return _build_component_safe(
                    self_model_ledger,
                    builder_attr="vae_decoder_builder",
                    missing_message=(
                        "Video decoder not initialized. Please provide a checkpoint path "
                        "to the ModelLedger constructor."
                    ),
                    label="Video decoder",
                ).to(self.device)

            def patched_audio_encoder(self_model_ledger: ModelLedger) -> object:
                return _build_component_safe(
                    self_model_ledger,
                    builder_attr="audio_encoder_builder",
                    missing_message=(
                        "Audio encoder not initialized. Please provide a checkpoint path "
                        "to the ModelLedger constructor."
                    ),
                    label="Audio encoder",
                ).to(self.device)

            def patched_audio_decoder(self_model_ledger: ModelLedger) -> object:
                return _build_component_safe(
                    self_model_ledger,
                    builder_attr="audio_decoder_builder",
                    missing_message=(
                        "Audio decoder not initialized. Please provide a checkpoint path "
                        "to the ModelLedger constructor."
                    ),
                    label="Audio decoder",
                ).to(self.device)

            def patched_transformer(self_model_ledger: ModelLedger) -> object:
                return _build_transformer_safe(self_model_ledger)

            setattr(ModelLedger, "text_encoder", patched_text_encoder)
            setattr(ModelLedger, "transformer", patched_transformer)
            setattr(ModelLedger, "video_encoder", patched_video_encoder)
            setattr(ModelLedger, "video_decoder", patched_video_decoder)
            setattr(ModelLedger, "audio_encoder", patched_audio_encoder)
            setattr(ModelLedger, "audio_decoder", patched_audio_decoder)

            for module_name in (
                "ltx_pipelines.utils.helpers",
                "ltx_pipelines.distilled",
                "ltx_pipelines.ti2vid_one_stage",
                "ltx_pipelines.ti2vid_two_stages",
                "ltx_pipelines.ic_lora",
                "ltx_pipelines.a2vid_two_stage",
                "ltx_pipelines.retake",
                "ltx_pipelines.retake_pipeline",
            ):
                try:
                    module = __import__(module_name, fromlist=["cleanup_memory"])
                    if hasattr(module, "cleanup_memory"):
                        setattr(module, "cleanup_memory", patched_cleanup_memory)
                except ModuleNotFoundError:
                    logger.debug("Skipping unavailable cleanup_memory patch module %s", module_name)
                except Exception:
                    logger.warning("Failed to patch cleanup_memory for module %s", module_name, exc_info=True)

            self._model_ledger_patched = True
            logger.info("Installed ModelLedger text encoder patch")
        except Exception as exc:
            logger.warning("Failed to patch ModelLedger: %s", exc, exc_info=True)

    def _install_encode_text_patch(self, state_getter: Callable[[], AppState]) -> None:
        if self._encode_text_patched:
            return

        try:
            from ltx_core.text_encoders import gemma as text_enc_module
            from ltx_pipelines import distilled as distilled_module

            original_encode_text = text_enc_module.encode_text

            def patched_encode_text(
                text_encoder: object,
                prompts: PromptInput,
                *args: object,
                **kwargs: object,
            ) -> list[tuple[torch.Tensor, TensorOrNone]]:
                state = state_getter()
                te_state = state.text_encoder
                if te_state is not None and te_state.api_embeddings is not None:
                    video_context = te_state.api_embeddings.video_context
                    audio_context = te_state.api_embeddings.audio_context
                    num_prompts = len(prompts) if not isinstance(prompts, str) else 1
                    out: list[tuple[torch.Tensor, TensorOrNone]] = []
                    for i in range(num_prompts):
                        if i == 0:
                            out.append((video_context, audio_context))
                        else:
                            zero_video = torch.zeros_like(video_context)
                            zero_audio = torch.zeros_like(audio_context) if audio_context is not None else None
                            out.append((zero_video, zero_audio))
                    return out

                prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
                return cast(
                    list[tuple[torch.Tensor, TensorOrNone]],
                    original_encode_text(cast(Any, text_encoder), prompt_list, *args, **kwargs),
                )

            setattr(text_enc_module, "encode_text", patched_encode_text)
            setattr(distilled_module, "encode_text", patched_encode_text)

            for module_name in (
                "ltx_pipelines.ti2vid_one_stage",
                "ltx_pipelines.ti2vid_two_stages",
                "ltx_pipelines.ic_lora",
                "ltx_pipelines.a2vid_two_stage",
                "ltx_pipelines.retake",
                "ltx_pipelines.retake_pipeline",
            ):
                try:
                    module = __import__(module_name, fromlist=["encode_text"])
                    setattr(module, "encode_text", patched_encode_text)
                except ModuleNotFoundError:
                    logger.debug("Skipping unavailable encode_text patch module %s", module_name)
                except Exception:
                    logger.warning("Failed to patch encode_text for module %s", module_name, exc_info=True)

            self._encode_text_patched = True
            logger.info("Installed encode_text API embeddings patch")
        except Exception as exc:
            logger.warning("Failed to patch encode_text: %s", exc, exc_info=True)

    def get_model_id_from_checkpoint(self, checkpoint_path: str) -> str | None:
        try:
            from safetensors import safe_open

            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                if metadata and "encrypted_wandb_properties" in metadata:
                    return metadata["encrypted_wandb_properties"]
        except Exception as exc:
            logger.warning("Could not extract model_id from checkpoint: %s", exc, exc_info=True)
        return None

    def encode_via_api(self, prompt: str, api_key: str, checkpoint_path: str, enhance_prompt: bool) -> TextEncodingResult | None:
        model_id = self.get_model_id_from_checkpoint(checkpoint_path)
        if not model_id:
            return None

        try:
            start = time.time()
            response = self.http.post(
                f"{self.ltx_api_base_url}/v1/prompt-embedding",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json_payload={
                    "prompt": prompt,
                    "model_id": model_id,
                    "enhance_prompt": enhance_prompt,
                },
                timeout=60,
            )

            if response.status_code != 200:
                logger.warning("LTX API error %s: %s", response.status_code, response.text)
                return None

            conditioning = pickle.loads(response.content)  # noqa: S301
            if not conditioning or len(conditioning) == 0:
                logger.warning("LTX API returned unexpected conditioning format")
                return None

            embeddings = conditioning[0][0]
            video_dim = 4096
            if embeddings.shape[-1] > video_dim:
                video_context = embeddings[..., :video_dim].contiguous().to(dtype=torch.bfloat16, device=self.device)
                audio_context = embeddings[..., video_dim:].contiguous().to(dtype=torch.bfloat16, device=self.device)
            else:
                video_context = embeddings.contiguous().to(dtype=torch.bfloat16, device=self.device)
                audio_context = None

            logger.info("Text encoded via API in %.1fs", time.time() - start)
            return TextEncodingResult(video_context=video_context, audio_context=audio_context)

        except Exception as exc:
            logger.warning("LTX API encoding failed: %s", exc, exc_info=True)
            return None


class DummyTextEncoder:
    pass
