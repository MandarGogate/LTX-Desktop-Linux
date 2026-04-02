"""Lazy GGUF loader — keeps weights quantized in memory, dequantizes on-the-fly.

This is the ComfyUI-style approach: instead of dequantizing all GGUF tensors to
bf16 at load time (which takes minutes and uses 3-4x memory), we:

1. Read raw quantized byte blocks from the GGUF file
2. Store them as GGUFParameter tensors (raw bytes + quant_type metadata)
3. Replace nn.Linear modules with GGUFLinear that dequantizes per-layer during forward()

Load time goes from ~60-120s → ~2-5s for a 22GB Q8_0 model.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GGUFParameter: keeps raw quantized bytes + quant metadata
# ---------------------------------------------------------------------------

try:
    from gguf import GGML_QUANT_SIZES, GGMLQuantizationType  # type: ignore[reportMissingImports]
except ImportError:
    GGML_QUANT_SIZES = {}
    GGMLQuantizationType = None  # type: ignore[assignment,misc]

# Unquantized types that don't need special handling
_UNQUANTIZED_TYPES: set[int] = set()
if GGMLQuantizationType is not None:
    _UNQUANTIZED_TYPES = {
        int(GGMLQuantizationType.F32),
        int(GGMLQuantizationType.F16),
        int(GGMLQuantizationType.BF16),
    }


def _quant_shape_from_byte_shape(
    shape: tuple[int, ...], type_size: int, block_size: int
) -> tuple[int, ...]:
    return (*shape[:-1], shape[-1] // type_size * block_size)


def _load_alternate_gguf_reader() -> Any | None:
    """Try importing a GGUFReader from common external Python installs.

    Some locally-installed GGUF packages can parse files that the environment's
    bundled package cannot. Prefer the current environment first and fall back
    only when construction fails.
    """
    home = Path.home()
    patterns = (
        home / "miniconda3" / "lib",
        home / "SageAttention" / "miniconda3" / "lib",
    )
    for base in patterns:
        if not base.exists():
            continue
        for init_py in sorted(base.glob("python*/site-packages/gguf/__init__.py")):
            module_name = f"_ltx_external_gguf_{hash(init_py)}"
            spec = importlib.util.spec_from_file_location(
                module_name,
                init_py,
                submodule_search_locations=[str(init_py.parent)],
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                continue
            reader = getattr(module, "GGUFReader", None)
            if reader is not None:
                logger.info("Using alternate GGUFReader from %s", init_py.parent)
                return reader
    return None


def _open_gguf_reader(gguf_path: Path | str) -> Any:
    try:
        from gguf import GGUFReader  # type: ignore[reportMissingImports]
    except ImportError:
        GGUFReader = None  # type: ignore[assignment]

    errors: list[Exception] = []
    if GGUFReader is not None:
        try:
            return GGUFReader(str(gguf_path))
        except Exception as exc:
            errors.append(exc)

    alt_reader = _load_alternate_gguf_reader()
    if alt_reader is not None:
        try:
            return alt_reader(str(gguf_path))
        except Exception as exc:
            errors.append(exc)

    if errors:
        raise errors[-1]
    raise ImportError(
        "The 'gguf' package is required. Install with: pip install gguf"
    )


class GGUFParameter(torch.nn.Parameter):
    """A parameter that stores raw GGUF quantized bytes with metadata."""

    def __new__(
        cls,
        data: torch.Tensor,
        requires_grad: bool = False,
        quant_type: int | None = None,
        tensor_shape: tuple[int, ...] | None = None,
    ) -> "GGUFParameter":
        data = data if data is not None else torch.empty(0)
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.quant_type = quant_type  # type: ignore[attr-defined]
        self.tensor_shape = tensor_shape or tuple(self.shape)  # type: ignore[attr-defined]
        return self

    def as_tensor(self) -> torch.Tensor:
        return torch.Tensor._make_subclass(torch.Tensor, self, self.requires_grad)

    def to(self, *args: Any, **kwargs: Any) -> "GGUFParameter":  # type: ignore[override]
        moved = super().to(*args, **kwargs)
        return GGUFParameter(
            moved,
            requires_grad=self.requires_grad,
            quant_type=getattr(self, "quant_type", None),
            tensor_shape=getattr(self, "tensor_shape", tuple(moved.shape)),
        )


# ---------------------------------------------------------------------------
# Dequantization — torch-native (no numpy), matching diffusers/ComfyUI
# ---------------------------------------------------------------------------

QK_K = 256
K_SCALE_SIZE = 12


def _to_uint32(x: torch.Tensor) -> torch.Tensor:
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def _split_block_dims(
    blocks: torch.Tensor, *args: int
) -> tuple[torch.Tensor, ...]:
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def _get_scale_min(
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8).reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mn = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc.reshape((n_blocks, 8)), mn.reshape((n_blocks, 8))


def _dequant_Q8_0(
    blocks: torch.Tensor,
    block_size: int,
    type_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    d, x = _split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return d * x


def _dequant_Q5_1(
    blocks: torch.Tensor,
    block_size: int,
    type_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, m, qh, qs = _split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = _to_uint32(qh)
    qh = qh.reshape((n_blocks, 1)) >> torch.arange(
        32, device=d.device, dtype=torch.int32
    ).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))
    qs_out = ql | (qh << 4)
    return (d * qs_out) + m


def _dequant_Q5_0(
    blocks: torch.Tensor,
    block_size: int,
    type_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, qh, qs = _split_block_dims(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype)
    qh = _to_uint32(qh)
    qh = qh.reshape(n_blocks, 1) >> torch.arange(
        32, device=d.device, dtype=torch.int32
    ).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)
    qs_out = (ql | (qh << 4)).to(torch.int8) - 16
    return d * qs_out


def _dequant_Q4_1(
    blocks: torch.Tensor,
    block_size: int,
    type_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, m, qs = _split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)
    return (d * qs) + m


def _dequant_Q4_0(
    blocks: torch.Tensor,
    block_size: int,
    type_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, qs = _split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return d * qs


def _dequant_Q6_K(
    blocks: torch.Tensor,
    block_size: int,
    type_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    ql, qh, scales, d = _split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)
    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))
    return (d * q).reshape((n_blocks, QK_K))


def _dequant_Q5_K(
    blocks: torch.Tensor,
    block_size: int,
    type_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, dmin, scales, qh, qs = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qh_bits = qh.reshape((n_blocks, -1, 1, 32)) >> torch.arange(
        0, 8, device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh_bits = (qh_bits & 0x01).reshape((n_blocks, -1, 32))
    q = ql | (qh_bits << 4)
    return (d * q - dm).reshape((n_blocks, QK_K))


def _dequant_Q4_K(
    blocks: torch.Tensor,
    block_size: int,
    type_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))
    return (d * qs - dm).reshape((n_blocks, QK_K))


def _dequant_Q3_K(
    blocks: torch.Tensor,
    block_size: int,
    type_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    hmask, qs, scales, d = _split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8
    ).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    sc = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    sc = sc.to(torch.int8) - 32
    dl = (d * sc).reshape((n_blocks, 16, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 4, 1))
    qh_bits = hmask.reshape(n_blocks, -1, 1, 32) >> torch.arange(
        0, 8, device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh_bits = (qh_bits.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = ql.to(torch.int8) - (qh_bits << 2).to(torch.int8)
    return (dl * q).reshape((n_blocks, QK_K))


def _dequant_Q2_K(
    blocks: torch.Tensor,
    block_size: int,
    type_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    scales, qs, d, dmin = _split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))
    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 4, 1)
    )
    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml
    return qs.reshape((n_blocks, -1))


def _dequant_BF16(
    blocks: torch.Tensor,
    block_size: int,
    type_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


_DEQUANT_FNS: dict[int, Any] = {}
if GGMLQuantizationType is not None:
    _DEQUANT_FNS = {
        int(GGMLQuantizationType.Q8_0): _dequant_Q8_0,
        int(GGMLQuantizationType.Q5_1): _dequant_Q5_1,
        int(GGMLQuantizationType.Q5_0): _dequant_Q5_0,
        int(GGMLQuantizationType.Q4_1): _dequant_Q4_1,
        int(GGMLQuantizationType.Q4_0): _dequant_Q4_0,
        int(GGMLQuantizationType.Q6_K): _dequant_Q6_K,
        int(GGMLQuantizationType.Q5_K): _dequant_Q5_K,
        int(GGMLQuantizationType.Q4_K): _dequant_Q4_K,
        int(GGMLQuantizationType.Q3_K): _dequant_Q3_K,
        int(GGMLQuantizationType.Q2_K): _dequant_Q2_K,
        int(GGMLQuantizationType.BF16): _dequant_BF16,
    }


def dequantize_gguf_tensor(param: GGUFParameter, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Dequantize a GGUFParameter to a regular tensor on the same device."""
    quant_type: int = param.quant_type  # type: ignore[attr-defined]

    if quant_type in _UNQUANTIZED_TYPES:
        # F32 / F16 / BF16 — just reinterpret
        if quant_type == int(GGMLQuantizationType.F32):
            return param.as_tensor().view(torch.float32).to(dtype)
        elif quant_type == int(GGMLQuantizationType.F16):
            return param.as_tensor().view(torch.float16).to(dtype)
        else:  # BF16
            return (
                param.as_tensor()
                .view(torch.int16)
                .to(torch.int32)
                .__lshift__(16)
                .view(torch.float32)
                .to(dtype)
            )

    dequant_fn = _DEQUANT_FNS.get(quant_type)
    if dequant_fn is None:
        raise NotImplementedError(
            f"Unsupported GGUF quant type: {quant_type}"
        )

    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    shape = tuple(getattr(param, "tensor_shape", tuple(param.shape)))

    raw = param.as_tensor().view(torch.uint8)
    n_blocks = raw.numel() // type_size
    blocks = raw.reshape((n_blocks, type_size))

    dequant = dequant_fn(blocks, block_size, type_size, dtype=dtype)
    return dequant.reshape(shape)


# ---------------------------------------------------------------------------
# GGUFLinear — drop-in nn.Linear replacement with lazy dequant
# ---------------------------------------------------------------------------


class GGUFLinear(nn.Linear):
    """Linear layer backed by quantized GGUF weights. Dequantizes on forward()."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        # Use meta device to avoid allocating weights twice
        super().__init__(in_features, out_features, bias, device="meta")
        self.compute_dtype = compute_dtype

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        weight = dequantize_gguf_tensor(self.weight, dtype=self.compute_dtype)  # type: ignore[arg-type]
        if weight.device != inputs.device:
            weight = weight.to(inputs.device)
        weight = weight.to(self.compute_dtype)
        bias = self.bias.to(self.compute_dtype) if self.bias is not None else None
        return torch.nn.functional.linear(inputs, weight, bias)


class GGUFEmbedding(nn.Embedding):
    """Embedding layer backed by quantized GGUF weights."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        padding_idx: int | None = None,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, device="meta")
        self.compute_dtype = compute_dtype

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        weight = dequantize_gguf_tensor(self.weight, dtype=self.compute_dtype)  # type: ignore[arg-type]
        if weight.device != inputs.device:
            weight = weight.to(inputs.device)
        return torch.nn.functional.embedding(
            inputs,
            weight.to(self.compute_dtype),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


# ---------------------------------------------------------------------------
# GGUF State Dict loader — reads raw bytes, no dequant
# ---------------------------------------------------------------------------


def load_gguf_lazy_state_dict(
    gguf_path: Path | str,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Load GGUF file into a state dict of GGUFParameter tensors (still quantized).

    This is ~10-50x faster than the old approach because we skip dequantization.
    The raw quantized bytes are stored as uint8 tensors with quant_type metadata.

    For mmap'd numpy arrays from GGUFReader, we use np.array() instead of .copy()
    which can be faster, and for quantized tensors we create a contiguous copy
    only once (as uint8) without any dtype conversion overhead.
    """
    import numpy as np
    import warnings

    t0 = time.perf_counter()
    reader = _open_gguf_reader(gguf_path)

    state_dict: dict[str, torch.Tensor] = {}
    for tensor in reader.tensors:
        name = tensor.name
        qtype = int(tensor.tensor_type)
        target_shape = tuple(reversed(tensor.shape))

        if qtype in _UNQUANTIZED_TYPES:
            # For F32/F16/BF16, convert to bf16 torch tensor.
            # tensor.data is a mmap'd numpy array; use np.array for a contiguous copy.
            if qtype == int(GGMLQuantizationType.F32):
                np_data = np.array(tensor.data, dtype=np.float32).reshape(target_shape)
                t = torch.from_numpy(np_data).to(dtype=torch.bfloat16, device=device)
            elif qtype == int(GGMLQuantizationType.F16):
                np_data = np.array(tensor.data.view(np.float16).reshape(target_shape))
                t = torch.from_numpy(np_data).to(dtype=torch.bfloat16, device=device)
            else:
                # BF16
                raw16 = np.array(tensor.data.view(np.uint16).reshape(target_shape))
                t = (
                    torch.from_numpy(raw16.astype(np.int32))
                    .to(torch.int32)
                    .__lshift__(16)
                    .view(torch.float32)
                    .to(dtype=torch.bfloat16, device=device)
                )
            state_dict[name] = t
        else:
            # Quantized — store raw bytes as GGUFParameter (NO dequantization).
            # The GGUFReader provides mmap'd uint8 data; we need a contiguous
            # torch tensor. torch.from_numpy on mmap'd array creates a view,
            # so we clone() to own the memory and allow the mmap to be freed.
            block_size, type_size = GGML_QUANT_SIZES[qtype]
            n_elements_per_row = target_shape[-1] if len(target_shape) > 0 else 1
            packed_per_row = (n_elements_per_row // block_size) * type_size
            if len(target_shape) >= 2:
                raw_shape = (target_shape[0], packed_per_row)
            else:
                n_elements = 1
                for s in target_shape:
                    n_elements *= s
                packed = (n_elements // block_size) * type_size
                raw_shape = (packed,)

            # Zero-copy from mmap, then clone to own memory
            raw_np = tensor.data.view(np.uint8)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The given NumPy array is not writable",
                )
                raw_bytes = torch.from_numpy(raw_np).clone()
            if device != "cpu" and device != torch.device("cpu"):
                raw_bytes = raw_bytes.to(device=device)
            try:
                raw_bytes = raw_bytes.reshape(raw_shape)
            except RuntimeError:
                pass

            param = GGUFParameter(
                raw_bytes,
                requires_grad=False,
                quant_type=qtype,
                tensor_shape=target_shape,
            )
            state_dict[name] = param

    elapsed = time.perf_counter() - t0
    logger.info(
        "Lazy-loaded %d tensors from GGUF in %.2fs (no dequantization)",
        len(state_dict),
        elapsed,
    )
    return state_dict


def replace_linear_with_gguf(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    compute_dtype: torch.dtype = torch.bfloat16,
    prefix: str = "",
) -> nn.Module:
    """Replace nn.Linear modules with GGUFLinear where state_dict has GGUFParameter weights."""
    for name, module in model.named_children():
        module_prefix = f"{prefix}{name}."
        replace_linear_with_gguf(module, state_dict, compute_dtype, module_prefix)

        if isinstance(module, nn.Linear):
            weight_key = f"{module_prefix}weight"
            if weight_key in state_dict and isinstance(
                state_dict[weight_key], GGUFParameter
            ):
                new_module = GGUFLinear(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    compute_dtype=compute_dtype,
                )
                new_module.requires_grad_(False)
                model._modules[name] = new_module

    return model


def replace_embedding_with_gguf(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    compute_dtype: torch.dtype = torch.bfloat16,
    prefix: str = "",
) -> nn.Module:
    """Replace nn.Embedding modules with GGUFEmbedding where possible."""
    for name, module in model.named_children():
        module_prefix = f"{prefix}{name}."
        replace_embedding_with_gguf(module, state_dict, compute_dtype, module_prefix)

        if isinstance(module, nn.Embedding):
            weight_key = f"{module_prefix}weight"
            if weight_key in state_dict and isinstance(state_dict[weight_key], GGUFParameter):
                new_module = GGUFEmbedding(
                    module.num_embeddings,
                    module.embedding_dim,
                    padding_idx=module.padding_idx,
                    compute_dtype=compute_dtype,
                )
                new_module.requires_grad_(False)
                model._modules[name] = new_module

    return model


def assign_gguf_linear_weights(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str = "",
) -> tuple[nn.Module, dict[str, torch.Tensor]]:
    """Assign GGUF-backed linear params directly and return remaining state dict.

    ``load_state_dict`` still performs shape checks against the underlying raw-byte
    storage shape of a quantized tensor. Assigning the quantized params directly to
    GGUFLinear modules avoids those checks while still allowing normal tensors to be
    loaded through the standard path.
    """

    remaining = dict(state_dict)
    for name, module in model.named_children():
        module_prefix = f"{prefix}{name}."
        updated_module, remaining = assign_gguf_linear_weights(module, remaining, module_prefix)
        model._modules[name] = updated_module

        if not isinstance(updated_module, (GGUFLinear, GGUFEmbedding)):
            continue

        weight_key = f"{module_prefix}weight"
        weight = remaining.get(weight_key)
        if isinstance(weight, GGUFParameter):
            updated_module._parameters["weight"] = weight
            remaining.pop(weight_key, None)

        if isinstance(updated_module, GGUFEmbedding):
            continue

        bias_key = f"{module_prefix}bias"
        bias = remaining.get(bias_key)
        if isinstance(bias, torch.Tensor):
            updated_module.bias = torch.nn.Parameter(bias, requires_grad=False)
            remaining.pop(bias_key, None)

    return model, remaining


def remap_gguf_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap community GGUF key prefixes to match LTX model keys."""
    remapped: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        key = name
        if key.startswith("model.diffusion_model."):
            key = key.replace("model.diffusion_model.", "")
        elif key.startswith("transformer."):
            key = key.replace("transformer.", "")
        remapped[key] = tensor
    return remapped
