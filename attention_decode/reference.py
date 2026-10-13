"""FP32 reference and equivalent PyTorch/FlashAttention-2 baselines."""

from contextlib import contextmanager
from functools import lru_cache
import math

import torch
from torch.nn import functional as F


ATOL = 2e-2
RTOL = 2e-2


def validate_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    """Validate the public shape/dtype/layout contract without synchronizing."""
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("Q, K, and V must be four-dimensional")
    if q.shape[2] != 1:
        raise ValueError("query sequence length must be 1")
    if k.shape != v.shape:
        raise ValueError("K and V shapes must match")
    if q.shape[0] != k.shape[0] or q.shape[3] != k.shape[3]:
        raise ValueError("Q and K/V batch size and head dimension must match")
    if min(q.shape) < 1 or min(k.shape) < 1:
        raise ValueError("all dimensions must be positive")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("query head count must be divisible by KV head count")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise TypeError("Q, K, and V must have dtype torch.bfloat16")
    if any(x.device != q.device for x in (k, v)):
        raise ValueError("Q, K, and V must be on the same device")
    if not all(x.is_contiguous() for x in (q, k, v)):
        raise ValueError("Q, K, and V must be contiguous")


@contextmanager
def _fp32_matmul():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


@torch.no_grad()
def reference_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """FP32 QK, stable softmax, and PV; group heads without duplicating KV."""
    validate_inputs(q, k, v)
    batch, query_heads, _, dim = q.shape
    kv_heads = k.shape[1]
    group = query_heads // kv_heads
    with _fp32_matmul():
        grouped_q = q.float().reshape(batch, kv_heads, group, dim)
        logits = torch.matmul(grouped_q, k.float().transpose(-1, -2))
        logits.mul_(1.0 / math.sqrt(dim))
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        output = torch.matmul(probabilities, v.float())
    return output.reshape(batch, query_heads, 1, dim)


@torch.no_grad()
def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The contract's comparison target: FP32 calculation rounded to BF16."""
    return reference_fp32(q, k, v).to(torch.bfloat16)


# Explicit public alias used by the correctness suite.
reference_attention = reference


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The native BHSD-layout SDPA baseline, including GQA when required."""
    return F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=False, enable_gqa=q.shape[1] != k.shape[1]
    )


@lru_cache(maxsize=1)
def get_flash_attn_func():
    """Load the installed FA2 lazily so references work without the extension."""
    from flash_attn import flash_attn_func

    return flash_attn_func


def fa2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """FA2 with the complete BHSD -> BSHD -> BHSD layout adapter timed."""
    output = get_flash_attn_func()(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
    )
    return output.transpose(1, 2).contiguous()


@torch.no_grad()
def error_metrics(output: torch.Tensor, expected: torch.Tensor) -> dict:
    """Record peak errors; relative error uses a documented 1e-6 denominator floor."""
    shape_ok = output.shape == expected.shape
    dtype_ok = output.dtype == torch.bfloat16
    device_ok = output.device == expected.device
    if not (shape_ok and dtype_ok and device_ok):
        return {
            "correct": False,
            "max_abs_error": None,
            "max_rel_error": None,
            "error": f"output contract violation: shape={output.shape}, dtype={output.dtype}, device={output.device}",
        }
    finite = bool(torch.isfinite(output).all())
    difference = (output.float() - expected.float()).abs()
    return {
        "correct": finite and bool(torch.allclose(output, expected, atol=ATOL, rtol=RTOL)),
        "max_abs_error": float(difference.max()),
        "max_rel_error": float((difference / expected.float().abs().clamp_min(1e-6)).max()),
        "error": "" if finite else "output contains non-finite values",
    }
