"""Contract tests for BF16 single-query attention on SM120.

Run ``python -m pytest -q tests/test_attention_decode.py``.  Add
``--junitxml=runs/correctness.xml`` to persist per-case maximum errors.
GPU absence is an explicit skip; out-of-memory and runtime failures are failures.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from attention_decode import attention_decode
from attention_decode.reference import reference_attention


ATOL = 2e-2
RTOL = 2e-2
SEED = 20261013
WORKLOADS = [
    (batch, hq, hkv, length, dim)
    for batch, (hq, hkv), dim, length in itertools.product(
        (1, 4), ((32, 8), (32, 32)), (64, 128), (1, 127, 128, 129, 512, 2048, 8192)
    )
]


def _case_id(shape: tuple[int, int, int, int, int]) -> str:
    batch, hq, hkv, length, dim = shape
    return f"B{batch}-H{hq}x{hkv}-T{length}-D{dim}"


@pytest.fixture(scope="session")
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; rerun the contract tests on the target SM120 GPU")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("These kernel tests require the target SM120 GPU (compute capability 12.0)")
    return torch.device("cuda", torch.cuda.current_device())


def _inputs(shape, device, *, seed=SEED):
    batch, hq, hkv, length, dim = shape
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        batch, hq, 1, dim, dtype=torch.bfloat16, device=device, generator=generator
    )
    k = torch.randn(
        batch, hkv, length, dim, dtype=torch.bfloat16, device=device, generator=generator
    )
    v = torch.randn(
        batch, hkv, length, dim, dtype=torch.bfloat16, device=device, generator=generator
    )
    return q, k, v


def _assert_result(output, expected, q, record_property):
    assert isinstance(output, torch.Tensor)
    assert output.shape == q.shape
    assert output.dtype == torch.bfloat16
    assert output.device == q.device
    assert torch.isfinite(output).all().item(), "output contains NaN or Inf"
    assert torch.isfinite(expected).all().item(), "reference contains NaN or Inf"
    absolute = (output.float() - expected.float()).abs()
    # The floor only defines the diagnostic; torch.testing uses unmodified
    # expected values with the fixed contract's absolute/relative tolerance.
    relative = absolute / expected.float().abs().clamp_min(1e-6)
    max_abs = absolute.max().item()
    max_rel = relative.max().item()
    record_property("max_abs_error", max_abs)
    record_property("max_rel_error", max_rel)
    record_property("relative_error_denominator_floor", 1e-6)
    record_property("atol", ATOL)
    record_property("rtol", RTOL)
    torch.testing.assert_close(output, expected, atol=ATOL, rtol=RTOL)


def test_workload_contract_contains_all_56_cases():
    assert len(WORKLOADS) == 56
    assert len(set(WORKLOADS)) == 56


@pytest.mark.parametrize("shape", WORKLOADS, ids=_case_id)
def test_full_workload(shape, cuda_device, record_property):
    q, k, v = _inputs(shape, cuda_device)
    expected = reference_attention(q, k, v)
    output = attention_decode(q, k, v)
    for name, value in zip(("B", "Hq", "Hkv", "T", "D"), shape):
        record_property(name, value)
    _assert_result(output, expected, q, record_property)


@pytest.mark.parametrize("dim", (64, 128))
@pytest.mark.parametrize("hkv", (8, 32))
@pytest.mark.parametrize("length", (255, 256, 257, 513))
def test_split_boundaries_and_tail_padding(dim, hkv, length, cuda_device, record_property):
    q, k, v = _inputs((1, 32, hkv, length, dim), cuda_device, seed=SEED + 4)
    expected = reference_attention(q, k, v)
    _assert_result(attention_decode(q, k, v), expected, q, record_property)


@pytest.mark.parametrize("dim,hkv", ((64, 8), (128, 32)))
@pytest.mark.parametrize("length", (2, 15, 16, 17))
def test_short_sequence_mma_padding(dim, hkv, length, cuda_device, record_property):
    q, k, v = _inputs((1, 32, hkv, length, dim), cuda_device, seed=SEED + 5)
    expected = reference_attention(q, k, v)
    _assert_result(attention_decode(q, k, v), expected, q, record_property)


@pytest.mark.parametrize("dim", (64, 128))
@pytest.mark.parametrize("hkv", (8, 32))
def test_split_merge_with_widely_separated_maxima(dim, hkv, cuda_device, record_property):
    q, k, v = _inputs((1, 32, hkv, 513, dim), cuda_device, seed=SEED + 6)
    q.fill_(1)
    q[:, 1::2].neg_()
    k[:, :, :256].fill_(-32)
    k[:, :, 256:512].zero_()
    k[:, :, 512:].fill_(32)
    # Adjacent 256-token splits differ in their maximum logit by at least 256.
    # Alternating Q signs put the dominant split first or last, while a V that
    # is constant across tokens gives an independent normalization invariant.
    values = (
        torch.arange(hkv, device=cuda_device).view(1, hkv, 1, 1) / 8 + 0.125
    ).to(torch.bfloat16)
    v.copy_(values)
    expected = values.repeat_interleave(32 // hkv, dim=1).expand_as(q).contiguous()
    reference = reference_attention(q, k, v)
    torch.testing.assert_close(reference, expected, atol=ATOL, rtol=RTOL)
    _assert_result(attention_decode(q, k, v), expected, q, record_property)


@pytest.mark.parametrize("dim", (64, 128))
@pytest.mark.parametrize("hkv", (8, 32))
@pytest.mark.parametrize("length", (129, 513))
@pytest.mark.parametrize("offset_indices", ((0,), (1,), (2,), (0, 1, 2)), ids=("Q", "K", "V", "QKV"))
def test_contiguous_unaligned_storage_offset(
    dim, hkv, length, offset_indices, cuda_device, record_property
):
    args = list(_inputs((1, 32, hkv, length, dim), cuda_device, seed=SEED + 7))
    for index in offset_indices:
        original = args[index]
        backing = torch.empty(original.numel() + 1, device=cuda_device, dtype=torch.bfloat16)
        shifted = backing[1:].view(original.shape)
        shifted.copy_(original)
        assert shifted.is_contiguous()
        assert shifted.storage_offset() == 1
        assert shifted.data_ptr() % 16 != 0
        args[index] = shifted
    expected = reference_attention(*args)
    _assert_result(attention_decode(*args), expected, args[0], record_property)


@pytest.mark.parametrize("dim", (64, 128))
@pytest.mark.parametrize("hkv", (8, 32))
def test_single_token_is_exact_value(dim, hkv, cuda_device, record_property):
    q, k, v = _inputs((4, 32, hkv, 1, dim), cuda_device)
    q = q * 32
    k = k * 32
    expected = v.repeat_interleave(32 // hkv, dim=1)
    output = attention_decode(q, k, v)
    _assert_result(output, expected, q, record_property)
    assert torch.equal(output, expected), "T=1 must return the mapped V value exactly"


@pytest.mark.parametrize("length", (127, 129))
def test_gqa_uses_contiguous_query_head_groups(length, cuda_device, record_property):
    q, k, v = _inputs((4, 32, 8, length, 64), cuda_device)
    q.zero_()
    k.zero_()
    # Every KV head and batch has a distinct exactly representable V. Uniform
    # attention must preserve it, distinguishing // grouping from % mapping.
    values = (
        torch.arange(4, device=cuda_device).view(4, 1, 1, 1) * 8
        + torch.arange(8, device=cuda_device).view(1, 8, 1, 1)
    ).to(torch.bfloat16)
    v.copy_(values)
    expected = values.repeat_interleave(4, dim=1).expand_as(q).contiguous()
    _assert_result(attention_decode(q, k, v), expected, q, record_property)


@pytest.mark.parametrize("dim", (64, 128))
@pytest.mark.parametrize("length", (129, 8192))
@pytest.mark.parametrize("scale", (8.0, 32.0))
def test_stable_softmax_with_large_logits(dim, length, scale, cuda_device, record_property):
    q, k, v = _inputs((1, 32, 8, length, dim), cuda_device, seed=SEED + 1)
    q = q * scale
    k = k * scale
    expected = reference_attention(q, k, v)
    _assert_result(attention_decode(q, k, v), expected, q, record_property)


@pytest.mark.parametrize("dim", (64, 128))
def test_softmax_handles_large_common_offset(dim, cuda_device, record_property):
    q, k, v = _inputs((1, 32, 8, 129, dim), cuda_device, seed=SEED + 2)
    q.fill_(32)
    k.fill_(32)
    # Equal scores around 8192 or 11585 overflow an unstabilized exp, while
    # the correct attention remains the uniform mean over every token.
    expected = v.float().mean(dim=2, keepdim=True).repeat_interleave(4, dim=1)
    expected = expected.to(torch.bfloat16)
    _assert_result(attention_decode(q, k, v), expected, q, record_property)


@pytest.mark.parametrize("length", (1, 129, 8192))
def test_current_nondefault_cuda_stream(length, cuda_device, record_property):
    # Warm up compilation and allocation before creating the asynchronous
    # dependency. The producer and consumer must both follow the active stream.
    q, k, v = _inputs((1, 32, 8, length, 64), cuda_device, seed=SEED + 3)
    attention_decode(q, k, v)
    torch.cuda.synchronize(cuda_device)
    stream = torch.cuda.Stream(device=cuda_device)
    stream.wait_stream(torch.cuda.current_stream(cuda_device))
    with torch.cuda.stream(stream):
        q.mul_(2)
        k.mul_(2)
        v.add_(1)
        expected = reference_attention(q, k, v)
        output = attention_decode(q, k, v)
        # Enqueue the consumer immediately, before any host synchronization.
        consumed = output.clone()
    stream.synchronize()
    _assert_result(consumed, expected, q, record_property)


def test_fp32_reference_matches_explicit_gqa_formula():
    q, k, v = _inputs((1, 32, 8, 127, 64), torch.device("cpu"))
    heads = []
    for head in range(32):
        kv_head = head // 4
        scores = q[:, head].float() @ k[:, kv_head].float().transpose(-2, -1)
        weights = torch.softmax(scores / math.sqrt(64), dim=-1)
        heads.append(weights @ v[:, kv_head].float())
    expected = torch.stack(heads, dim=1).to(torch.bfloat16)
    actual = reference_attention(q, k, v)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_rejects_cpu_inputs():
    q, k, v = _inputs((1, 32, 8, 129, 64), torch.device("cpu"))
    with pytest.raises((ValueError, RuntimeError, TypeError), match="(?i)cuda|device"):
        attention_decode(q, k, v)


def test_rejects_non_tensor_input():
    q, k, v = _inputs((1, 32, 8, 129, 64), torch.device("cpu"))
    with pytest.raises(TypeError):
        attention_decode(None, k, v)


@pytest.mark.parametrize("index", (0, 1, 2), ids=("Q", "K", "V"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32, torch.int32))
def test_rejects_wrong_dtype(index, dtype, cuda_device):
    args = list(_inputs((1, 32, 8, 129, 64), cuda_device))
    args[index] = args[index].to(dtype)
    with pytest.raises((ValueError, TypeError, RuntimeError), match="(?i)dtype|bf16|bfloat16"):
        attention_decode(*args)


@pytest.mark.parametrize("index", (0, 1, 2), ids=("Q", "K", "V"))
def test_rejects_noncontiguous_input(index, cuda_device):
    args = list(_inputs((1, 32, 8, 129, 64), cuda_device))
    shape = args[index].shape
    backing = torch.empty(*shape[:-1], shape[-1] * 2, device=cuda_device, dtype=torch.bfloat16)
    args[index] = backing[..., ::2]
    assert not args[index].is_contiguous()
    with pytest.raises((ValueError, RuntimeError), match="(?i)contiguous|layout|stride"):
        attention_decode(*args)


INVALID_SHAPES = [
    ("q_rank", 0, (1, 32, 64)),
    ("k_rank", 1, (1, 8, 64)),
    ("v_rank", 2, (1, 8, 64)),
    ("query_length", 0, (1, 32, 2, 64)),
    ("key_batch", 1, (2, 8, 129, 64)),
    ("value_batch", 2, (2, 8, 129, 64)),
    ("value_heads", 2, (1, 4, 129, 64)),
    ("value_length", 2, (1, 8, 128, 64)),
    ("key_dim", 1, (1, 8, 129, 128)),
    ("value_dim", 2, (1, 8, 129, 128)),
    ("head_divisibility", 0, (1, 31, 1, 64)),
    ("empty_query_heads", 0, (1, 0, 1, 64)),
]


@pytest.mark.parametrize("name,index,shape", INVALID_SHAPES, ids=[s[0] for s in INVALID_SHAPES])
def test_rejects_invalid_shapes(name, index, shape, cuda_device):
    args = list(_inputs((1, 32, 8, 129, 64), cuda_device))
    args[index] = torch.empty(shape, device=cuda_device, dtype=torch.bfloat16)
    with pytest.raises((ValueError, RuntimeError)):
        attention_decode(*args)


@pytest.mark.parametrize("shape", ((0, 32, 8, 129, 64), (1, 32, 0, 129, 64), (1, 32, 8, 0, 64)))
def test_rejects_empty_dimensions(shape, cuda_device):
    args = _inputs(shape, cuda_device)
    with pytest.raises((ValueError, RuntimeError)):
        attention_decode(*args)


@pytest.mark.parametrize("dim", (0, 32, 96))
def test_rejects_unsupported_head_dimension(dim, cuda_device):
    args = _inputs((1, 32, 8, 129, dim), cuda_device)
    with pytest.raises((ValueError, RuntimeError)):
        attention_decode(*args)


@pytest.mark.parametrize("index", (0, 1, 2), ids=("Q", "K", "V"))
def test_rejects_mixed_devices(index, cuda_device):
    args = list(_inputs((1, 32, 8, 129, 64), cuda_device))
    args[index] = args[index].cpu()
    with pytest.raises((ValueError, RuntimeError), match="(?i)cuda|device"):
        attention_decode(*args)


@pytest.mark.parametrize("hkv", (4, 16))
def test_rejects_unsupported_gqa_group(hkv, cuda_device):
    args = _inputs((1, 32, hkv, 129, 64), cuda_device)
    with pytest.raises((ValueError, RuntimeError)):
        attention_decode(*args)


@pytest.mark.parametrize("index", (0, 1, 2), ids=("Q", "K", "V"))
def test_rejects_autograd_inputs(index, cuda_device):
    args = list(_inputs((1, 32, 8, 129, 64), cuda_device))
    args[index].requires_grad_(True)
    with pytest.raises((ValueError, RuntimeError), match="(?i)grad|backward|inference|forward"):
        attention_decode(*args)
