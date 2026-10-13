"""Additional boundaries and execution-lifetime checks for extreme candidates."""
from __future__ import annotations

import pytest
import torch

from attention_decode import attention_decode
from attention_decode.reference import reference_attention


SEED = 20261013
ATOL = RTOL = 2e-2


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("These tests require the target SM120 GPU")
    return torch.device("cuda", torch.cuda.current_device())


def inputs(device, length, dim=128, hkv=8, batch=1, seed=SEED, hq=32):
    generator = torch.Generator(device=device).manual_seed(seed)
    return tuple(torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)
                 for shape in ((batch, hq, 1, dim), (batch, hkv, length, dim), (batch, hkv, length, dim)))


def check(output, expected, q, record_property, label="result"):
    assert output.shape == q.shape and output.dtype == torch.bfloat16 and output.device == q.device
    assert torch.isfinite(output).all().item()
    absolute = (output.float() - expected.float()).abs()
    relative = absolute / expected.float().abs().clamp_min(1e-6)
    record_property(f"{label}_max_abs_error", absolute.max().item())
    record_property(f"{label}_max_rel_error", relative.max().item())
    record_property("atol", ATOL)
    record_property("rtol", RTOL)
    torch.testing.assert_close(output, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("split", (64, 128, 512, 1024))
@pytest.mark.parametrize("delta", (-1, 0, 1))
@pytest.mark.parametrize("dim,hkv", ((64, 32), (128, 8)))
def test_extreme_split_boundaries(split, delta, dim, hkv, device, record_property):
    q, k, v = inputs(device, split + delta, dim, hkv, seed=SEED + 11)
    check(attention_decode(q, k, v), reference_attention(q, k, v), q, record_property)


@pytest.mark.parametrize("offset", range(1, 8))
@pytest.mark.parametrize("dim,hkv", ((64, 8), (128, 32)))
def test_extreme_alignment_residues(offset, dim, hkv, device, record_property):
    args = list(inputs(device, 1025, dim, hkv, seed=SEED + 12))
    for index, residue in enumerate((offset, (offset * 2) % 8, (offset * 3) % 8)):
        original = args[index]
        storage = torch.empty(original.numel() + residue, dtype=original.dtype, device=device)
        shifted = storage[residue:].view_as(original)
        shifted.copy_(original)
        assert shifted.is_contiguous()
        assert shifted.data_ptr() % 16 == 2 * residue
        args[index] = shifted
    check(attention_decode(*args), reference_attention(*args), args[0], record_property)


@pytest.mark.parametrize("offsets", ((0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 1, 1)),
                         ids=("K-odd", "Q-odd", "V-odd", "QKV-odd"))
@pytest.mark.parametrize("dim,hkv", ((64, 8), (128, 32)))
def test_extreme_mixed_pair_alignment(offsets, dim, hkv, device, record_property):
    args = list(inputs(device, 1025, dim, hkv, seed=SEED + 17))
    for index, offset in enumerate(offsets):
        original = args[index]
        storage = torch.empty(original.numel() + offset, dtype=original.dtype, device=device)
        shifted = storage[offset:].view_as(original)
        shifted.copy_(original)
        assert shifted.is_contiguous()
        assert shifted.data_ptr() % 4 == 2 * offset
        args[index] = shifted
    check(attention_decode(*args), reference_attention(*args), args[0], record_property)


@pytest.mark.parametrize("split", (64, 128, 512, 1024))
@pytest.mark.parametrize("dim,hkv", ((64, 32), (128, 8)))
def test_extreme_separated_split_maxima(split, dim, hkv, device, record_property):
    q, k, v = inputs(device, 2 * split + 1, dim, hkv, seed=SEED + 13)
    q.fill_(1)
    q[:, 1::2].neg_()
    k[:, :, :split].fill_(-32)
    k[:, :, split:2 * split].zero_()
    k[:, :, 2 * split:].fill_(32)
    # Both first-dominant and last-dominant attention occur in this one call;
    # retain random V so a wrong dominant token/split cannot pass as a constant.
    check(attention_decode(q, k, v), reference_attention(q, k, v), q, record_property)


@pytest.mark.parametrize("dim,hkv", ((64, 8), (128, 32)))
def test_extreme_common_offset_across_splits(dim, hkv, device, record_property):
    q, k, v = inputs(device, 2049, dim, hkv, seed=SEED + 14)
    q.fill_(32)
    k.fill_(32)
    expected = v.float().mean(dim=2, keepdim=True).repeat_interleave(32 // hkv, dim=1).to(torch.bfloat16)
    torch.testing.assert_close(reference_attention(q, k, v), expected, atol=ATOL, rtol=RTOL)
    check(attention_decode(q, k, v), expected, q, record_property)


@pytest.mark.parametrize("length,dim,hkv", ((8193, 64, 8), (8449, 128, 32)))
def test_extreme_large_split_count(length, dim, hkv, device, record_property):
    q, k, v = inputs(device, length, dim, hkv, seed=SEED + 17)
    check(attention_decode(q, k, v), reference_attention(q, k, v), q, record_property)


@pytest.mark.parametrize("length,dim,hkv", ((129, 64, 8), (1025, 128, 32)))
def test_extreme_graph_input_updates_and_output_lifetime(length, dim, hkv, device, record_property):
    args = inputs(device, length, dim, hkv, seed=SEED + 15)
    replacements = inputs(device, length, dim, hkv, seed=SEED + 16)
    expected_first, expected_second = reference_attention(*args), reference_attention(*replacements)
    attention_decode(*args)
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            attention_decode(*args)
    torch.cuda.current_stream(device).wait_stream(capture_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output = attention_decode(*args)
    graph.replay()
    torch.cuda.synchronize(device)
    check(captured_output, expected_first, args[0], record_property, "first")
    retained_copy = captured_output.clone()
    retained_ordinary = attention_decode(*args)
    for target, replacement in zip(args, replacements):
        target.copy_(replacement)
    graph.replay()
    torch.cuda.synchronize(device)
    check(captured_output, expected_second, args[0], record_property, "updated")
    check(retained_copy, expected_first, args[0], record_property, "retained_copy")
    check(retained_ordinary, expected_first, args[0], record_property, "retained_ordinary")
    del graph
    # User-held graph output must remain valid after destroying its executor.
    torch.cuda.synchronize(device)
    check(captured_output, expected_second, args[0], record_property, "after_graph_destruction")


@pytest.mark.parametrize("length,dim,hkv", ((1025, 128, 8), (513, 64, 32)))
def test_extreme_two_concurrent_streams(length, dim, hkv, device, record_property):
    argument_sets = [inputs(device, length, dim, hkv, seed=SEED + 20 + i) for i in range(2)]
    expected = [reference_attention(*args) for args in argument_sets]
    for args in argument_sets:
        attention_decode(*args)
    streams = [torch.cuda.Stream(device=device) for _ in range(2)]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream(device))
    outputs = []
    for repeat in range(3):
        for index, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                result = attention_decode(*argument_sets[index])
                # An immediate consumer exercises ordering of internal scratch.
                outputs.append((index, repeat, result, result.clone()))
    for stream in streams:
        stream.synchronize()
    for index, repeat, result, consumed in outputs:
        check(result, expected[index], argument_sets[index][0], record_property, f"s{index}_r{repeat}")
        check(consumed, expected[index], argument_sets[index][0], record_property, f"s{index}_r{repeat}_consumer")


@pytest.mark.parametrize("length", (127, 128, 129, 255, 256, 257, 511, 512, 513, 2047, 2048, 2049))
@pytest.mark.parametrize("dim,hkv", ((64, 8), (64, 32), (128, 8), (128, 32)))
def test_extreme_batch4_sequence_boundaries(length, dim, hkv, device, record_property):
    # Every head family must preserve the mathematical operation on both sides
    # of public length boundaries, independently of the selected kernel policy.
    q, k, v = inputs(device, length, dim, hkv, batch=4, seed=SEED + 30)
    check(attention_decode(q, k, v), reference_attention(q, k, v), q, record_property)


@pytest.mark.parametrize("length", (2559, 2560, 2561))
def test_extreme_batch4_long_sequence_tail(length, device, record_property):
    q, k, v = inputs(device, length, dim=64, hkv=32, batch=4, seed=SEED + 31)
    check(attention_decode(q, k, v), reference_attention(q, k, v), q, record_property)


@pytest.mark.parametrize("length,dim,hkv", ((129, 128, 8), (257, 64, 8), (2049, 64, 32), (511, 128, 32)))
@pytest.mark.parametrize("offset_indices", ((0,), (1,), (2,), (0, 1, 2)), ids=("Q", "K", "V", "QKV"))
def test_extreme_batch4_contiguous_unaligned_inputs(length, dim, hkv, offset_indices, device, record_property):
    args = list(inputs(device, length, dim, hkv, batch=4, seed=SEED + 32))
    for index in offset_indices:
        original = args[index]
        storage = torch.empty(original.numel() + 1, dtype=original.dtype, device=device)
        shifted = storage[1:].view_as(original)
        shifted.copy_(original)
        assert shifted.is_contiguous()
        assert shifted.storage_offset() == 1
        assert shifted.data_ptr() % 4 == 2
        args[index] = shifted
    check(attention_decode(*args), reference_attention(*args), args[0], record_property)


@pytest.mark.parametrize("batch,hq,hkv,dim", ((2, 32, 8, 64), (2, 32, 32, 128), (1, 12, 3, 128), (4, 7, 7, 64)),
                         ids=("B2-GQA", "B2-MHA", "H12-GQA", "H7-MHA"))
@pytest.mark.parametrize("length", (1, 129, 513))
def test_extreme_valid_additional_batch_and_head_counts(batch, hq, hkv, dim, length, device, record_property):
    # These valid public inputs intentionally lie outside the tuned 32-head,
    # B=1/4 benchmark grid. No assertion depends on the implementation route.
    q, k, v = inputs(device, length, dim, hkv, batch=batch, hq=hq, seed=SEED + 33)
    expected = reference_attention(q, k, v)
    output = attention_decode(q, k, v)
    check(output, expected, q, record_property)
    if length == 1:
        assert torch.equal(output, v.repeat_interleave(hq // hkv, dim=1))
