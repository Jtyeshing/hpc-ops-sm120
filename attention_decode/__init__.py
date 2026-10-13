"""CUDA C++ BF16 Tensor Core attention decode for SM120."""

from functools import lru_cache
from pathlib import Path
import os


@lru_cache(maxsize=1)
def _get_extension():
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[1]
    build = root / ".build" / "attention_decode_sm120"
    build.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MAX_JOBS", "2")
    return load(
        name="attention_decode_sm120",
        sources=[str(root / "csrc" / "binding.cpp"), str(root / "csrc" / "decode.cu")],
        build_directory=str(build),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_120,code=sm_120", "--ptxas-options=-v"],
        verbose=os.environ.get("ATTENTION_DECODE_BUILD_VERBOSE") == "1",
    )


def attention_decode(q, k, v):
    """Forward-only attention for contiguous BF16 CUDA tensors.

    Q: [B,Hq,1,D], K/V: [B,Hkv,T,D]. D is 64 or 128 and Hq/Hkv
    is 1 or 4. T=1 copies the mapped V; other lengths use BF16 Tensor
    Cores and FP32 softmax/accumulation. Uses the current CUDA stream.
    """
    import torch

    if not all(isinstance(x, torch.Tensor) for x in (q, k, v)):
        raise TypeError("Q, K, and V must be tensors")
    if not all(x.is_cuda for x in (q, k, v)):
        raise ValueError("Q, K, and V must be CUDA tensors")
    return _get_extension().forward(q, k, v)


__all__ = ["attention_decode"]
