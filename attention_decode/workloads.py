"""The fixed first-version workload contract and reproducible inputs."""

from dataclasses import dataclass
from itertools import product

import torch


LENGTHS = (1, 127, 128, 129, 512, 2048, 8192)


@dataclass(frozen=True)
class Workload:
    batch: int
    query_heads: int
    kv_heads: int
    head_dim: int
    sequence_length: int

    @property
    def key(self) -> str:
        return (
            f"b{self.batch}_hq{self.query_heads}_hkv{self.kv_heads}"
            f"_d{self.head_dim}_t{self.sequence_length}"
        )

    @property
    def q_shape(self) -> tuple[int, ...]:
        return (self.batch, self.query_heads, 1, self.head_dim)

    @property
    def kv_shape(self) -> tuple[int, ...]:
        return (self.batch, self.kv_heads, self.sequence_length, self.head_dim)

    @property
    def kv_bytes(self) -> int:
        return 2 * 2 * self.batch * self.kv_heads * self.sequence_length * self.head_dim


def all_workloads() -> tuple[Workload, ...]:
    """Return all 56 required shapes in a stable order."""
    return tuple(
        Workload(batch, query_heads, kv_heads, head_dim, length)
        for batch, (query_heads, kv_heads), head_dim, length in product(
            (1, 4), ((32, 8), (32, 32)), (64, 128), LENGTHS
        )
    )


def make_inputs(
    workload: Workload, *, device: str | torch.device = "cuda", seed: int = 2026
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make contiguous BF16 tensors without modifying the global RNG state."""
    generator = torch.Generator(device=device).manual_seed(seed)
    kwargs = {"device": device, "dtype": torch.bfloat16, "generator": generator}
    return (
        torch.randn(workload.q_shape, **kwargs),
        torch.randn(workload.kv_shape, **kwargs),
        torch.randn(workload.kv_shape, **kwargs),
    )
