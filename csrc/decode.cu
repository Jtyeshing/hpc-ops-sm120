#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <climits>

namespace {
using bf16 = __nv_bfloat16;
constexpr int kThreads = 128;
constexpr int kSplit = 256;

__device__ __forceinline__ unsigned pack(bf16 a, bf16 b) {
  return unsigned(__bfloat16_as_ushort(a)) |
         (unsigned(__bfloat16_as_ushort(b)) << 16);
}

__device__ __forceinline__ void mma(float (&c)[4], unsigned a0, unsigned a1,
                                   unsigned a2, unsigned a3,
                                   unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float warp_max(float x) {
  #pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset));
  return x;
}

__device__ __forceinline__ float warp_sum(float x) {
  #pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    x += __shfl_xor_sync(0xffffffff, x, offset);
  return x;
}

template<int D, bool Aligned>
__device__ __forceinline__ unsigned key_pair(const bf16* ptr, int token,
                                            int col, int length) {
  if (token >= length) return 0;
  const bf16* pair = ptr + token * D + col;
  if constexpr (Aligned)
    return *reinterpret_cast<const unsigned*>(pair);
  return pack(pair[0], pair[1]);
}

template<int D>
__device__ __forceinline__ unsigned value_pair(const bf16* ptr, int token,
                                              int dim, int length) {
  bf16 lo = token < length ? ptr[token * D + dim] : __float2bfloat16(0.f);
  bf16 hi = token + 1 < length ? ptr[(token + 1) * D + dim] : __float2bfloat16(0.f);
  return pack(lo, hi);
}

__device__ __forceinline__ unsigned transpose8(unsigned x) {
  unsigned y;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(y) : "r"(x));
  return y;
}

// Matrix rows are tokens for QK and output dimensions for PV. Matrix columns
// are grouped query heads. The unused MMA columns are always zero.
template<int D, int G, bool Split, bool Aligned>
__global__ void decode_bf16_mma(const bf16* __restrict__ q,
                               const bf16* __restrict__ k,
                               const bf16* __restrict__ v,
                               bf16* __restrict__ out,
                               float* __restrict__ partial,
                               int length, int splits) {
  __shared__ float scores[G][kSplit];
  __shared__ bf16 weights[G][kSplit];
  __shared__ float denominators[G];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = lane >> 2;
  const int col = (lane & 3) * 2;
  const int start = blockIdx.y * kSplit;
  const int count = min(kSplit, length - start);
  const int head_base = blockIdx.x * G;
  q += head_base * D;
  k += (size_t(blockIdx.x) * length + start) * D;
  v += (size_t(blockIdx.x) * length + start) * D;
  const float scale = rsqrtf(float(D));

  for (int token = warp * 16; token < count; token += 64) {
    float acc[4] = {0, 0, 0, 0};
    #pragma unroll
    for (int d = 0; d < D; d += 16) {
      unsigned a0 = key_pair<D, Aligned>(k, token + row, d + col, count);
      unsigned a1 = key_pair<D, Aligned>(k, token + row + 8, d + col, count);
      unsigned a2 = key_pair<D, Aligned>(k, token + row, d + col + 8, count);
      unsigned a3 = key_pair<D, Aligned>(k, token + row + 8, d + col + 8, count);
      unsigned b0 = key_pair<D, Aligned>(q, row, d + col, G);
      unsigned b1 = key_pair<D, Aligned>(q, row, d + col + 8, G);
      mma(acc, a0, a1, a2, a3, b0, b1);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      int h = col + (i & 1);
      int t = token + row + (i >= 2 ? 8 : 0);
      if (h < G) scores[h][t] = t < count ? acc[i] * scale : -CUDART_INF_F;
    }
  }
  __syncthreads();

  if (warp < G) {
    float maximum = -CUDART_INF_F;
    for (int t = lane; t < count; t += 32) maximum = fmaxf(maximum, scores[warp][t]);
    maximum = warp_max(maximum);
    float denominator = 0;
    for (int t = lane; t < kSplit; t += 32) {
      float p = t < count ? expf(scores[warp][t] - maximum) : 0.f;
      denominator += p;
      weights[warp][t] = __float2bfloat16_rn(p);
    }
    denominator = warp_sum(denominator);
    if (lane == 0) {
      denominators[warp] = denominator;
      if constexpr (Split) {
        size_t base = (size_t(head_base + warp) * splits + blockIdx.y) * (D + 2);
        partial[base + D] = maximum;
        partial[base + D + 1] = denominator;
      }
    }
  }
  __syncthreads();

  for (int d = warp * 16; d < D; d += 64) {
    float acc[4] = {0, 0, 0, 0};
    for (int token = 0; token < count; token += 16) {
      unsigned a0, a1, a2, a3;
      if constexpr (Aligned) {
        a0 = transpose8(key_pair<D, true>(v, token + row, d + col, count));
        a1 = transpose8(key_pair<D, true>(v, token + row, d + col + 8, count));
        a2 = transpose8(key_pair<D, true>(v, token + row + 8, d + col, count));
        a3 = transpose8(key_pair<D, true>(v, token + row + 8, d + col + 8, count));
      } else {
        a0 = value_pair<D>(v, token + col, d + row, count);
        a1 = value_pair<D>(v, token + col, d + row + 8, count);
        a2 = value_pair<D>(v, token + col + 8, d + row, count);
        a3 = value_pair<D>(v, token + col + 8, d + row + 8, count);
      }
      unsigned b0 = row < G ? pack(weights[row][token + col], weights[row][token + col + 1]) : 0;
      unsigned b1 = row < G ? pack(weights[row][token + col + 8], weights[row][token + col + 9]) : 0;
      mma(acc, a0, a1, a2, a3, b0, b1);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      int h = col + (i & 1);
      int dim = d + row + (i >= 2 ? 8 : 0);
      if (h < G) {
        if constexpr (Split) {
          partial[(size_t(head_base + h) * splits + blockIdx.y) * (D + 2) + dim] = acc[i];
        } else {
          out[(head_base + h) * D + dim] = __float2bfloat16_rn(acc[i] / denominators[h]);
        }
      }
    }
  }
}

template<int D>
__global__ void merge_splits(const float* __restrict__ partial,
                              bf16* __restrict__ out, int splits) {
  const int dim = threadIdx.x;
  if (dim >= D) return;
  const float* p = partial + size_t(blockIdx.x) * splits * (D + 2);
  float maximum = -CUDART_INF_F;
  for (int s = 0; s < splits; ++s) maximum = fmaxf(maximum, p[s * (D + 2) + D]);
  float numerator = 0, denominator = 0;
  for (int s = 0; s < splits; ++s) {
    float factor = expf(p[s * (D + 2) + D] - maximum);
    numerator += factor * p[s * (D + 2) + dim];
    denominator += factor * p[s * (D + 2) + D + 1];
  }
  out[blockIdx.x * D + dim] = __float2bfloat16_rn(numerator / denominator);
}

__global__ void copy_single_token(const bf16* __restrict__ v,
                                  bf16* __restrict__ out, int heads, int dim,
                                  int group) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < heads * dim) out[i] = v[(i / dim / group) * dim + i % dim];
}

template<int D, int G, bool Aligned>
void launch(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
            at::Tensor& out, cudaStream_t stream) {
  int length = k.size(2), heads = q.size(0) * q.size(1);
  int splits = (length + kSplit - 1) / kSplit;
  auto qp = reinterpret_cast<const bf16*>(q.data_ptr<at::BFloat16>());
  auto kp = reinterpret_cast<const bf16*>(k.data_ptr<at::BFloat16>());
  auto vp = reinterpret_cast<const bf16*>(v.data_ptr<at::BFloat16>());
  auto op = reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>());
  if (splits == 1) {
    decode_bf16_mma<D, G, false, Aligned><<<heads / G, kThreads, 0, stream>>>(qp, kp, vp, op, nullptr, length, 1);
  } else {
    auto partial = at::empty({heads, splits, D + 2}, q.options().dtype(at::kFloat));
    decode_bf16_mma<D, G, true, Aligned><<<dim3(heads / G, splits), kThreads, 0, stream>>>(qp, kp, vp, op, partial.data_ptr<float>(), length, splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    merge_splits<D><<<heads, D, 0, stream>>>(partial.data_ptr<float>(), op, splits);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<int D, int G>
void dispatch_alignment(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                        at::Tensor& out, cudaStream_t stream) {
  bool aligned = ((reinterpret_cast<uintptr_t>(q.data_ptr()) |
                   reinterpret_cast<uintptr_t>(k.data_ptr()) |
                   reinterpret_cast<uintptr_t>(v.data_ptr())) & 3) == 0;
  if (aligned) launch<D, G, true>(q, k, v, out, stream);
  else launch<D, G, false>(q, k, v, out, stream);
}
} // namespace

at::Tensor attention_decode_cuda(const at::Tensor& q, const at::Tensor& k,
                                 const at::Tensor& v) {
  for (const auto& x : {q, k, v}) {
    TORCH_CHECK(x.is_cuda(), "Q/K/V must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "Q/K/V dtype must be BF16 (bfloat16)");
    TORCH_CHECK(x.dim() == 4, "Q/K/V must have rank four");
    TORCH_CHECK(x.is_contiguous(), "Q/K/V must be contiguous");
    TORCH_CHECK(!x.requires_grad(), "attention_decode is forward inference only; no backward/grad support");
    TORCH_CHECK(x.numel() > 0 && x.numel() <= INT_MAX, "empty or excessively large tensors are unsupported");
  }
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "Q/K/V device must match");
  TORCH_CHECK(k.sizes() == v.sizes(), "K/V shape must match");
  TORCH_CHECK(q.size(2) == 1, "query length must be 1");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(3) == k.size(3), "Q/K/V batch and head dimension must match");
  TORCH_CHECK(q.size(1) % k.size(1) == 0, "query heads must be divisible by KV heads");
  int group = q.size(1) / k.size(1), dim = q.size(3);
  TORCH_CHECK(group == 1 || group == 4, "supported GQA group sizes are 1 and 4");
  TORCH_CHECK(dim == 64 || dim == 128, "head dimension must be 64 or 128");
  TORCH_CHECK((k.size(2) + kSplit - 1) / kSplit <= 65535, "sequence exceeds CUDA split grid limit");
  c10::cuda::CUDAGuard guard(q.device());
  auto properties = at::cuda::getDeviceProperties(q.get_device());
  TORCH_CHECK(properties->major == 12 && properties->minor == 0, "this extension requires SM120 (compute capability 12.0)");
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  auto out = at::empty(q.sizes(), q.options());
  if (k.size(2) == 1) {
    int heads = q.size(0) * q.size(1);
    copy_single_token<<<(heads * dim + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const bf16*>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>()), heads, dim, group);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if (dim == 64 && group == 1) dispatch_alignment<64, 1>(q, k, v, out, stream);
  else if (dim == 64 && group == 4) dispatch_alignment<64, 4>(q, k, v, out, stream);
  else if (dim == 128 && group == 1) dispatch_alignment<128, 1>(q, k, v, out, stream);
  else dispatch_alignment<128, 4>(q, k, v, out, stream);
  return out;
}
