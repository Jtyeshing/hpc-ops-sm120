#include <torch/extension.h>

at::Tensor attention_decode_cuda(const at::Tensor& q, const at::Tensor& k,
                                 const at::Tensor& v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &attention_decode_cuda, "SM120 BF16 attention decode");
}
