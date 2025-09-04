#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

namespace llm_ops_xpu {

torch::Tensor matmul(torch::Tensor a,
		 torch::Tensor b);

torch::Tensor grouped_gemm(torch::Tensor a,
		 torch::Tensor b, torch::Tensor offs);

}  // namespace llm_ops_xpu
