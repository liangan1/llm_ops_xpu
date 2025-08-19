#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

namespace llm_xpu_ops {

torch::Tensor matmul(torch::Tensor a,
		 torch::Tensor b);

}  // namespace llm_ops_xpu
