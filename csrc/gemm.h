#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

namespace llm_ops_xpu {

 at::Tensor matmul( at::Tensor a,
		  at::Tensor b);

 at::Tensor grouped_gemm(
		 const  at::Tensor& a,
		 const  at::Tensor& b, 
		 const  std::optional<at::Tensor>& offs,
		 const std::optional<at::Tensor>& bias,
		 std::optional<c10::ScalarType> out_dtype
		 );

}  // namespace llm_ops_xpu
