import torch

from torch.testing._internal.common_utils import TestCase
import unittest
from llm_ops_xpu import ops as sycl_ex

a = torch.randn(4096, 4096, device="xpu", dtype=torch.bfloat16, requires_grad=False)
b = torch.randn(4096, 4096, device="xpu", dtype=torch.bfloat16, requires_grad=False)

class TestMatmul(TestCase):
    def _test_matmul(self):
        # Test the matmul operation
        result = sycl_ex.matmul(a, b).to(torch.bfloat16)
        expected = torch.matmul(a, b)
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

    @unittest.skipIf(not torch.xpu.is_available(), "requires Intel GPU")
    def test_matmul_xpu(self):
        # Test the matmul operation on XPU
        self._test_matmul()  

if __name__ == "__main__":
    unittest.main()

            