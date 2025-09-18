import torch

from torch.testing._internal.common_utils import TestCase
import unittest
from llm_ops_xpu import ops as sycl_ex

@unittest.skip("")
class TestMatmul(TestCase):
    def _test_matmul(self):
        a = torch.randn(4096, 2048, device="xpu", dtype=torch.bfloat16, requires_grad=False)
        b = torch.randn(2048, 4096, device="xpu", dtype=torch.bfloat16, requires_grad=False)
        # Test the matmul operation
        result = sycl_ex.matmul(a, b).to(torch.bfloat16)
        expected = torch.matmul(a, b)
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

    @unittest.skipIf(not torch.xpu.is_available(), "requires Intel GPU")
    def test_matmul_xpu(self):
        # Test the matmul operation on XPU
        self._test_matmul()  

# @unittest.skip("reason")
class TestGroupedGEMM(TestCase):
    def grouped_mm_helper(self, alist, blist, outlist):
        for a, b, out in zip(alist, blist, outlist):
            out_ref = torch.matmul(a, b.transpose(-2, -1).contiguous())
            self.assertEqual(out, out_ref)
            
    def test_grouped_gemm_2d_3d(self):
        device = "xpu"
        
        m, n, k, n_groups = 1024, 4096, 2048, 4
        a = torch.randn(m * n_groups, k, device=device).to(torch.bfloat16)[:, :k]
        b = torch.randn(n_groups, n, k, device=device).to(torch.bfloat16)[::(1), :, :k]
        offs = torch.arange(m, n_groups * m + 1, m, device="xpu", dtype=torch.int32)
        out = torch.ops.llm_ops_xpu.grouped_gemm(a, b.transpose(-2, -1).contiguous(), offs=offs).to(torch.bfloat16)
        torch.accelerator.synchronize()
        print("outs:", out)
        offs_cpu = offs.cpu()
        alist, outlist = [], []
        start = 0
        for i in range(n_groups):
            alist.append(a[start:offs_cpu[i]])
            outlist.append(out[start:offs_cpu[i]])
            start = offs_cpu[i]
        self.grouped_mm_helper(alist, b, outlist)
if __name__ == "__main__":
    unittest.main()

            