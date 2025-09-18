import torch

from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    TestCase,
)
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)   
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
        
class TestGroupedGEMM(TestCase):

    def grouped_mm_helper(self, alist, blist, gOlist, agradlist, outlist):
        for a, b, gO, agrad, out in zip(alist, blist, gOlist, agradlist, outlist):
            a = a.clone().detach().requires_grad_()
            b = b.clone().detach().requires_grad_()
            out_ref = torch.mm(a, b.t()).to(torch.float32)
            #out_ref.backward(gO)
            self.assertEqual(out, out_ref, rtol=1e-3, atol=1e-3)
            #self.assertEqual(agrad, a.grad)
            #self.assertEqual(bgrad, b.grad)

    @parametrize("strided", [False, True])
    @parametrize("a_row_major", [True])
    @parametrize("b_row_major", [False, True])
    def test_grouped_gemm_2d_2d(self, strided, a_row_major, b_row_major):
        device = "xpu"
        dtype = torch.bfloat16
        m, n, k, n_groups = 16, 16, 16, 4  # all sizes have to be divisible by 16
        if a_row_major:
            a = torch.randn(m, k * n_groups + k * int(strided), device=device, dtype=dtype)[:, :k * n_groups]
        else:
            a = torch.randn(k * n_groups + k * int(strided), m, device=device, dtype=dtype).t()[:, :k * n_groups]

        if b_row_major:
            b = torch.randn(n, k * n_groups + k * int(strided), device=device, dtype=dtype)[:, :k * n_groups]
        else:
            b = torch.randn(k * n_groups + k * int(strided), n, device=device, dtype=dtype).t()[:, :k * n_groups]

        #a.requires_grad_(True)
        #b.requires_grad_(True)
        offs = torch.arange(k, n_groups * k + 1, k, device=device, dtype=torch.int32)
        out = torch.ops.llm_ops_xpu.grouped_gemm(a, b.t(), offs=offs,
                                out_dtype=torch.float32)
        #gO = torch.rand_like(out)
        gO = torch.ones_like(out)
        #out.backward(gO)
        offs_cpu = offs.cpu()
        alist, blist, agradlist, bgradlist = [], [], [], []
        start = 0
        for i in range(n_groups):
            alist.append(a[:, start:offs_cpu[i]])
            blist.append(b[:, start:offs_cpu[i]])
            #agradlist.append(a.grad[:, start:offs_cpu[i]])
            #bgradlist.append(b.grad[:, start:offs_cpu[i]])
            start = offs_cpu[i]
        self.grouped_mm_helper(alist, blist, gO, agradlist, out)

    @parametrize("strided", [False, True])
    @parametrize("a_row_major", [True]) 
    @parametrize("b_row_major", [False, True])
    def test_grouped_gemm_2d_3d(self, strided, a_row_major, b_row_major):
        device = "xpu"
        dtype = torch.bfloat16
        s_int = int(strided)
        m, n, k, n_groups = 1024, 4096, 1024, 4
        if a_row_major:
            a = torch.randn(m * n_groups, k * (1 + s_int), device=device, dtype=dtype)[:, :k]
        else:
            a = torch.randn(k, (m + 2 * s_int) * n_groups, device=device, dtype=dtype).t()[:m * n_groups, :]

        if b_row_major:
            b = torch.randn(n_groups * (1 + s_int), n, k * (1 + s_int), device=device, dtype=dtype)[::(1 + s_int), :, :k]
        else:
            b = torch.randn(n_groups * (1 + s_int), k * (1 + s_int), n, device=device,
                            dtype=dtype).transpose(-2, -1)[::(1 + s_int), :, :k]

        #a.requires_grad_(True)
        #b.requires_grad_(True)

        a_contig = a if a_row_major else a.t()
        self.assertTrue(a_contig.is_contiguous() is not strided)
        b_contig = b if b_row_major else b.transpose(-2, -1)
        self.assertTrue(b_contig.is_contiguous() is not strided)
        offs = torch.arange(m, n_groups * m + 1, m, device="xpu", dtype=torch.int32)
        out = torch.ops.llm_ops_xpu.grouped_gemm(a, b.transpose(-2, -1), offs=offs,
                                out_dtype=torch.float32)
        gO = torch.rand_like(out)
        #out.backward(gO)
        offs_cpu = offs.cpu()
        alist, agradlist, gOlist, outlist = [], [], [], []
        start = 0
        for i in range(n_groups):
            alist.append(a[start:offs_cpu[i]])
            #agradlist.append(a.grad[start:offs_cpu[i]])
            outlist.append(out[start:offs_cpu[i]])
            #gOlist.append(gO[start:offs_cpu[i]])
            start = offs_cpu[i]
        self.grouped_mm_helper(alist, b, gOlist, agradlist, outlist)
    
    @unittest.skip("ToDo on the XPU device")
    @parametrize("strided", [False])
    @parametrize("a_row_major", [True])
    @parametrize("b_row_major", [False, True])
    def test_grouped_gemm_3d_3d(self, strided, a_row_major, b_row_major):
        device = "xpu"
        dtype = torch.bfloat16
        s_int = int(strided)
        m, n, k, n_groups = 1024, 4096, 1024, 4
        if a_row_major:
            a = torch.randn(n_groups * (1 + s_int), m, k * (1 + s_int), device=device, dtype=dtype)[::(1 + s_int), :, :k]
        else:
            a = torch.randn(n_groups * (1 + s_int), k * (1 + s_int), m, device=device,
                            dtype=dtype).transpose(-2, -1)[::(1 + s_int), :, :k]
        if b_row_major:
            b = torch.randn(n_groups * (1 + s_int), n, k * (1 + s_int), device=device, dtype=dtype)[::(1 + s_int), :, :k]
        else:
            b = torch.randn(n_groups * (1 + s_int), k * (1 + s_int), n, device=device,
                            dtype=dtype).transpose(-2, -1)[::(1 + s_int), :, :k]
        #a.requires_grad_(True)
        #b.requires_grad_(True)

        a_contig = a if a_row_major else a.transpose(-2, -1)
        self.assertTrue(a_contig.is_contiguous() is not strided)
        b_contig = b if b_row_major else b.transpose(-2, -1)
        self.assertTrue(b_contig.is_contiguous() is not strided)

        out = torch.ops.llm_ops_xpu.grouped_gemm(a, b.transpose(-2, -1), out_dtype=torch.float32)
        gO = torch.rand_like(out)
        #out.backward(gO)
        self.grouped_mm_helper(a, b, gO, gO, out)
    
    @parametrize("strided", [False, True])
    @parametrize("a_row_major", [True])
    @parametrize("b_row_major", [False, True])
    def test_grouped_gemm_3d_2d(self, strided, a_row_major, b_row_major):
        device = "xpu"
        dtype = torch.bfloat16
        s_int = int(strided)
        m, n, k, n_groups = 1024, 4096, 1024, 4
        if a_row_major:
            a = torch.randn(n_groups * (1 + s_int), m, k * (1 + s_int), device=device, dtype=dtype)[::(1 + s_int), :, :k]
        else:
            a = torch.randn(n_groups * (1 + s_int), k * (1 + s_int), m, device=device,
                            dtype=dtype).transpose(-2, -1)[::(1 + s_int), :, :k]
        if b_row_major:
            b = torch.randn(n * n_groups, k * (1 + s_int), device=device, dtype=dtype)[:, :k]
        else:
            b = torch.randn(k, n * (n_groups + s_int), device=device, dtype=dtype).transpose(-2, -1)[:n * n_groups, :]

        #a.requires_grad_(True)
        #b.requires_grad_(True)

        a_contig = a if a_row_major else a.transpose(-2, -1)
        self.assertTrue(a_contig.is_contiguous() is not strided)
        b_contig = b if b_row_major else b.transpose(-2, -1)
        self.assertTrue(b_contig.is_contiguous() is not strided)
        offs = torch.arange(n, n_groups * n + 1, n, device="xpu", dtype=torch.int32)
        out = torch.ops.llm_ops_xpu.grouped_gemm(a, b.transpose(-2, -1), offs=offs,
                                out_dtype=torch.bfloat16)
        gO = torch.rand_like(out)
        #out.backward(gO)
        offs_cpu = offs.cpu()
        blist, outlist, bgradlist, gOlist = [], [], [], []
        start = 0
        for i in range(n_groups):
            blist.append(b[start:offs_cpu[i]])
            #bgradlist.append(b.grad[start:offs_cpu[i]])
            outlist.append(out[:, start:offs_cpu[i]])
            #gOlist.append(gO[:, start:offs_cpu[i]])
            start = offs_cpu[i]
        self.grouped_mm_helper(a, blist, gOlist, a, outlist)

instantiate_device_type_tests(TestGroupedGEMM, globals(), allow_xpu=True, except_for="cpu")
  
if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()

            