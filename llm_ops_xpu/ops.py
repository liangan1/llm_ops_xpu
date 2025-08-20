import torch
from torch import Tensor
import llm_ops_xpu
__all__ = ["matmul"]

def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Performs matrix multiplication"""
    return torch.ops.llm_ops_xpu.matmul.default(a, b)