import torch
from torch import Tensor
__all__ = ["matmul"]

def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Performs matrix multiplication"""
    return torch.ops.llm_xpu_ops.matmul.default(a, b)