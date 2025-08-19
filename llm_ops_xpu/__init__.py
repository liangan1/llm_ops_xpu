import ctypes
from pathlib import Path

# import torch

# current_dir = Path(__file__).parent.parent
# build_dir = current_dir / "build"
# so_files = list(build_dir.glob("**/*.so"))
# print(so_files)
# assert len(so_files) == 1, f"Expected one _C*.so file, found {len(so_files)}"

# with torch._ops.dl_open_guard():
#     loaded_lib = ctypes.CDLL(so_files[0])

from . import ops

__all__ = [
    # "loaded_lib",
    "ops",
]
