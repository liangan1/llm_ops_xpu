import os
import torch
import glob
from setuptools import find_packages, setup
from torch.utils.cpp_extension import SyclExtension, BuildExtension

library_name = "llm_ops_xpu"
py_limited_api = True
extra_compile_args = {
    "sycl": ["-O3",
             "-DCUTLASS_ENABLE_SYCL",
             "-DSYCL_INTEL_TARGET",
             "-fsycl-targets=intel_gpu_bmg_g21",
            ]
}

assert(torch.xpu.is_available()), "XPU is not available, please check your environment"
# Source files collection
this_dir = os.path.dirname(os.path.abspath(__file__))
extensions_dir = os.path.join(this_dir, "csrc")
sources = list(glob.glob(os.path.join(extensions_dir, "*.sycl")))
# Construct extension
ext_modules = [
    SyclExtension(
        f"{library_name}._C",
        sources,
        include_dirs = [
            f"{this_dir}/third_party/cutlass/include/",
            f"{this_dir}/third_party/cutlass/examples/common",
            f"{this_dir}/third_party/cutlass/tools/util/include/",
            f"{this_dir}/csrc"
        ],
        extra_compile_args=extra_compile_args,
        py_limited_api=py_limited_api,
    )
]
setup(
    name=library_name,
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=["torch"],
    description="A repo to implementate the GEMM ops based on cutlass on XPU",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)
