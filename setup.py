from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="field_ops",
    ext_modules=[
        CUDAExtension(
            "field_ops",
            ["my_kernel.cu"],
            extra_compile_args={"nvcc": ["-O3"]} 
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
