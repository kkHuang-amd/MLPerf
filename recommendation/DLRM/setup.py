"""Simple setup script"""

import os
import subprocess
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import re
import pathlib

abspath = os.path.dirname(os.path.realpath(__file__))

with open("requirements.txt") as f:
    requirements = f.read().splitlines()
"""Get the rocm version info"""
ver = open("/opt/rocm/.info/version-dev").read().strip().split('.')
major,minor = int(ver[0]),int(ver[1])
rocm_version=(major*100 + minor)

print(find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]))

is_rocm_pytorch = False
maj_ver, min_ver, _ = torch.__version__.split('.')
if int(maj_ver) >= 1 and int(min_ver) >= 5:
    from torch.utils.cpp_extension import ROCM_HOME
    is_rocm_pytorch = True \
            if ((torch.version.hip is not None) and (ROCM_HOME is not None)) \
            else False

if not is_rocm_pytorch:
    extension = [
            CUDAExtension(
                name="dlrm.cuda_ext",
                sources=[
                    os.path.join(abspath, "src/pytorch_ops.cpp"),
                    os.path.join(abspath, "src/dot_based_interact_pytorch_types.cu"),
                    os.path.join(abspath, "src/gather_gpu.cu")
                    ],
                extra_compile_args={
                    'cxx': [],
                    'nvcc' : [
                        '-DUSE_WMMA=1',
                        '-DCUDA_HAS_FP16=1',
                        '-D__CUDA_NO_HALF_OPERATORS__',
                        '-D__CUDA_NO_HALF_CONVERSIONS__',
                        '-D__CUDA_NO_HALF2_OPERATORS__',
                        '-gencode', 'arch=compute_70,code=sm_70',
                        '-gencode', 'arch=compute_70,code=compute_70',
                        '-gencode', 'arch=compute_80,code=sm_80']
                    })
            ]
else:
    hipify_version = [int(x) for x in torch.utils.hipify.__version__.split(".")] \
            if hasattr(torch.utils.hipify, "__version__") else [0,0,0]

    use_rocwmma = int(os.getenv("USE_ROCWMMA", 0)) == 1

    if hipify_version < [1,0,0]:
        import shutil
        import re
        import tempfile
        from torch.utils.hipify import hipify_python

        with hipify_python.GeneratedFileCleaner(keep_intermediates=True) as clean_ctx:
            hipify_python.hipify(
                    project_directory=abspath,
                    output_directory=abspath,
                    includes="src/*",
                    show_detailed=True,
                    is_pytorch_extension=True,
                    clean_ctx=clean_ctx)

            shutil.copy("src/pytorch_ops.cpp", "src/hip/pytorch_ops.cpp")
            shutil.copy("src/dense_sparse_add.hip", "src/hip/dense_sparse_add.hip")

            def replace_pattern(hip_file, regexp, replacement):
                pattern = re.compile(regexp)
                with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
                    with open(hip_file) as src_file:
                        for line in src_file:
                            tmp_file.write(pattern.sub(replacement, line))
                shutil.copystat(hip_file, tmp_file.name)
                shutil.move(tmp_file.name, hip_file)

            replace_pattern("src/hip/dot_based_interact_pytorch_types.hip", "(#include.*)[.]cu", "\\1.hip")

        sources = [
                os.path.join(abspath, "src/hip/pytorch_ops.cpp"),
                os.path.join(abspath, "src/hip/dot_based_interact_pytorch_types.hip"),
                os.path.join(abspath, "src/hip/gather_gpu.hip"),
                os.path.join(abspath, "src/hip/dense_sparse_add.hip"),
                os.path.join(abspath, "src/hip/unique.hip"),
                os.path.join(abspath, "src/hip/dedup.hip"),
                os.path.join(abspath, "src/hip/index_select.hip"),
                os.path.join(abspath, "src/hip/fused_dot_emb_compression.hip"),
                os.path.join(abspath, "src/hip/dot_based_interact_common.cuh"),
                os.path.join(abspath, "src/hip/dot_based_interact.cu"),
                os.path.join(abspath, "src/hip/dot_based_interact_no_wmma.cu"),
                ]

        extra_args = []
        include_dirs = []
        cxx_args = []
    else:
        sources = [
                os.path.join(abspath, "src/pytorch_ops.cpp"),
                os.path.join(abspath, "src/dot_based_interact_pytorch_types.cu"),
                os.path.join(abspath, "src/gather_gpu.cu"),
                os.path.join(abspath, "src/dense_sparse_add.hip"),
                os.path.join(abspath, "src/unique.hip"),
                os.path.join(abspath, "src/dedup.hip"),
                os.path.join(abspath, "src/index_select.hip"),
                os.path.join(abspath, "src/fused_dot_emb_compression.hip"),
                os.path.join(abspath, "src/dot_based_interact_common.cuh"),
                os.path.join(abspath, "src/dot_based_interact.cu"),
                os.path.join(abspath, "src/dot_based_interact_no_wmma.cu"),
                ]

        extra_args = ["-DHIP_ATOMIC_RETURN_FLOAT"]
        include_dirs = []
        cxx_args = []

    if use_rocwmma:
        extra_args += ["-U__HIP_NO_HALF_CONVERSIONS__", "-DUSE_WMMA=1"]
        cxx_args.append("-DUSE_WMMA=1")

        rocwmma_path = os.getenv("ROCWMMA_PATH", "")
        if rocwmma_path == "":
            raise Exception("ROCWMMA_PATH must be set when USE_ROCWMMA=1")

        include_dirs.append(os.path.join(rocwmma_path, "library/include"))

    extra_args += ["-DROCM_VERSION=%d" % rocm_version]

    gpu_target = os.getenv("DLRM_AMDGPU_TARGET")
    if gpu_target is None:
        rocm_info = pathlib.Path("/opt/rocm/bin/rocminfo")
        if rocm_info.is_file():
            gpus = subprocess.check_output("/opt/rocm/bin/rocminfo").decode('UTF-8').split('\n')
            gpus = list(set([re.search('(gfx90.)', g).group(0) for g in gpus if 'gfx90' in g]))
            extra_args += ["--amdgpu-target=" + g for g in gpus]
    else:
        print("Found DLRM_AMDGPU_TARGET={}".format(gpu_target))
        extra_args += ["--amdgpu-target=" + gpu_target]

    extension = [
            CUDAExtension(
                name="dlrm.cuda_ext",
                sources=sources,
                extra_compile_args={
                    'cxx': cxx_args,
                    'nvcc' : [
                        '-DCUDA_HAS_FP16=1',
                        '-D__CUDA_NO_HALF_OPERATORS__',
                        '-D__CUDA_NO_HALF_CONVERSIONS__',
                        '-D__CUDA_NO_HALF2_OPERATORS__'] + extra_args
                    },
                include_dirs=include_dirs)
            ]

setup(name="dlrm",
      package_dir={'dlrm': 'dlrm'},
      version="1.0.0",
      description="Reimplementation of Facebook's DLRM",
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      install_requires=requirements,
      zip_safe=False,
      ext_modules=extension,
      cmdclass={"build_ext": BuildExtension})
