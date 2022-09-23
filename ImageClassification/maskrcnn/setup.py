# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#!/usr/bin/env python

import glob
import os
import copy
import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension
from torch.utils.hipify import hipify_python

requirements = ["torch", "torchvision"]

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 4], "Requires PyTorch >= 1.4"


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "maskrcnn_benchmark", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "vision.cpp"))
    main_file_nhwc = glob.glob(os.path.join(extensions_dir, "cuda/nhwc.cpp"))
    main_file_nhwcr = glob.glob(os.path.join(extensions_dir, "cuda/nhwcr.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda_nhwc = glob.glob(os.path.join(extensions_dir, "cuda/nhwc", "*.cu"))
    source_cpp_nhwc = glob.glob(os.path.join(extensions_dir, "cuda/nhwc", "*.cpp"))
    source_cuda_nhwcr = glob.glob(os.path.join(extensions_dir, "cuda/nhwcr", "*.cu"))
    source_cpp_nhwcr = glob.glob(os.path.join(extensions_dir, "cuda/nhwcr", "*.cpp"))
    source_cuda_upsample = glob.glob(os.path.join(extensions_dir, "cuda/nhwc/UpSampleNearest2d.cu"))

    sources = main_file + source_cpu
    sources_nhwc = source_cpp_nhwc + source_cuda_nhwc + main_file_nhwc
    sources_nhwcr = main_file_nhwcr + source_cuda_upsample + source_cpp_nhwcr + source_cuda_nhwcr
    if os.getenv("MASKRCNN_ROCM_BUILD", "0") == "1":
        sources_nhwc=sources_nhwcr

    is_rocm_pytorch = False
    if torch_ver >= [1, 5]:
        from torch.utils.cpp_extension import ROCM_HOME

        is_rocm_pytorch = (
            True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
        )

    print('rocm pytorch detected', is_rocm_pytorch)
    hipify_ver = (
        [int(x) for x in torch.utils.hipify.__version__.split(".")]
        if hasattr(torch.utils.hipify, "__version__")
        else [0, 0, 0]
    )

    if is_rocm_pytorch and hipify_ver < [1, 0, 0]:
        hipify_python.hipify(
            project_directory=this_dir,
            output_directory=this_dir,
            includes="maskrcnn_benchmark/csrc/*",
            show_detailed=True,
            is_pytorch_extension=True,
        )

        # Current version of hipify function in pytorch creates an intermediate directory
        # named "hip" at the same level of the path hierarchy if a "cuda" directory exists,
        # or modifying the hierarchy, if it doesn't. Once pytorch supports
        # "same directory" hipification (https://github.com/pytorch/pytorch/pull/40523),
        # the source_cuda will be set similarly in both cuda and hip paths, and the explicit
        # header file copy (below) will not be needed.
        #source_cuda = glob.glob(path.join(extensions_dir, "**", "hip", "*.hip")) + glob.glob(
        #    path.join(extensions_dir, "hip", "*.hip")
        #)
        source_cuda = glob.glob(os.path.join(extensions_dir, "hip", "*.hip"))

        #shutil.copy(
        #    "detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_utils.h",
        #    "detectron2/layers/csrc/box_iou_rotated/hip/box_iou_rotated_utils.h",
        #)
        #shutil.copy(
        #    "detectron2/layers/csrc/deformable/deform_conv.h",
        #    "detectron2/layers/csrc/deformable/hip/deform_conv.h",
        #)

    else:
        source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
        #source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        #    path.join(extensions_dir, "*.cu")
        #)

    #sources = [main_source] + sources
    #sources = [
    #    s
    #    for s in sources
    #    if not is_rocm_pytorch or torch_ver < [1, 7] or not s.endswith("hip/vision.cpp")
    #]

    ### old code resumes here

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    #if CUDA_HOME is not None:
    #if (torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)):
    if (torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda

        if not is_rocm_pytorch:
            define_macros += [("WITH_CUDA", None)]
            extra_compile_args["nvcc"] = [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]
        else:
            define_macros += [("WITH_HIP", None)]
            extra_compile_args["nvcc"] = []

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "maskrcnn_benchmark._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        ),
        extension("maskrcnn_benchmark.NHWC",
            sources_nhwc,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=copy.deepcopy(extra_compile_args),
        )
    ]

    return ext_modules


setup(
    name="maskrcnn_benchmark",
    version="0.1",
    author="fmassa",
    url="https://github.com/facebookresearch/maskrcnn-benchmark",
    description="object detection in pytorch",
    packages=find_packages(exclude=("configs", "tests",)),
    # install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
