import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME
from setuptools import setup, find_packages
import subprocess

import sys
import warnings
import os

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor

print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

def check_if_rocm_pytorch():
    is_rocm_pytorch = False
    if TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 5):
        from torch.utils.cpp_extension import ROCM_HOME
        is_rocm_pytorch = True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False

    return is_rocm_pytorch

IS_ROCM_PYTORCH = check_if_rocm_pytorch()

if not torch.cuda.is_available() and not IS_ROCM_PYTORCH:
    # https://github.com/NVIDIA/apex/issues/486
    # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt to query torch.cuda.get_device_capability(),
    # which will fail if you are compiling in an environment without visible GPUs (e.g. during an nvidia-docker build command).
    print('\nWarning: Torch did not find available GPUs on this system.\n',
          'If your intention is to cross-compile, this is not an error.\n'
          'By default, Apex will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n'
          'Volta (compute capability 7.0), Turing (compute capability 7.5),\n'
          'and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0).\n'
          'If you wish to cross-compile for a single specific architecture,\n'
          'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n')
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        _, bare_metal_major, _ = get_cuda_bare_metal_version(CUDA_HOME)
        if int(bare_metal_major) == 11:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"
elif not torch.cuda.is_available() and IS_ROCM_PYTORCH:
    print('\nWarning: Torch did not find available GPUs on this system.\n',
          'If your intention is to cross-compile, this is not an error.\n'
          'By default, Apex will cross-compile for the same gfx targets\n'
          'used by default in ROCm PyTorch\n')

if TORCH_MAJOR == 0 and TORCH_MINOR < 4:
      raise RuntimeError("Apex requires Pytorch 0.4 or newer.\n" +
                         "The latest stable release can be obtained from https://pytorch.org/")

cmdclass = {}
ext_modules = []

extras = {}
if "--pyprof" in sys.argv:
    string = "\n\nPyprof has been moved to its own dedicated repository and will " + \
             "soon be removed from Apex.  Please visit\n" + \
             "https://github.com/NVIDIA/PyProf\n" + \
             "for the latest version."
    warnings.warn(string, DeprecationWarning)
    with open('requirements.txt') as f:
        required_packages = f.read().splitlines()
        extras['pyprof'] = required_packages
    try:
        sys.argv.remove("--pyprof")
    except:
        pass
else:
    warnings.warn("Option --pyprof not specified. Not installing PyProf dependencies!")

if "--cpp_ext" in sys.argv or "--cuda_ext" in sys.argv:
    if TORCH_MAJOR == 0:
        raise RuntimeError("--cpp_ext requires Pytorch 1.0 or later, "
                           "found torch.__version__ = {}".format(torch.__version__))
    cmdclass['build_ext'] = BuildExtension
if "--cpp_ext" in sys.argv:
    sys.argv.remove("--cpp_ext")
    ext_modules.append(
        CppExtension('fused_lars_C',
                     ['csrc/flatten_unflatten.cpp',]))

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor

def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_major = torch.version.cuda.split(".")[0]
    torch_binary_minor = torch.version.cuda.split(".")[1]

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):
        raise RuntimeError("Cuda extensions are being compiled with a version of Cuda that does " +
                           "not match the version used to compile Pytorch binaries.  " +
                           "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda) +
                           "In some cases, a minor-version mismatch will not cause later errors:  " +
                           "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
                           "You can try commenting out this check (at your own risk).")

# Set up macros for forward/backward compatibility hack around
# https://github.com/pytorch/pytorch/commit/4404762d7dd955383acee92e6f06b48144a0742e
# and
# https://github.com/NVIDIA/apex/issues/456
# https://github.com/pytorch/pytorch/commit/eb7b39e02f7d75c26d8a795ea8c7fd911334da7e#diff-4632522f237f1e4e728cb824300403ac
version_ge_1_1 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
    version_ge_1_1 = ['-DVERSION_GE_1_1']
version_ge_1_3 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    version_ge_1_3 = ['-DVERSION_GE_1_3']
version_ge_1_5 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 4):
    version_ge_1_5 = ['-DVERSION_GE_1_5']
version_dependent_macros = version_ge_1_1 + version_ge_1_3 + version_ge_1_5

if "--cuda_ext" in sys.argv:
    from torch.utils.cpp_extension import CUDAExtension

    if torch.utils.cpp_extension.CUDA_HOME is None and not IS_ROCM_PYTORCH:
        raise RuntimeError("--cuda_ext was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.")
    else:
        if not IS_ROCM_PYTORCH:
            check_cuda_torch_binary_vs_bare_metal(torch.utils.cpp_extension.CUDA_HOME)

        print ("INFO: Building the multi-tensor apply extension.")
        nvcc_args_multi_tensor = ['-lineinfo', '-O3', '--use_fast_math'] + version_dependent_macros
        hipcc_args_multi_tensor = ['-O3'] + version_dependent_macros
        ext_modules.append(
            CUDAExtension(name='fused_lars_C',
                          sources=['csrc/fused_lars_C_frontend.cpp',
                                   'csrc/multi_tensor_scale_kernel.cu',
                                   'csrc/multi_tensor_axpby_kernel.cu',
                                   'csrc/multi_tensor_l2norm_kernel.cu',
                                   'csrc/multi_tensor_l2norm_kernel_mp.cu',
                                   'csrc/multi_tensor_l2norm_scale_kernel.cu',
                                   'csrc/multi_tensor_lars.cu'],
                          include_dirs=[os.path.join(this_dir, 'csrc')],
                          extra_compile_args={'cxx': ['-O3'] + version_dependent_macros,
                                              'nvcc': nvcc_args_multi_tensor if not IS_ROCM_PYTORCH else hipcc_args_multi_tensor}))

# Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
# See https://github.com/pytorch/pytorch/pull/70650
generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "cuda", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DNEW_GENERATOR_PATH"]

if "--cuda_ext" in sys.argv:
    sys.argv.remove("--cuda_ext")

setup(
    name='fused_lars',
    version='0.1',
    packages=find_packages(exclude=('build',
                                    'csrc',
                                    'include',
                                    'dist',
                                    'docs',
                                    'fused_lars.egg-info',)),
    description='PyTorch Extensions written by AMD',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    #cmdclass={'build_ext': BuildExtension} if ext_modules else {},
    extras_require=extras,
)
