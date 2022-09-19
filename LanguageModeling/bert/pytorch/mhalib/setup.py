import torch
import setuptools
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.utils.hipify import hipify_python
import os
import subprocess
import re

this_dir = os.path.dirname(os.path.abspath(__file__))

#sets_rocm_pytorch = False
maj_ver, min_ver, _ = torch.__version__.split('.')
if int(maj_ver) >= 1 and int(min_ver) >= 5:
    from torch.utils.cpp_extension import ROCM_HOME
    is_rocm_pytorch = True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False

ext_modules = []

generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, 'include', 'ATen', 'CUDAGenerator.h')):
    generator_flag = ['-DOLD_GENERATOR']

# https://github.com/pytorch/pytorch/pull/71881
# For the extensions which have rocblas_gemm_flags_fp16_alt_impl we need to make sure if at::BackwardPassGuard exists.
# It helps the extensions be backward compatible with old PyTorch versions.
# The check and ROCM_BACKWARD_PASS_GUARD in nvcc/hipcc args can be retired once the PR is merged into PyTorch upstream.

context_file = os.path.join(torch_dir, "include", "ATen", "Context.h")
if os.path.exists(context_file):
    lines = open(context_file, 'r').readlines()
    found_Backward_Pass_Guard = False
    found_ROCmBackward_Pass_Guard = False
    for line in lines:
        if "BackwardPassGuard" in line:
            # BackwardPassGuard has been renamed to ROCmBackwardPassGuard
            # https://github.com/pytorch/pytorch/pull/71881/commits/4b82f5a67a35406ffb5691c69e6b4c9086316a43
            if "ROCmBackwardPassGuard" in line:
                found_ROCmBackward_Pass_Guard = True
            else:
                found_Backward_Pass_Guard = True
            break

print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

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

include_dirs=[os.path.join(this_dir, 'csrc')]

#if is_rocm_pytorch:
#    import shutil
#    with hipify_python.GeneratedFileCleaner(keep_intermediates=True) as clean_ctx:
#        hipify_python.hipify(project_directory=this_dir, output_directory=this_dir, includes="csrc/*",
#				show_detailed=True, is_pytorch_extension=True, clean_ctx=clean_ctx)

if not is_rocm_pytorch:
    ext_modules.append(
		CUDAExtension(
		    name='mhalib',
		    sources=['mha_funcs.cu'],
		    extra_compile_args={
				       'cxx': ['-O3',],
				        'nvcc':['-O3','-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', "--expt-relaxed-constexpr", "-ftemplate-depth=1024", '-gencode=arch=compute_70,code=sm_70','-gencode=arch=compute_80,code=sm_80','-gencode=arch=compute_80,code=compute_80']
				       }
		    )
	    )
elif is_rocm_pytorch:
    gpus = subprocess.check_output("/opt/rocm/bin/rocminfo").decode('UTF-8').split('\n')
    gpus = [re.search('(gfx.*[0-9a-z])', g).group(0) for g in gpus if 'gfx' in g]
    extra_args = ["--amdgpu-target=" + gpus[0]]
    if found_Backward_Pass_Guard:
        extra_args = extra_args + ['-DBACKWARD_PASS_GUARD'] + ['-DBACKWARD_PASS_GUARD_CLASS=BackwardPassGuard']
    if found_ROCmBackward_Pass_Guard:
        extra_args = extra_args + ['-DBACKWARD_PASS_GUARD'] + ['-DBACKWARD_PASS_GUARD_CLASS=ROCmBackwardPassGuard']
    #if torch.__version__ <= '1.8':
    hipify_ver = [int(x) for x in torch.utils.hipify.__version__.split(".")] if hasattr(torch.utils.hipify, "__version__") else [0,0,0]
    if hipify_ver < [1,0,0]:
        import shutil
        with hipify_python.GeneratedFileCleaner(keep_intermediates=True) as clean_ctx:
            hipify_python.hipify(project_directory=this_dir, output_directory=this_dir, includes="csrc/*",
                                    show_detailed=True, is_pytorch_extension=True, clean_ctx=clean_ctx)

        ext_modules.append(
                    CUDAExtension(
                        name='mhalib',
                        sources=['./csrc/hip/mha_funcs.hip'],
                        extra_compile_args={
                                           'cxx': ['-O3',] + version_dependent_macros,
                                           'nvcc':['-O3'] + extra_args
                                           }
                        )
                )
    else:
        ext_modules.append(
                    CUDAExtension(
                        name='mhalib',
                        sources=['./csrc/mha_funcs.cu'],
                        include_dirs=include_dirs,
                        extra_compile_args={
                                           'cxx': ['-O3',],
                                            'nvcc':['-O3','-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', 
                                     "-ftemplate-depth=1024"] + extra_args
                                           }
                        )
                )


setup(
    name='mhalib',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
})

