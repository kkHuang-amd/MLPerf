FROM rocm/pytorch:rocm5.0.1_ubuntu18.04_py3.7_pytorch_1.10.0

ARG DLRM_AMDGPU_TARGET

WORKDIR /root/mlperf_utils
COPY mlperf_mgpu_utils/. .
RUN python3 setup.py install

WORKDIR /root/dlrm
COPY . .
RUN rm -rf mlperf_mgpu_utils

# Install DLRM
RUN DLRM_AMDGPU_TARGET=$DLRM_AMDGPU_TARGET ./build_dlrm.sh
