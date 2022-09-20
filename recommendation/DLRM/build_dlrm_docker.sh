#!/bin/bash

cp -r ../../utils mlperf_mgpu_utils

sudo docker build --no-cache -t dlrm_rocm -f Dockerfile \
  --build-arg DLRM_AMDGPU_TARGET=`./get_rocminfo.sh` .

rm -r mlperf_mgpu_utils
