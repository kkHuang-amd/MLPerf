# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG FROM_IMAGE_NAME=rocm/tensorflow:rocm5.0-tf2.6-dev
FROM ${FROM_IMAGE_NAME}

# Install system dependencies
RUN wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add - \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        cmake \
        python2.7 \
        infiniband-diags \
        pciutils \
 && rm -rf /var/lib/apt/lists/*

# ROCM: build TF C++ API
RUN cd /root/tensorflow/ && bazel build --config=opt --config=rocm //tensorflow:libtensorflow_cc.so --verbose_failures

# Copy MiniGo code
WORKDIR /opt/reinforcement
COPY . .

ENV BOARD_SIZE 19
ENV MINIGO_BAZEL_CACHE_DIR /opt/reinforcement/minigo-bazel-cache

# Copy TF dependency
RUN mkdir minigo/cc/tensorflow/lib \
 && cp /root/tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so.2 minigo/cc/tensorflow/lib \
 && cp /root/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so minigo/cc/tensorflow/lib/libtensorflow_cc.so \
 && cp -r /usr/local/lib/python3.9/dist-packages/tensorflow/include minigo/cc/tensorflow/include

# Install Python dependencies
WORKDIR /opt/reinforcement/minigo
RUN pip3 install --no-cache-dir https://github.com/mlperf/logging/archive/9ea0afa.zip \
 && LC_ALL=C.UTF-8 pip3 install --no-cache-dir -r requirements.txt

# Install pybind11 to enable C++-python interface
RUN pip3 install --no-cache-dir pytest \
 && git clone --branch v2.4.3 --depth 1 https://github.com/pybind/pybind11 /usr/local/src/pybind11 \
 && cd /usr/local/src/pybind11 \
 && cmake -DPYTHON_EXECUTABLE=/usr/bin/python3 . \
 && make install -j$(nproc) \
 && pip3 install --no-cache-dir .

# Copy TF config files - specific to ROCm
RUN grep -F -v "build:rocm --crosstool_top=@local_config_rocm//crosstool:toolchain" /root/tensorflow/.bazelrc > ./.bazelrc \
 && cp /root/tensorflow/.tf_configure.bazelrc .

# Build MIGraphphX
WORKDIR /root
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
RUN git clone -b dev/aayujain/save_load_buffer https://github.com/aayushmaanjain/AMDMIGraphX.git \
 && cd AMDMIGraphX \
 && pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz \
 && rbuild build -d depend --cxx=/opt/rocm/llvm/bin/clang++ \
 && cd build && make install

ENV LD_LIBRARY_PATH "/opt/rocm/lib:${LD_LIBRARY_PATH}"
ENV PYTHONPATH "/opt/rocm/lib:${PYTHONPATH}"

WORKDIR /opt/reinforcement/minigo/cc/migraphx
RUN mkdir include && cp -r /opt/rocm/include/migraphx include \
 && mkdir lib && cp /opt/rocm/lib/libmigraphx* lib

# Build Minigo
# TODO remove verbose_failures
WORKDIR /opt/reinforcement/minigo
RUN mkdir -p "${MINIGO_BAZEL_CACHE_DIR}" \
 && bazel --output_user_root="${MINIGO_BAZEL_CACHE_DIR}" build -c opt -s \
      --copt=-O3 \
      --config=rocm \
      --define=board_size="${BOARD_SIZE}" \
      --define=migraphx=1 \
      --define=tf=1 \
      --verbose_failures \
      cc:minigo_python.so

# Alias python to python3 (it defaults to py2 in out containers"
RUN ln -fs /usr/bin/python3 /usr/bin/python

ENV PYTHONPATH "${PYTHONPATH}:/opt/reinforcement/minigo/bazel-bin/cc"
RUN echo '/usr/local/lib/python3.9/dist-packages/tensorflow_core' > /etc/ld.so.conf.d/tensorflow.conf && ldconfig

# Install tf2onnx
RUN python3 -m pip install --upgrade pip \
 && pip3 install -U tf2onnx

# back to where run* files are
WORKDIR /opt/reinforcement
