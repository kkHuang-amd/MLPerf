# Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
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

ARG FROM_IMAGE_NAME=rocm/pytorch:rocm5.3_ubuntu20.04_py3.7_pytorch_1.12.1
FROM ${FROM_IMAGE_NAME}

RUN pip install --upgrade pip
RUN apt install -y sox jq

WORKDIR /workspace/rnnt

COPY requirements.txt .
RUN pip install --no-cache --disable-pip-version-check -U -r requirements.txt
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
COPY distributed_fused_lamb.py /opt/conda/lib/python3.7/site-packages/apex/contrib/optimizers/distributed_fused_lamb.py
