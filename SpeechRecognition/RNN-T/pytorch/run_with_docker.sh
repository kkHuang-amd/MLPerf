#!/bin/bash

# Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
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

set -euxo pipefail

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"

# Vars with defaults
: "${NEXP:=5}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${DATADIR:=/lustre/fsr/datasets/speech/jasper/LibriSpeech/}"
: "${LOGDIR:=$(pwd)/results}"
: "${API_LOG_DIR:=./api_logs}" # apiLog.sh output dir
: "${SEED:=$RANDOM}"
: "${WALLTIME_DRYRUN:=$WALLTIME}"

# Other vars
readonly _config_file="./config_${DGXSYSTEM}.sh"
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name=rnn_speech_recognition
_cont_mounts=(
    "--volume=${DATADIR}:/datasets/"
    "--volume=${LOGDIR}:/results"
    "--volume=${METADATA_DIR}:/metadata"
    "--volume=${SENTENCEPIECES_DIR}:/sentencepieces"
)

if [ "${API_LOGGING:-0}" -eq 1 ]; then
    _cont_mounts+=("--volume=${API_LOG_DIR}:/logs")
fi

_cont_mounts+=("--volume=$(pwd):/workspace/rnnt")

# MLPerf vars
MLPERF_HOST_OS=$(
    source /etc/os-release
    source /etc/dgx-release || true
    echo "${PRETTY_NAME} / ${DGX_PRETTY_NAME:-???} ${DGX_OTA_VERSION:-${DGX_SWBUILD_VERSION:-???}}"
)
export MLPERF_HOST_OS

# Setup directories
mkdir -p "${LOGDIR}"

# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(MLPERF_HOST_OS)
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)

# Cleanup container
cleanup_docker() {
    docker container rm -f "${_cont_name}" || true
}
cleanup_docker
# trap 'set -eux; cleanup_docker' EXIT

# Setup container
readonly _docker_gpu_args="--device=/dev/kfd --group-add=video \
	--cap-add=SYS_PTRACE --cap-add SYS_ADMIN --device /dev/fuse --device=/dev/dri"

docker run ${_docker_gpu_args} --rm -it \
    --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
    --ulimit=stack=67108864 --ulimit=memlock=-1 \
    --name="${_cont_name}" "${_cont_mounts[@]}" \
    "${CONT}"
