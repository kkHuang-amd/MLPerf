#!/bin/bash

set -euxo pipefail

: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"

: "${LOGDIR:=$(pwd)/results}"
: "${DGXNGPU:?Number gpus not set}"

mkdir -p ${LOGDIR}

readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly TORCH_RUN="python -m torch.distributed.run --standalone --no_python"

(${TORCH_RUN} --nproc_per_node=${DGXNGPU} ./run_and_time.sh) |& tee "${_logfile_base}.log"
