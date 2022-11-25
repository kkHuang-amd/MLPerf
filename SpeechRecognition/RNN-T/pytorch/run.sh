#!/bin/bash

kill `ps aux | grep train.py | awk '{print $2}'`

set -euxo pipefail

: "${LOGDIR:=$(pwd)/results}"

source config_MI250.sh
export ROCBLAS_INTERNAL_FP16_ALT_IMPL=1
export MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=1

mkdir -p ${LOGDIR}

export EPOCH=1000

readonly _logfile_base="${LOGDIR}/${LR}_${WARMUP}_${HOLD_EPOCHS}_${WEIGHTS_INIT_SCALE}_${BATCHSIZE}_${AMP_LVL}_${APEX_LOSS}"
readonly TORCH_RUN="python -m torch.distributed.run --standalone --no_python"

(${TORCH_RUN} --nproc_per_node=${DGXNGPU} ./run_and_time.sh) |& tee "${_logfile_base}.log"
