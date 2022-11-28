#!/bin/bash

kill `ps aux | grep train.py | awk '{print $2}'`

set -euxo pipefail

: "${LOGDIR:=$(pwd)/results}"

source config_MI250.sh
export ROCBLAS_INTERNAL_FP16_ALT_IMPL=1
export MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=1

mkdir -p ${LOGDIR}
rm -rf `find . -name __pycache__` ${LOGDIR}/train ${LOGDIR}/dev_ema ${LOGDIR}/nvlog*

readonly log="${LOGDIR}/${LR}_${EMA}_${WARMUP}_${HOLD_EPOCHS}_${WEIGHTS_INIT_SCALE}_${BATCHSIZE}_${AMP_LVL}_${APEX_LOSS}.log"

rm -f ${LOGDIR}/latest.log
echo > ${log}
ln -s `basename ${log}` ${LOGDIR}/latest.log

readonly TORCH_RUN="python -m torch.distributed.run --standalone --no_python"
(${TORCH_RUN} --nproc_per_node=${DGXNGPU} ./run_and_time.sh) |& tee ${log}
