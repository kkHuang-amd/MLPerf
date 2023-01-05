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

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

set +x
set -e

# Only rank print
[ "${SLURM_LOCALID-0}" -ne 0 ] && set +x

# Set variables
[ "${DEBUG}" = "1" ] && set -x
USERNAME=${USERNAME:-"amd"}
PLATFORM=${PLATFORM:-"MI250x"}
LR=${LR:-0.0001}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
BATCHSIZE=${BATCHSIZE:-2}
EVALBATCHSIZE=${EVALBATCHSIZE:-${BATCHSIZE}}
NUMEPOCHS=${NUMEPOCHS:-10}
LOG_INTERVAL=${LOG_INTERVAL:-20}
DATASET_DIR=${DATASET_DIR:-"/workspace/datasets/retinanet"}
TORCH_HOME=${TORCH_HOME:-"$PWD/torch-home"}
TIME_TAGS=${TIME_TAGS:-0}
NVTX_FLAG=${NVTX_FLAG:-0}
NCCL_TEST=${NCCL_TEST:-0}
EPOCH_PROF=${EPOCH_PROF:-0}
SYNTH_DATA=${SYNTH_DATA:-0}
DISABLE_CG=${DISABLE_CG:-0}
TRACEDUMP=${TRACEDUMP:-0}

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
LOG_DIR="${PWD}/logs"
GPU_SUFFIX="gpu"
if [ ${DGXNGPU} -gt 1 ]; then
  GPU_SUFFIX="${GPU_SUFFIX}s"
fi
RUN_NAME="${USERNAME}_${PLATFORM}_${DGXNGPU}${GPU_SUFFIX}_${HOSTNAME}_${TIMESTAMP}"
RUN_DIR="${LOG_DIR}/${RUN_NAME}"
[ -d ${RUN_DIR} ] || mkdir -p ${RUN_DIR}

PARAM_FILE="${RUN_DIR}/parameters.log"
LOG_FILE="${RUN_DIR}/result.log"

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt" | tee $LOG_FILE

echo "===  parameter setting  ===" | tee $PARAM_FILE
echo "USERNAME: $USERNAME" | tee -a $PARAM_FILE
echo "LR: $LR" | tee -a $PARAM_FILE
echo "WARMUP_EPOCHS: $WARMUP_EPOCHS" | tee -a $PARAM_FILE
echo "BATCHSIZE: $BATCHSIZE" | tee -a $PARAM_FILE
echo "EVALBATCHSIZE: $EVALBATCHSIZE" | tee -a $PARAM_FILE
echo "NUMEPOCHS: $NUMEPOCHS" | tee -a $PARAM_FILE
echo "LOG_INTERVAL: $LOG_INTERVAL" | tee -a $PARAM_FILE
echo "DATASET_DIR: $DATASET_DIR" | tee -a $PARAM_FILE
echo "TORCH_HOME: $TORCH_HOME" | tee -a $PARAM_FILE
echo "TIME_TAGS: $TIME_TAGS" | tee -a $PARAM_FILE
echo "NVTX_FLAG: $NVTX_FLAG" | tee -a $PARAM_FILE
echo "NCCL_TEST: $NCCL_TEST" | tee -a $PARAM_FILE
echo "EPOCH_PROF: $EPOCH_PROF" | tee -a $PARAM_FILE
echo "SYNTH_DATA: $SYNTH_DATA" | tee -a $PARAM_FILE
echo "DISABLE_CG: $DISABLE_CG" | tee -a $PARAM_FILE
echo "TRACEDUMP: $TRACEDUMP" | tee -a $PARAM_FILE
echo "===========================" | tee -a $PARAM_FILE
echo "" | tee -a $PARAM_FILE

# run benchmark
echo "===  running benchmark  ===" | tee -a $PARAM_FILE
if [ ${NVTX_FLAG} -gt 0 ]; then
  # FIXME mfrank 2022-May-24: NSYSCMD needs to be an array, not a space-separated string
  NSYSCMD="/nsight/bin/nsys profile --capture-range cudaProfilerApi --capture-range-end stop --sample=none --cpuctxsw=none  --trace=cuda,nvtx  --force-overwrite true --output /results/single_stage_detector_pytorch_${DGXNNODES}x${DGXNGPU}x${BATCHSIZE}_${DATESTAMP}_${SLURM_PROCID}_${SYNTH_DATA}_${DISABLE_CG}.nsys-rep "
else
  NSYSCMD=""
fi

if [ ${SYNTH_DATA} -gt 0 ]; then
  EXTRA_PARAMS+=" --syn-dataset --cuda-graphs-syn "
  EXTRA_PARAMS=$(echo $EXTRA_PARAMS | sed 's/--dali//')
fi

declare -a CMD
<<comment
if [ -n "${SLURM_LOCALID-}" ]; then
    # Mode 1: Slurm launched a task for each GPU and set some envvars; no need for parallel launch
  if [ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NODES}" ]; then
    CMD=( 'bindpcie' '--ib=single' '--' ${NSYSCMD} 'python' '-u' )
  else
    CMD=( ${NSYSCMD} 'python' '-u' )
  fi
else
  # Mode 2: Single-node Docker, we've been launched with torch_run
  # TODO: Replace below CMD with NSYSCMD..., but make sure NSYSCMD is an array, not a string
  # CMD=( "python" )
  CMD=( "python3" "-m" "torch.distributed.launch" "--use_env" "--standalone" "--nnodes=1" "--nproc_per_node=${DGXNGPU}" )
  [ "$MEMBIND" = false ] && CMD+=( "--no_membind" )
fi
comment

## If TRACEDUMP is set to 1 (or any value larger than 0),
## then trace will be recorded during execution.
if [ ${TRACEDUMP} -gt 0 ]; then
  EXTRA_PARAMS=$(echo $EXTRA_PARAMS | sed 's/--async-coco//')
  if [ -x "$(command -v nvidia-smi)" ]; then
    # nsys
    echo "Use nsys as tracer..."
    TARGET_ID=0
    NSYS_FILE="${RUN_DIR}/retinanet_gpu${TARGET_ID}.nsys-rep"
    TRACER_CMD="nsys profile --capture-range cudaProfilerApi --capture-range-end stop --trace=cuda,nvtx,cudnn,cublas --gpu-metrics-device ${TARGET_ID} --force-overwrite true --output ${NSYS_FILE}"
  elif [ -x "$(command -v rocm-smi)" ]; then
    # rocprof
    echo "Use rocprof as tracer..."
    ROCPROF_FILE="${RUN_DIR}/retinanet.csv"
    TRACER_CMD="rocprof --stats -i in.txt --hip-trace --roctx-trace --timestamp on -d ${RUN_DIR}/rocprof -o ${ROCPROF_FILE}"
  fi
fi

CMD=( "python3" "-m" "torch.distributed.launch" "--use_env" "--standalone" "--nnodes=1" "--nproc_per_node=${DGXNGPU}" )
[ "$MEMBIND" = false ] && CMD+=( "--no_membind" )

if [ "$LOGGER" = "apiLog.sh" ]; then
  LOGGER="${LOGGER} -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
  # TODO(ahmadki): track the apiLog.sh bug and remove the workaround
  # there is a bug in apiLog.sh preventing it from collecting
  # NCCL logs, the workaround is to log a single rank only
  # LOCAL_RANK is set with an enroot hook for Pytorch containers
  # SLURM_LOCALID is set by Slurm
  # OMPI_COMM_WORLD_LOCAL_RANK is set by mpirun
  readonly node_rank="${SLURM_NODEID:-0}"
  readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
  if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ]; then
    LOGGER=$LOGGER
  else
    LOGGER=""
  fi
fi

PARAMS=(
      --lr                      "${LR}"
      --batch-size              "${BATCHSIZE}"
      --eval-batch-size         "${EVALBATCHSIZE}"
      --epochs                  "${NUMEPOCHS}"
      --print-freq              "${LOG_INTERVAL}"
      --dataset-path            "${DATASET_DIR}"
      --warmup-epochs           "${WARMUP_EPOCHS}"
)

echo "EXTRA_PARAMS: $EXTRA_PARAMS" | tee -a $PARAM_FILE
echo "CMD: ${CMD[@]} train.py ${PARAMS[@]} $EXTRA_PARAMS" | tee -a $PARAM_FILE
echo "===========================" | tee -a $PARAM_FILE

# run training
${LOGGER:-} ${TRACER_CMD:-} "${CMD[@]}" train.py "${PARAMS[@]}" ${EXTRA_PARAMS} 2>&1 | tee -a $LOG_FILE
ret_code=$?

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt" | tee -a $LOG_FILE

# report result
result=$(( $end - $start ))
result_name="SINGLE_STAGE_DETECTOR"

echo "RESULT,$result_name,$result,$USERNAME,$start_fmt" | tee -a $LOG_FILE
echo "DIR: ${RUN_DIR}" | tee -a $LOG_FILE

