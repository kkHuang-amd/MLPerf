#!/bin/bash

## User setting
CLUSTER="munich"
if [[ $CLUSTER == "munich" ]]; then
    export DATASET_PATH=/global/scratch/mlperf_datasets/open-images-v6
fi

export TORCH_HOME=$PWD/.cache/torch
export BATCHSIZE=32
export NUMEPOCHS=1
export LR=0.0001
export ROCPROF=1
export EXTRA_PARAMS='--frozen-bn-opt --frozen-bn-fp16 --apex-adam --apex-focal-loss --fp16-allreduce --disable-ddp-broadcast-buffers --reg-head-pad --cls-head-pad --skip-metric-loss --async-coco'
#export EXTRA_PARAMS='--amp --apex-focal-loss --apex-adam --frozen-bn-fp16 --fp16-allreduce'

## Dataset
export DATASET_DIR=${DATASET_PATH:-"/workspace/dataset/retinanet"}

## DL params
export BATCHSIZE=${BATCHSIZE:-32}
export NUMEPOCHS=${NUMEPOCHS:-8}
export LR=${LR:-0.000085}
export WARMUP_EPOCHS=${WARMUP_EPOCHS:-0}
export EXTRA_PARAMS=${EXTRA_PARAMS:-'--amp --frozen-bn-opt --frozen-bn-fp16 --apex-adam --apex-focal-loss --fp16-allreduce --reg-head-pad --cls-head-pad --skip-metric-loss --async-coco'}

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=160
export WALLTIME=$((${NEXP:-1} * ${WALLTIME_MINUTES}))

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1

echo "CLUSTER=$CLUSTER"
echo "HOSTNAME=$HOSTNAME"
echo "DATASET_DIR=$DATASET_DIR"
