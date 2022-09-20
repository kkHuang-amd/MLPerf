#!/bin/bash
set -e

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>

export OMP_NUM_THREADS=8

SEED=${1:--1}

USER="amd"
MAX_EPOCHS=4000
QUALITY_THRESHOLD="0.86330"
START_EVAL_AT=500
EVALUATE_EVERY=20
LEARNING_RATE="1.0"
LR_WARMUP_EPOCHS=200
DATASET_DIR="$PWD/data"
NPROC=8
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1


if [ -d ${DATASET_DIR} ]
then
    # start timing
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "NPROC $NPROC, STARTING TIMING RUN AT $start_fmt"

    # CLEAR YOUR CACHE HERE
    python -c "
from mlperf_logging.mllog import constants
from runtime.logging import mllog_event
mllog_event(key=constants.CACHE_CLEAR, value=True)"

    torchrun --nproc_per_node=${NPROC} main.py \
    --data_dir ${DATASET_DIR} \
    --epochs ${MAX_EPOCHS} \
    --evaluate_every ${EVALUATE_EVERY} \
    --start_eval_at ${START_EVAL_AT} \
    --quality_threshold ${QUALITY_THRESHOLD} \
    --nproc_per_node ${NPROC} \
    --batch_size ${BATCH_SIZE} \
    --optimizer sgd \
    --ga_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --seed ${SEED} \
    --lr_warmup_epochs ${LR_WARMUP_EPOCHS}

    # end timing
    end=$(date +%s)
    end_fmt=$(date +%Y-%m-%d\ %r)
    echo "ENDING TIMING RUN AT $end_fmt"

    # report result
    result=$(( $end - $start ))
    result_name="image_segmentation"
    echo "NPROC $NPROC, RESULT,$result_name,$SEED,$result,$USER,$start_fmt"
else
	echo "Directory ${DATASET_DIR} does not exist"
fi
