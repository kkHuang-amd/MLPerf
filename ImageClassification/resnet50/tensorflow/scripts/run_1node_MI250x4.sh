#!/bin/bash -l

# runs benchmark and reports time to convergence
# to use the script:
#   run.sh
if [[ ${SLURM_NTASKS} -gt 1 ]]; then
    source ./scripts/set_tf_config.sh
fi
export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=master
export DATASETS_NUM_PRIVATE_THREADS=32
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_ROCM_FUSION_ENABLE=1

# SYSTEM
SYSTEM_NAME=${SYSTEM_NAME:-"MI200system"}
NNODES=1
NUM_GPUS_PER_NODE=8
NUM_GPUS=$(($NUM_GPUS_PER_NODE * $NNODES))

# HYPER-PARAMETERS
BASE_LEARNING_RATE=8.5
BATCH_SIZE_PER_DEVICE=384
BATCH_SIZE=$(($BATCH_SIZE_PER_DEVICE * $NUM_GPUS))
ACCUMULATION_STEPS=1
TRAIN_EPOCHS=39
OPTIMIZER="LARS"
LABEL_SMOOTHING="0.1"
LARS_EPSILON=0
LOG_STEPS=125
LR_SCHEDULE="polynomial"
MOMENTUM="0.9"
WARMUP_EPOCHS=2
WEIGHT_DECAY="0.0002"
DATA_TYPE="fp16"
EVAL_OFFSET=3
EVAL_PERIOD=4
WARMUP_STEPS=1

# ID, NAME and PATHS
EXP_ID=${EXP_ID:-"$(date +%y%m%d_%H%M%S)"}
OUT_DIR=${OUT_DIR:-"out/${SYSTEM_NAME}"}
MODEL_DIR=${MODEL_DIR:-"${OUT_DIR}/${EXP_ID}/"}
IMAGENET_HOME=${DATASET_PATH:-"/global/scratch/mlperf_datasets/imagenet/tf_record/"}


# VARIOUS
ALL_REDUCE_ALG=${ALL_REDUCE_ALG:-"nccl"}
TARGET_ACCURACY=${TARGET_ACCURACY:-0.759}
NUM_CLASSES=${NUM_CLASSES:-1000}
EVAL_PREFETCH_BATCHS=${EVAL_PREFETCH_BATCHS:-192}
DATASETS_NUM_PRIVATE_THREADS=${DATASETS_NUM_PRIVATE_THREADS:-32}
TF_GPU_THREAD_MODE=${TF_GPU_THREAD_MODE:-"gpu_private"}

if [ $NNODES -ge 2 ]; then
  DISTRIBUTION_STRATEGY="multi_worker_mirrored"
fi
DISTRIBUTION_STRATEGY="mirrored"

gbs=$((BATCH_SIZE_PER_DEVICE*NUM_GPUS))
tag=gbs${gbs}_${NUM_GPUS}GPUs_epoch${TRAIN_EPOCHS}_lr${BASE_LEARNING_RATE}_lr-sched-${LR_SCHEDULE}_node${SLURM_NODEID}
CURRENTDATE=`date +"%Y-%m-%d-%T"`
LOG=r_tf_${tag}_${CURRENTDATE}.log

env 2>&1 | tee -a ${LOG}
echo "WORKERS_LIST: $WORKERS_LIST"2>&1 | tee -a ${LOG}
echo "TF_CONFIG: $TF_CONFIG" 2>&1 | tee -a ${LOG}

## RUN BENCHMARK
# start timing
start=$(date +%s)
echo "### RUN_START: $(date +%Y-%m-%d\ %r)"
steps_per_loop=$((1281167 / $gbs + 1))

  # --enable_device_warmup \
  # --enable_tensorboard \
  # --model_dir=${MODEL_DIR} \
  # --num_accumulation_steps=${ACCUMULATION_STEPS} \
  # --data_format="channels_last" \
export MIOPEN_USER_DB_PATH=/tmp/miopen-db-luise_node${SLURM_NODEID}
rm -rf ${MIOPEN_USER_DB_PATH}
python3 ./src/tensorflow2/resnet_ctl_imagenet_main.py \
  --data_dir=${IMAGENET_HOME} \
  --distribution_strategy ${DISTRIBUTION_STRATEGY} \
  --num_gpus=${NUM_GPUS} \
  --all_reduce_alg ${ALL_REDUCE_ALG} \
  --base_learning_rate=${BASE_LEARNING_RATE} \
  --batch_size=${BATCH_SIZE} \
  --datasets_num_private_threads=${DATASETS_NUM_PRIVATE_THREADS} \
  --dtype=${DATA_TYPE} \
  --device_warmup_steps=${WARMUP_STEPS} \
  --epochs_between_evals=${EVAL_PERIOD} \
  --eval_offset_epochs=${EVAL_OFFSET} \
  --eval_prefetch_batchs=${EVAL_PREFETCH_BATCHS} \
  --train_epochs=${TRAIN_EPOCHS} \
  --optimizer=${OPTIMIZER} \
  --label_smoothing=${LABEL_SMOOTHING} \
  --lars_epsilon=${LARS_EPSILON} \
  --log_steps=${LOG_STEPS} \
  --lr_schedule=${LR_SCHEDULE} \
  --momentum=${MOMENTUM} \
  --warmup_epochs=${WARMUP_EPOCHS} \
  --weight_decay=${WEIGHT_DECAY} \
  --target_accuracy=${TARGET_ACCURACY} \
  --num_classes=${NUM_CLASSES} \
  --tf_gpu_thread_mode=${TF_GPU_THREAD_MODE} \
  --notrace_warmup \
  --notf_data_experimental_slack \
  --notraining_dataset_cache \
  --noeval_dataset_cache \
  --training_prefetch_batchs=128 \
  --nouse_synthetic_data \
  --noreport_accuracy_metrics \
  --single_l2_loss_op \
  --noskip_eval \
  --steps_per_loop=1252 \
  --enable_eager \
  --noenable_xla \
  --enable_device_warmup \
  --use_tf_keras_layers 2>&1 | tee -a ${LOG}

stop=$(date +%s)
echo "### RUN_END: $(date +%Y-%m-%d\ %r)" 2>&1 | tee -a ${LOG}
echo "### RUN_TIME: $(( $stop - $start ))" 2>&1 | tee -a ${LOG}
