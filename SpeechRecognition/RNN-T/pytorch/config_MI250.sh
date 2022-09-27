## System config params
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export DGXNGPU=8
export DWU_GROUP_SIZE=4
# export DGXSOCKETCORES=64
# export DGXNSOCKET=2
# export DGXHT=2         # HT is on is 2, HT off is 1

## Run specific params
export DATADIR="/datasets/"
export METADATA_DIR="/datasets/tokenized/"
export SENTENCEPIECES_DIR="/datasets/sentpiece"
export BATCHSIZE=120
export EVAL_BATCHSIZE=128
export GRAD_ACCUMULATION_STEPS=1
WALLTIME_MINUTES=120
export WALLTIME=${WALLTIME:-$(( ${NEXP:-1} * ${WALLTIME_MINUTES} ))}
export MAX_SYMBOL=300
export DATA_CPU_THREADS=16

source $(dirname ${BASH_SOURCE[0]})/hyperparameters_512.sh

## Opt flag
export FUSE_RELU_DROPOUT=true
export MULTI_TENSOR_EMA=true
export BATCH_EVAL_MODE=no_cg
export APEX_LOSS=fp32
export APEX_JOINT=pack_w_relu_dropout
export AMP_LVL=0
export BUFFER_PREALLOC=true
export VECTORIZED_SA=true
export EMA_UPDATE_TYPE=fp32
export DIST_LAMB=false
export MULTILAYER_LSTM=true
export ENABLE_PREFETCH=true
export TOKENIZED_TRANSCRIPT=true
export VECTORIZED_SAMPLER=true
export DIST_SAMPLER=true
export MIN_SEQ_SPLIT_LEN=20
export FC_IMPL=apex_fused_dense
export PRE_SORT_FOR_SEQ_SPLIT=true
export LOG_FREQUENCY=1
