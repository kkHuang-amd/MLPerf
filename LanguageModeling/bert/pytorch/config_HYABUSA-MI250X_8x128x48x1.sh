## DL params
export BATCHSIZE=48
export GRADIENT_STEPS=1
export LR=0.0020992
export MAX_SAMPLES_TERMINATION=4500000
export MAX_STEPS=1059
export OPT_LAMB_BETA_1=0.60466
export OPT_LAMB_BETA_2=0.85437
export START_WARMUP_STEP=0
export WARMUP_STEPS=0

export EXTRA_PARAMS="--dense_seq_output --unpad --exchange_padding --fused_gelu_bias --fused_mha"
export PHASE=2
export EVAL_ITER_START_SAMPLES=175000
export EVAL_ITER_SAMPLES=175000

export RESULT_DIR="./results/8-nodes"
mkdir -p $RESULT_DIR
