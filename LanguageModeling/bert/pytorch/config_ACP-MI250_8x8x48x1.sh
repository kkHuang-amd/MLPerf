## DL params
export BATCHSIZE=48
export GRADIENT_STEPS=1
export LR=1.5e-3
export MAX_SAMPLES_TERMINATION=4500000
export MAX_STEPS=1271
export OPT_LAMB_BETA_1=0.83
export OPT_LAMB_BETA_2=0.925
export START_WARMUP_STEP=-25
export WARMUP_STEPS=100

export EXTRA_PARAMS="--dense_seq_output --unpad --exchange_padding --fused_gelu_bias --fused_mha"
export PHASE=2
export EVAL_ITER_START_SAMPLES=175000
export EVAL_ITER_SAMPLES=175000

export RESULT_DIR="./results/8-nodes"
mkdir -p $RESULT_DIR
