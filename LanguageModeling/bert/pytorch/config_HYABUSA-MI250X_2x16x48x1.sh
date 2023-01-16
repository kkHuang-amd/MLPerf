## DL params
export BATCHSIZE=48
export GRADIENT_STEPS=1
export LR=0.002
export MAX_SAMPLES_TERMINATION=18400000
export MAX_STEPS=3600
export OPT_LAMB_BETA_1=0.6
export OPT_LAMB_BETA_2=0.996
export START_WARMUP_STEP=0
export WARMUP_PROPORTION=0.0
export WEIGHT_DECAY_RATE=0.01
export INIT_LOSS_SCALE=4096.0

export EXTRA_PARAMS="--dense_seq_output --unpad --exchange_padding --fused_gelu_bias --fused_mha"
export PHASE=2
export EVAL_ITER_START_SAMPLES=150000
export EVAL_ITER_SAMPLES=150000

export RESULT_DIR="./results/4-nodes"
mkdir -p $RESULT_DIR
