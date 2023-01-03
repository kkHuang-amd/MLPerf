## DL params
export BATCHSIZE=56
export GRADIENT_STEPS=1
export LR=0.000425
export MAX_SAMPLES_TERMINATION=4500000
export MAX_STEPS=6700
export OPT_LAMB_BETA_1=0.9
export OPT_LAMB_BETA_2=0.999
export START_WARMUP_STEP=0
export WARMUP_PROPORTION=0.0
export WEIGHT_DECAY_RATE=0.01
export INIT_LOSS_SCALE=1024.0

export EXTRA_PARAMS="--dense_seq_output --unpad --exchange_padding --fused_gelu_bias --fused_mha"
#export EXTRA_PARAMS="--dense_seq_output --unpad --exchange_padding --fused_bias_fc --fused_bias_mha --fused_dropout_add  --fused_gelu_bias --fused_mha"
export PHASE=2
export EVAL_ITER_START_SAMPLES=150000
export EVAL_ITER_SAMPLES=150000

export RESULT_DIR="./results/1-nodes"
mkdir -p $RESULT_DIR
