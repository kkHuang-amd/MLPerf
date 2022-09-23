## DL params
export EXTRA_PARAMS=""
#export EXTRA_CONFIG='SOLVER.BASE_LR 0.06 SOLVER.MAX_ITER 80000 SOLVER.WARMUP_FACTOR 0.000096 SOLVER.WARMUP_ITERS 625 SOLVER.WARMUP_METHOD mlperf_linear SOLVER.STEPS (24000,32000) SOLVER.IMS_PER_BATCH 48 TEST.IMS_PER_BATCH 8 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 6000 NHWC True'
#export EXTRA_CONFIG='SOLVER.BASE_LR 0.06 SOLVER.MAX_ITER 80000 SOLVER.WARMUP_FACTOR 0.000096 SOLVER.WARMUP_ITERS 625 SOLVER.WARMUP_METHOD mlperf_linear SOLVER.STEPS (24000,32000) SOLVER.IMS_PER_BATCH 48 TEST.IMS_PER_BATCH 8 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 6000 NHWC False'
export EXTRA_CONFIG='SOLVER.BASE_LR 0.01 SOLVER.MAX_ITER 480000 SOLVER.WARMUP_FACTOR 0.000016 SOLVER.WARMUP_ITERS 625 SOLVER.WARMUP_METHOD mlperf_linear SOLVER.STEPS (144000,192000) SOLVER.IMS_PER_BATCH 6 TEST.IMS_PER_BATCH 8 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 6000 NHWC False'

export PYTORCH_JIT=0

## System run parms
export DGXNNODES=1
#export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export DGXSYSTEM='DGX1_rocm_1gpu'
export WALLTIME=04:00:00

## System config params
export DGXNGPU=1
export DGXSOCKETCORES=8
export DGXNSOCKET=1
export DGXHT=1         # HT is on is 2, HT off is 1

export MIOPEN_FIND_MODE=2
export MIOPEN_GEMM_ENFORCE_BACKEND=1
