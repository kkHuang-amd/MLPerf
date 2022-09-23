## DL params
export EXTRA_PARAMS=""
#export EXTRA_CONFIG='SOLVER.BASE_LR 0.06 SOLVER.MAX_ITER 80000 SOLVER.WARMUP_FACTOR 0.000096 SOLVER.WARMUP_ITERS 625 SOLVER.WARMUP_METHOD mlperf_linear SOLVER.STEPS (24000,32000) SOLVER.IMS_PER_BATCH 48 TEST.IMS_PER_BATCH 8 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 6000 NHWC True'
#export EXTRA_CONFIG='SOLVER.BASE_LR 0.06 SOLVER.MAX_ITER 80000 SOLVER.WARMUP_FACTOR 0.000096 SOLVER.WARMUP_ITERS 625 SOLVER.WARMUP_METHOD mlperf_linear SOLVER.STEPS (24000,32000) SOLVER.IMS_PER_BATCH 48 TEST.IMS_PER_BATCH 8 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 6000 NHWC False'
#export EXTRA_CONFIG='SOLVER.BASE_LR 0.01 SOLVER.MAX_ITER 480000 SOLVER.WARMUP_FACTOR 0.000016 SOLVER.WARMUP_ITERS 625 SOLVER.WARMUP_METHOD mlperf_linear SOLVER.STEPS (144000,192000) SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 8 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 6000 NHWC True'
export EXTRA_CONFIG='SOLVER.BASE_LR 0.08 SOLVER.MAX_ITER 60000 SOLVER.WARMUP_FACTOR 0.000128 SOLVER.WARMUP_ITERS 625 SOLVER.WARMUP_METHOD mlperf_linear SOLVER.STEPS (18000,24000) SOLVER.IMS_PER_BATCH 64 TEST.IMS_PER_BATCH 8 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 8000 NHWC False DATALOADER.HYBRID True PRECOMPUTE_RPN_CONSTANT_TENSORS True'

#export PYTORCH_JIT=0

## System run parms
export DGXNNODES=1
#export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export DGXSYSTEM='DGX1_rocm_8gpu_NCHW'
export WALLTIME=04:00:00

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=1         # HT is on is 2, HT off is 1

export MIOPEN_FIND_MODE=5
export MIOPEN_GEMM_ENFORCE_BACKEND=1
export MIOPEN_DEBUG_CONV_DIRECT=0
