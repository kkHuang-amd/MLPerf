## Environment variables for multi node runs
export HOROVOD_CYCLE_TIME=0.1
export HOROVOD_FUSION_THRESHOLD=67108864
export HOROVOD_NUM_STREAMS=2

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=23:00:00

## System config params
export DGXNGPU=1
export DGXSOCKETCORES=16
export DGXHT=2 	# HT is on is 2, HT off is 1

## Data mount location
export DATADIR=/data/minigo_data_19x19/

## supress Tensorflow messages
## 3->FATAL, 2->ERROR, 1->WARNING, 0
export TF_CPP_MIN_LOG_LEVEL=3

## Benchmark knobs for this config.
export NUM_GPUS_TRAIN=1
export NUM_ITERATIONS=18

#selfplay perf. params
export SP_THREADS=6
export PA_SEARCH=8
export PA_INFERENCE=2
export CONCURRENT_GAMES=32
export PROCS_PER_GPU=2

# Added
export VERBOSE=1
export USE_TRT=1
export TRAIN_BATCH_SIZE=512
# reducing selfplay so iteration completes quicker
export SUGGESTED_GAMES=200
export MIN_GAMES=200

