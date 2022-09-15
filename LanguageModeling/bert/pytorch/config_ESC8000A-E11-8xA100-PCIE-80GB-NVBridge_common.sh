## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}

## Data Paths
export DATADIR="/raid/datasets/bert/hdf5/4320_shards"
export EVALDIR="/raid/datasets/bert/hdf5/eval_4320_shard"
export DATADIR_PHASE2="/raid/datasets/bert/hdf5/4320_shards"
export CHECKPOINTDIR="./ci_checkpoints"
export RESULTSDIR="./results"
#using existing checkpoint_phase1 dir
export CHECKPOINTDIR_PHASE1="/raid/datasets/bert/checkpoints/checkpoint_phase1"
export UNITTESTDIR="/lustre/fsw/mlperf/mlperft-bert/unit_test"
