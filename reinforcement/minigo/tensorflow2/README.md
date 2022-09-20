# 1. Problem

This task benchmarks reinforcement learning for the 19x19 version of the boardgame Go.
The model plays games against itself and uses these games to improve play.


# 2. Directions
### Steps to launch training
To setup the environment using amd-docker you can use the commands below.


### Build docker and prepare dataset
```
    # go to the relative folder in your local
    cd ~MLPerf/reinforcement/minigo/tensorflow2 
    
    # Build a docker using Dockerfile in this directory
    docker build -t {YOUR TAG} .
    
    # Download dataset, needs gsutil.
    # Download & extract bootstrap checkpoint.
    gsutil cp gs://minigo-pub/ml_perf/0.7/checkpoint.tar.gz .
    tar xfz checkpoint.tar.gz -C ml_perf/

    # Download and freeze the target model.
    mkdir -p ml_perf/target/
    gsutil cp gs://minigo-pub/ml_perf/0.7/target.* ml_perf/target/

    # run docker
    docker run -it --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size=64G {YOUR TAG} 
    cd minigo
    
    # untar checkpoint.tar.gz
    tar xfz /your_path/to_checkpoint/checkpoint.tar.gz -C ml_perf/ 
    
    # put target files
    mkdir -p ml_perf/target/ 
    cp //your_path/to_checkpoint/target.* ml_perf/target/ 

    # comment out L331 in dual_net.py before running freeze_graph.
    # L331 is: optimizer = hvd.DistributedOptimizer(optimizer)
    # Horovod is initialized via train_loop.py and isn't needed for this step.
    HIP_VISIBLE_DEVICES=0 python3 freeze_graph.py --flagfile=ml_perf/flags/19/architecture.flags  --model_path=ml_perf/target/target 
    mv ml_perf/target/target.minigo ml_perf/target/target.minigo.tf

    # uncomment L331 in dual_net.py.
    # copy dataset to /data that is mapped to <path/to/store/checkpoint> outside of docker.
    # Needed because run_and_time.sh uses the following paths to load checkpoint
    # CHECKPOINT_DIR="/data/mlperf07"
    # TARGET_PATH="/data/target/target.minigo.tf"
    cp -a ml_perf/target /data/
    cp -a ml_perf/checkpoints/mlperf07 /data/
    
    # If you have to run with some specific number of  GPUs, modify   NUM_GPUS_TRAIN in   configs/config_MI100_8gpus.sh.  Both 4 and 8 work.
    export LD_LIBRARY_PATH=/opt/rocm/lib/:$LD_LIBRARY_PATH  
    DGXSYSTEM="mi100_8gpus" SLURM_NTASKS_PER_NODE=17 mpirun --allow-run-as-root -np 17 ./run_and_time.sh |& tee minigov1-8gpus-2ppg-convergence.log 
    # exit docker
```

