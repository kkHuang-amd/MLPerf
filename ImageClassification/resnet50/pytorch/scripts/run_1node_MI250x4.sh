#!/bin/bash
echo "Clear page cache"
sync && /sbin/sysctl vm.drop_caches=3

CURRENTDATE=`date +"%Y-%m-%d-%T"`

# Mention the number of gpus
gpus_per_node=8
batch_size=512
num_epochs=37
format="nhwc"
OUTDIR="./"
wd="0.00005"
base_lr="13.15"
lr_sched="polynomial"
warmup=2
data_dir=${DATASET_PATH:-"/datasets/imagenet/"}
nnodes=${SLURM_NTASKS:-1}
nodeid=${SLURM_NODEID:-0}
gbs=$((batch_size*gpus_per_node*nnodes))
tag=gbs${gbs}_${nnodes}x${gpus_per_node}GPUs_epoch${num_epochs}_${format}_lr${base_lr}_warmup${warmup}_lr-sched-${lr_sched}_node${nodeid}
submission_platform="MI200system"

MASTER_ADDR=${MASTER_ADDR:-`scontrol show hostnames | head -n 1`}
DDP=${1:-"torchDDP"}

nnuma_nodes=`lscpu | grep "NUMA node(s)" | sed -e "s/.*:\s*//g"`
nnuma_nodes=$((nnuma_nodes < gpus_per_node ? nnuma_nodes : gpus_per_node))
nsockets=`lscpu | grep "Socket(s)" | sed -e "s/.*:\s*//g"`
ncores_per_socket=`lscpu | grep "Core(s) per socket:" | sed -e "s/.*:\s*//g"`
ncores=$((nsockets*ncores_per_socket))
nDataWorkers=$((ncores/gpus_per_node))

CMD=""
if [[ ${DDP} == "horovod" ]]; then
    CMD="horovodrun -np ${gpus_per_node} python3 "
else
    CMD="python3 -u -m mlperf_utils.bind_launch --no_membind --nnuma_nodes $nnuma_nodes --nsockets_per_node ${nsockets} --ncores_per_socket ${ncores_per_socket} --node_rank ${nodeid} --nnodes ${nnodes} --master_port 23456 --nproc_per_node ${gpus_per_node} "
    if [[ ${MASTER_ADDR} != "" ]]; then
        CMD+=" --master_addr ${MASTER_ADDR}"
    fi
fi

CMD+=" ./src/main.py --num-nodes ${nnodes} --amp --dynamic-loss-scale --lr-schedule ${lr_sched} --num-gpus $gpus_per_node \
  --mom 0.9 --wd ${wd} --lr ${base_lr} --warmup ${warmup} --epochs ${num_epochs} --use-lars -b $batch_size \
  --eval-offset 2 --get-logs --submission-platform $submission_platform \
  --no-checkpoints --raport-file raport.json -j${nDataWorkers} -p 100 --arch resnet50 --data $data_dir"

if [[ $format == "nhwc" ]]; then
    CMD+=" --nhwc"
    export PYTORCH_MIOPEN_SUGGEST_NHWC=1
fi

if [[ ${DDP} == "deepspeed" ]]; then
    CMD+=" --deepspeed --deepspeed_config ds_config.json"
elif [[ ${DDP} == "horovod" ]]; then
    CMD+=" --horovod"
fi

LOG=${OUTDIR}/r_${DDP}_${tag}.${CURRENTDATE}.log

env 2>&1 | tee -a ${LOG}
echo ${CMD} | tee -a ${LOG}
SECONDS=0

export MIOPEN_USER_DB_PATH=~/miopen-db-luise_node${nodeid}_${CURRENTDATE}
rm $MIOPEN_USER_DB_PATH -r
${CMD} 2>&1 | tee -a ${LOG}

duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed." | tee -a ${LOG}

rm $MIOPEN_USER_DB_PATH -r
