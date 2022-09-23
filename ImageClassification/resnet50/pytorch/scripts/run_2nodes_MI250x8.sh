#!/bin/bash
echo "Clear page cache"
sync && /sbin/sysctl vm.drop_caches=3

CURRENTDATE=`date +"%Y-%m-%d-%T"`

# Mention the number of gpus
gpus_per_node=8
batch_size=204
num_epochs=42
format="nhwc"
OUTDIR="./"
wd="0.0002"
base_lr="10.5"
lr_sched="polynomial"
warmup=2
data_dir=${DATASET_PATH:-"/global/scratch/mlperf_datasets/imagenet/"}
nnodes=${SLURM_NTASKS:-1}
nodeid=${SLURM_NODEID:-0}
gbs=$((batch_size*gpus_per_node*nnodes))
tag=gbs${gbs}_${gpus_per_node}GPUs_epoch${num_epochs}_${format}_lr${base_lr}_warmup${warmup}_lr-sched-${lr_sched}_node${nodeid}_$6
submission_platform="MI200system"

MASTER_ADDR=${1:-`echo $SLURM_NODELIST | sed -e 's/\[\([0-9]\).*\]/\1/g' | sed -e 's/,.*//g'`}

CMD="--amp --dynamic-loss-scale --lr-schedule ${lr_sched} --num-gpus $gpus_per_node \
  --mom 0.9 --wd ${wd} --lr ${base_lr} --warmup ${warmup} --epochs ${num_epochs} --use-lars -b $batch_size \
  --eval-offset 2 --get-logs --submission-platform $submission_platform \
  --no-checkpoints --raport-file raport.json -j16 -p 100 --arch resnet50 --data $data_dir"

if [[ $format == "nhwc" ]]; then
    CMD+=" --nhwc"
    export PYTORCH_MIOPEN_SUGGEST_NHWC=1
fi

nnuma_nodes=`lscpu | grep "NUMA node(s)" | sed -e "s/.*:\s*//g"`
nnuma_nodes=$((nnuma_nodes < gpus_per_node ? nnuma_nodes : gpus_per_node))
nsockets=`lscpu | grep "Socket(s)" | sed -e "s/.*:\s*//g"`
ncores_per_socket=`lscpu | grep "Core(s) per socket:" | sed -e "s/.*:\s*//g"`

CMD="python3 -u -m mlperf_utils.bind_launch --nnuma_nodes $nnuma_nodes --nsockets_per_node ${nsockets} --ncores_per_socket ${ncores_per_socket} --node_rank ${nodeid} --nnodes ${nnodes} --master_addr ${MASTER_ADDR} --master_port 23456 --nproc_per_node ${gpus_per_node} ./src/main.py --num-nodes ${nnodes} ${CMD}"

LOG=${OUTDIR}/r_torchDistMP_${tag}.${CURRENTDATE}.log

echo ${CMD} | tee -a ${LOG}
SECONDS=0

export MIOPEN_USER_DB_PATH=~/miopen-db-luise_node${nodeid}_${CURRENTDATE}
rm $MIOPEN_USER_DB_PATH -r
${CMD} 2>&1 | tee -a ${LOG}

duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed." | tee -a ${LOG}

rm $MIOPEN_USER_DB_PATH -r
