#!/bin/bash

function gen_seq() {
  printf "%.s${1}," `seq ${2}` | sed 's/,$//g'
}

usage="$0 [options]"

opt=""
fp=16
dataset_dir=/dockerx/data/dlrm/binary_dataset
gpus_per_node=8
test_batch_size=131072
train_batch_size=32768
config_file=none
opt+="--cache_eval_data"
use_emb_comp=True

while true; do
  case "$1" in
    -h | --help ) help=True; shift ;;
    --fp16 ) fp=16; shift ;;
    --fp32 ) fp=32; shift ;;
    --dataset-dir ) dataset_dir=$2; shift 2 ;;
    --gpus-per-node ) gpus_per_node=$2; shift 2 ;;
    --seed ) opt+=" --seed $2"; shift 2 ;;
    --disable-apex-mlp ) opt+=" --nouse_apex_mlp"; shift ;;
    --mlperf-log ) mlperf_log=$2; shift 2 ;;
    --use-embedding-compression ) use_emb_comp=True; shift ;;
    --train-batch-size ) train_batch_size=$2; shift 2 ;;
    --config ) config_file=$2; shift 2 ;;
    --debug )  opt+=" --debug"; shift ;;
    --use-wmma-interaction ) opt+=" --use_wmma_interaction"; shift ;;
    * )
      if [[ $1 == *"-"* ]]; then not_found=True; help=True; fi; break ;;
  esac
done

if [[ $help == "True" ]]; then
  if [[ $not_found == "True" ]]; then
    echo "Error: Option $1 does not exist"
  fi
  echo $usage
  echo "Supported options:"
  echo "-h | --help                     print this help message"
  echo "--fp16                          use mixed precision training (FP16) (default)"
  echo "--fp32                          use FP32 training (FP32)"
  echo "--dataset-dir DIR               set dataset directory"
  echo "--gpus-per-node GPUS_PER_NODE   set number of GPUs per node (default: 8)"
  echo "--seed SEED                     fix random seed (default: None)"
  echo "--disable-apex-mlp              disable APEX MLP (use PyTorch MLP)"
  echo "--mlperf-log MLPERF_LOG         set MLPerf log file name"
  echo "--use-embedding-compression     use embedding compression optimization"
  echo "--train-batch_size              set train batch size (default: 32768)"
  echo "--config CONFIG                 set config file (preset configs are in run_scripts/config)"
  echo "--debug                         enable detailed printing"
  echo "--use-wmma-interaction          use WMMA for the dot interaction operation"
  exit 1
fi

if [[ $config_file == "default" ]]; then
  echo "Use default config"
else
  if [[ -f "${config_file}" ]]; then
    echo "Load config from $config_file"
    source $config_file
    opt+=" ${config_opt}"
  else
    echo "Config file $config_file does not exist."
    config_file=none
  fi
fi

if [[ $config_file == "none" ]]; then
  config_dir=`echo $0 | sed 's/dlrm_run_script.sh/config/g'`
  echo "Please specify the config to run by setting --config with one of the following options:"
  echo
  echo "  default"
  for f in `ls $config_dir`; do
    echo "  ${config_dir}/${f}"
  done
  echo
  echo "Or if you would like to create the config file, please look at one of preset configurations in ${config_dir}"
  exit 1
fi

export HIP_VISIBLE_DEVICES=`seq -s ',' 0 $((gpus_per_node - 1))`

if [[ $fp -eq 16 ]]; then
    opt+=" --fp16"
else
    opt+=" --nofp16"
fi

if [[ $gpus_per_node -eq 4 ]]; then
  if [[ $fp -eq 32 ]]; then
    train_batch_size=16384
    echo "WARNING: Adjust batch size to ${train_batch_size}. (The FP32 4-GPU run supports the largest batch size of ${train_batch_size}.)"
  fi
  test_batch_size=$train_batch_size
fi

if [[ $train_batch_size -eq 16384 ]]; then
  wsteps=6400
  dsteps=158750
  dsstep=125460
  lr=14
elif [[ $train_batch_size -eq 32768 ]]; then
  wsteps=8000
  dsteps=30000
  dsstep=70000
  if [[ $fp -eq 16 ]]; then
    lr=24
  else
    lr=28
  fi
elif [[ $train_batch_size -eq 65536 ]]; then
  # With these hyperparameters, ran it 10 times:
  # Converged in 0.95 epochs 7 times
  # Converged in 1 epochs 2 times
  # Did not converge 1 time
  wsteps=2750
  dsteps=15000
  dsstep=49315
  lr=24
fi

print_freq=$(( 1000 * 32768 / train_batch_size ))

if [[ $use_emb_comp == "True" ]]; then
  opt+=" --compress_embedding"
fi

# Clear page cache
echo "Clear page cache"
sudo sync && sudo /sbin/sysctl vm.drop_caches=3

if [[ $? -eq 0 ]]; then
  export MLPERF_CACHE_CLEAR=1
else
  export MLPERF_CACHE_CLEAR=0
fi

CURRENTDATE=`date +"%Y-%m-%d-%T" | sed -e "s/:/-/g"`
mlperf_log="dlrm-${CURRENTDATE}-${train_batch_size}-${wsteps}-${dsteps}-${dsstep}-${lr}-${fp}.log"
export MLPERF_LOG_FILE=$mlperf_log

python3 -u -m mlperf_utils.bind_launch \
  --nproc_per_node $gpus_per_node \
  --auto_binding \
  scripts/dist_train.py \
  --epochs 1 \
  --print_freq $print_freq \
  --batch_size $train_batch_size \
  --use_alltoall_base \
  --dataset $dataset_dir \
  --dataset_type dist \
  --lr $lr \
  --warmup_steps $wsteps \
  --decay_steps $dsteps \
  --decay_start_step $dsstep \
  --decay_end_lr 0 \
  --model_config dlrm/config/mlperf_40m.limit.json \
  --test_batch_size $test_batch_size \
  $opt

echo "MLPerf log: $mlperf_log"
