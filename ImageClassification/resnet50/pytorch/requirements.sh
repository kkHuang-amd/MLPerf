### Simple script to install dependencies for ResNet50
apt-get update -y && apt-get install -y numactl && apt-get install hwloc

export MAX_JOBS=128
python3 -m pip install --no-cache-dir git+https://github.com/mlcommons/logging.git@2.0.0-rc4 --user
python3 -m pip install deepspeed --user
HOROVOD_WITHOUT_MPI=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_GPU=ROCM python3 -m pip install --user -v git+https://github.com/horovod/horovod.git@9d56c5a6d55e9619e100d19480a11952d116f4b3
python3 -m pip install -v --install-option="--cpp_ext" --install-option="--cuda_ext" 'git+https://github.com/ROCmSoftwarePlatform/apex.git' --user

pushd src/fused_lars
bash build.sh
popd
pushd src/utils
python3 setup.py install --user
popd
