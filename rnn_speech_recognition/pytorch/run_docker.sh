#!/usr/bin/env bash

IMAGE_TAG=rnnt_train
# docker build --network=host -t $IMAGE_TAG --pull -f pytorch_rocm.Dockerfile .

CONTAINER_NAME=container_$IMAGE_TAG
docker rm -f $CONTAINER_NAME

mkdir -p log
LIBRISPEECH=${LIBRISPEECH:-/global/scratch/mlperf_datasets/LibriSpeech/}
SENTENCEPIECES=${SENTENCEPIECES:-/global/scratch/mlperf_datasets/sentencepieces/}

echo "Dataset LibriSpeech path: ${LIBRISPEECH}"
echo "Dataset setencepieces path: ${SENTENCEPIECES}"

docker run -it -d \
	--network host -u root --device=/dev/kfd --group-add=video \
	--cap-add=SYS_PTRACE --cap-add SYS_ADMIN --device /dev/fuse \
	--security-opt seccomp=unconfined --ipc=host --device=/dev/dri \
	-v "$LIBRISPEECH:/data/LibriSpeech" \
	-v "$SENTENCEPIECES:/data/sentencepieces/" \
	-v "$(pwd)/log:/log" \
	-v "$(pwd):/workspace" \
	--name $CONTAINER_NAME $IMAGE_TAG

echo "Enter docker with: docker attach $CONTAINER_NAME"

# docker rm -f $CONTAINER_NAME
