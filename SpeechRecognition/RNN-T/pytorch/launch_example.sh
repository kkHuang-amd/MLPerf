#!/usr/bin/env bash

set -e -x

source config_MI250.sh

CONT=mlperf/rnn_speech_recognition DATADIR=/global/scratch/mlperf_datasets/rnnt/ LOGDIR=log METADATA_DIR=/global/scratch/mlperf_datasets/rnnt/tokenized/ SENTENCEPIECES_DIR=/global/scratch/mlperf_datasets/rnnt/sentencepieces/ bash ./run_with_docker.sh
