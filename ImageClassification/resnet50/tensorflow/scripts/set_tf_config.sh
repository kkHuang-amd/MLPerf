#!/bin/bash -l

TF_PORT=${TF_PORT:-12321}

export WORKERS_LIST=$( \
  scontrol show hostnames | \
  sed "s/$/:${TF_PORT}\"/g" | \
  sed 's/^/"/g' | \
  paste -s -d ',' \
  )

export TF_CONFIG="{\"cluster\": {\"worker\": [${WORKERS_LIST}]}, \"task\": {\"type\": \"worker\", \"index\": $SLURM_NODEID} }"

