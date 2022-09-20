#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Set and create output directories for the run and send us to the right place.
RESULT_DIR=$WORKSPACE_DIR/results
LOG_DIR=/log
LOGGER_FILE=$LOG_DIR/rnnt_run.log

mkdir -p $RESULT_DIR

# Train benchmark
# bash run_rnnt_ootb_train.sh -l $LOG_DIR
(set -x; bash scripts/train.sh /data/LibriSpeech configs/baseline_v3-1023sp.yaml ${RESULT_DIR} ${LOGGER_FILE} 2>&1)
python3 scripts/fb5logging/result_summarizer.py -f $LOG_DIR

# DLM will match with the grep statement if we print it out.
set +x
sed -n -e 's/^.*RNN-T .*train \S* \(\S* \S*\) .*/performance: \1/p' $LOG_DIR/summary.txt
# remove summary file so not pollute other runs.
rm $LOG_DIR/summary.txt
