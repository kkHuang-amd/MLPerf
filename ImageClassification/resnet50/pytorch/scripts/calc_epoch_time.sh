#!/bin/bash

in=$1
start=`grep -inr "run_start" $in | head -n 1 | grep MLLOG| sed -e 's/.*time_ms": \([0-9]*\).*/\1/g' | grep -e "\S"`
stop=`grep -inr "run_stop" $in | head -n 1 | grep MLLOG| sed -e 's/.*time_ms": \([0-9]*\).*/\1/g' | grep -e "\S"`
ep_stop=(`grep -inr "epoch_stop" $in | grep MLLOG| sed -e 's/.*time_ms": \([0-9]*\).*/\1/g' | grep -e "\S"`)
ep_start=(`grep -inr "epoch_start" $in | grep MLLOG| sed -e 's/.*time_ms": \([0-9]*\).*/\1/g' | grep -e "\S"`)

prev=$start
echo "epoch,start,end,time(ms),betwEp(ms)"
for i in `seq 0 $((${#ep_stop[@]}-1))`; do
    cur=${ep_stop[$i]}
    ep_s=${ep_start[$i]}
    echo "${i},${prev},${cur},$(((cur - ep_s))),$((ep_s-prev))"
    prev=$cur
done
if [[ $stop != "" ]]; then
    echo "E2E,$(((stop-start)/60))"
fi
