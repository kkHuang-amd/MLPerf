#!/bin/bash

if [[ $# -lt 1 ]]; then
	echo "usage: $0 <log>"
  fi

  log=$1

  grep "rocblas-bench" $log  | sort | uniq -c | sort -nr
