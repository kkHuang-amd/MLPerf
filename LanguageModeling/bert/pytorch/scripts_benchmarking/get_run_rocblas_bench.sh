#!/bin/bash

function check_bin {
  bin=$1
  path=$2
  if [[ $path == "" ]]; then
    echo "Error: $bin not found.  Please install $bin and set $bin binary path in PATH."
    exit 1
  fi
}

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <ROCBLAS_LAYER=2 log> <output file.csv>"
  exit
fi

log=$1
outfile=$2

rocprof=`which rocprof`
check_bin rocprof $rocprof

rocblas_bench=`which rocblas-bench`
check_bin rocblas-bench $rocblas_bench

rocprof_out=`mktemp /tmp/rocprof_out_XXXX.csv`
rocblas_cmds=`mktemp /tmp/rocblas_cmds_XXXX`
rocblas_out=`mktemp /tmp/rocblas_out_XXXX`
output=`mktemp /tmp/run_rochblas_bench_out_XXXX`

num_rocblas_iters=10

#grep -o "rocblas-bench" $log | sort | uniq -c | sort -nr -k 1 > $rocblas_cmds
grep -o "./rocblas.*" $log | sort | uniq -c | sort -nr -k 1 > $rocblas_cmds

sed -i "s+./rocblas-bench+${rocblas_bench}+g" $rocblas_cmds

count=`cat $rocblas_cmds | wc -l`
if [[ $count -eq 0 ]]; then
  echo "Error: rocblas-bench commands not found"
  exit
fi

echo "Found $count distinct unique rocblas-bench commands"
rm -f $output

while read line; do
  cmd=`echo $line | cut -f 1 -d ' ' --complement`
  count=`echo $line | cut -f 1 -d ' '`
  echo "Running $cmd"
  $rocprof -o $rocprof_out --obj-tracking on --timestamp on \
    $cmd -i $num_rocblas_iters > $rocblas_out

  # skip two cold runs
  num_skip_profs=`cat $rocprof_out | wc -l | \
    awk -v iters=$num_rocblas_iters -v cold_run=2 \
    '{ print ((($1 - 1) / (iters + cold_run)) * cold_run) }'`

  # there can be multiple kernel per rocblas-bench command
  kernel_info=`cat $rocprof_out | awk -F ',' -v num_skip_profs=$num_skip_profs '
    NR == 1 {
      for (i = 1; i <=NF; i++) {
        if ($i == "BeginNs") { begin_idx = i; }
        else if ($i == "EndNs") { end_idx = i; }
        else if ($i == "KernelName") { kernel_idx = i; }
      }
      line = 0;
    }
    NR > 1 {
      if (line >= num_skip_profs) {
        sum[$(kernel_idx)] += $(end_idx) - $(begin_idx);
        count[$(kernel_idx)]++;
      }
      line++;
    }
    END {
      total = 0;
      count_kernels = 0;
      kernel_name = "";
      for (kernel in sum) {
        total += sum[kernel];
        count_kernels++;
        if (count_kernels == 1) {
          kernel_name = kernel
        }
        else {
          kernel_name = kernel_name";"kernel
        }
      }
      printf("%.2f,%d,%s", total/count[kernel]/1000, count_kernels, kernel_name)
    }'`

  kernel_time=`echo $kernel_info | cut -f 1 -d ','`
  kernel_info=`echo $kernel_info | cut -f 1 -d ',' --complement`

  grep -A 1 "rocblas-Gflops" $rocblas_out | \
    awk -v kernel_time=$kernel_time -v count=$count -v kernel_info=$kernel_info '
    NR == 1 { printf("%s,kernel-us,count,num-kernels,kernel-name\n", $0); }
    NR == 2 { printf("%s,%.2f,%d,%s\n", $0, kernel_time, count, kernel_info); }' >> $output
done < $rocblas_cmds

echo "Creating pretty output"
cat $output | awk '{
  if ($0 ~ /rocblas-Gflops/) {
    key = $0;
    if (arr_len[key] == "") { arr_len[key] = 0; } }
    else {
      perf[key, arr_len[key]] = $0;
      arr_len[key]++;
    }
  }
  END {
    for (key in arr_len) {
      print key;
      for (i = 0; i < arr_len[key]; i++) {
        print perf[key, i]
      }
    }
  }' > $outfile

rm $rocprof_out
rm $rocblas_cmds
rm $rocblas_out
rm $output

echo "Output: $outfile"

