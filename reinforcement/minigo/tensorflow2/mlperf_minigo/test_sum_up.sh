


awk '{print $(NF-1)}' /tmp/training > /tmp/t
awk '{s+=$1} END {print s}' /tmp/t

#while read p; do
#  echo "$p"
#  echo $p | sed 's/^*: //' | sed 's/ seconds$//'
#  $p | sed 's/^config_//' | sed 's/\.sh$//'
#done </tmp/training
