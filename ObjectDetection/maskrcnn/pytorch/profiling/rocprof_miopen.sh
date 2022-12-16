intermediate_file=/tmp/iii
storage_file=/tmp/sss.csv

echo "" > $storage_file

cat $1 | grep -iE "miopendriver convfp16"|sed s/.*[\]]// |cut -c 2-|tr -d '\r' > $intermediate_file
#cat $1 | grep -iE "miopendriver convfp16"|cut -d " " -f 4- |tr -d '\r' > $intermediate_file

IFS=$'\n'

counter=0

add_in_storage()
{
  echo "$1",0 >> $storage_file
}

increase_in_storage()
{
  #echo "debug script: " $1
  #echo "debug script: " $2
  string=$(echo "$1"|cut -d ' ' -f 3-)
  value=$(($(($2))+1))
  #echo "debug script: value = " $value
  #echo "debug script: string = " $string
  ori="$string","$2"
  mod="$string","$value"
  #echo "increase: ori = " $ori
  #echo "increase: mod = " $mod
  sed -i "s/${ori}/${mod}/g" $storage_file
}

find_in_storage()
{
  for l_s in `cat $storage_file`
  do
    #echo "testing ( in )" "$1"
    #echo "testing (loop)" "$i"
    the_cmd=$(echo "$l_s"|cut -d ',' -f 1)
    the_cnt=$(echo "$l_s"|cut -d ',' -f 2)
    #echo "the cmd =" "$the_cmd"
    if [ "$1" = "$the_cmd" ]; then
      echo "Found " "$the_cmd" "."
      increase_in_storage $the_cmd $the_cnt
      return 1
    fi
  done

  echo "$1" " not found."
  return 0
}

for l_i in `cat $intermediate_file`
do

  find_in_storage "$l_i"
  ret=$?
  if [ $ret == 1 ]; then
    continue
  fi

  var=`echo "$l_i"|sed -r 's/^.{6}//'`
  #eval $var
  add_in_storage "$l_i"

  counter=$(($counter+1))
  echo $counter
done

