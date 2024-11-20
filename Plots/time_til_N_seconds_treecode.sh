#!/bin/bash 

# Check al least the two must have parameters are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <Start_N> <Max_Time> <Steps>"
    exit 1
fi

#Delete old data just for safety
if [ -d /dev/shm/data ]; then
  rm -f /dev/shm/data/*.csv
  rm -f /dev/shm/data/*.bin
else
  mkdir /dev/shm/data
fi

max_time=$2

>'times.log'

for (( n=$1; n>0; n+=$3 ))
do 
    printf $n >> times.log

    if [ -d /dev/shm/data ]; then
      rm -f /dev/shm/data/*.csv >/dev/null
      rm -f /dev/shm/data/*.bin >/dev/null
    else
      mkdir /dev/shm/data >/dev/null
    fi

    cd ../Starting_Configurations
    make clean *>/dev/null
    make all >/dev/null
    ./plummer_configuration $n

    cd ../Tree_Code/Modified/
    make clean *>/dev/null
    make all >/dev/null

    ./time_treecode_secuencial 100.0 "../../Starting_Configurations/bin_files/plummer.bin" "../../Plots/times.log"
    cd ../../Plots
    t=$(tail -n 1 "times.log"  | awk '{print $2}')

    # if the time is bigger than max_time finish executing
    if (( $(echo "$t > $max_time" | bc -l) )); then
      break
done


#plot results
gnuplot> load 'times.p'