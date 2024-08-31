#!/bin/bash 

# Check al least the two must have parameters are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <Start_N> <Max_N> <Steps>"
    exit 1
fi

#Delete old data just for safety
if [ -d /dev/shm/data ]; then
  rm -f /dev/shm/data/*.csv
  rm -f /dev/shm/data/*.bin
else
  mkdir /dev/shm/data
fi

>'times.log'
for (( n=$1; n<=$2; n+=$3 ))
do 
    printf $n >> times.log

    cd ../Starting_Configurations
    make clean *>/dev/null
    make all >/dev/null
    ./plummer_configuration $n

    cd ../Tree_Code/Modified/
    make clean *>/dev/null
    make all >/dev/null

    ./time_treecode_secuencial 100.0 "../../Starting_Configurations/bin_files/plummer.bin" "../../Plots/times.log"
    cd ../../Plots
done


#plot results
gnuplot> load 'times.p'