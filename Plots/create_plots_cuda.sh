#!/bin/bash 

# Check if exactly two parameters are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <Max_N> <Steps>"
    exit 1
fi

#Delete old data just for safety
if [ -d ../Graphics/data ]; then
  rm -f ../Graphics/data/*.csv
  rm -f ../Graphics/data/*.bin
else
  mkdir ../Graphics/data
fi

#Run time_simulation
rm -f time_simulations_cuda
make time_simulations_cuda >/dev/null
./time_simulations_cuda $1 $2

#plot results
gnuplot> load 'times.p'
