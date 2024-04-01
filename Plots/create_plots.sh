#!/bin/bash 

# Check if exactly two parameters are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <Max_N> <Steps>"
    exit 1
fi

#Run time_simulation
make >/dev/null
./time_simulations $1 $2

#plot results
gnuplot load 'times.p'