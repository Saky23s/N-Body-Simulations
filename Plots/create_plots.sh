#!/bin/bash 

# Check al least the two must have parameters are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <Start_N> <Max_N> <Steps>"
    exit 1
fi

#Generate fresh executables
make clean *>/dev/null
make all  >/dev/null

TYPE="secuencial";
TIL_N=0
for arg in "$@" 
do
    case $arg in
        -s) 
            TYPE="secuencial"
            ;;
        -o) 
            TYPE="OpenMP"
            ;;
        -c)
            TYPE="cuda"
            ;;
        -c2)
            TYPE="cuda_V2"
            ;;
        -v)
            TYPE="secuencial_vectorial"
            ;;
        -vo)
            TYPE="OpenMP_vectorial"
            ;;
        -ov)
            TYPE="OpenMP_vectorial"
            ;;
        -t)
            TIL_N=1
            ;;
        *)
            ;;
    esac
done

#Delete old data just for safety
if [ -d /dev/shm/data ]; then
  rm -f /dev/shm/data/*.csv
  rm -f /dev/shm/data/*.bin
else
  mkdir /dev/shm/data
fi

if [ $TIL_N == 1 ]; then
    ./time_til_N_seconds_${TYPE} $1 $2 $3
else
    ./time_simulations_${TYPE} $1 $2 $3
fi

#plot results
gnuplot> load 'times.p'
