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
        
        -v)
            TYPE="secuencial_vectorial"
            ;;
        -vo)
            TYPE="OpenMP_vectorial"
            ;;
        -ov)
            TYPE="OpenMP_vectorial"
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

./time_simulations_${TYPE} $1 $2 $3

#plot results
gnuplot> load 'times.p'
