#!/bin/bash 

#Delete old data
if [ -d /dev/shm/data ]; then
rm -f /dev/shm/data/*.csv >/dev/null
rm -f /dev/shm/data/*.bin >/dev/null
else
mkdir /dev/shm/data >/dev/null
fi

./time_simulations_secuencial_new 1 2001 100 
cp times.log Secuencial.log
cp Times.jpeg Secuencial.jpeg

#Delete old data
if [ -d /dev/shm/data ]; then
rm -f /dev/shm/data/*.csv >/dev/null
rm -f /dev/shm/data/*.bin >/dev/null
else
mkdir /dev/shm/data >/dev/null
fi

./time_simulations_openmp_new 1 5001 100 
cp times.log OpenMP.log
cp Times.jpeg OpenMP.jpeg


