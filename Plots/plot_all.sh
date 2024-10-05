#!/bin/bash 

make clean
make

#Delete old data
./create_shm_directory.sh

./time_simulations_secuencial 1 3001 100 
cp times.log Secuencial.log
cp Times.jpeg Secuencial.jpeg

#Delete old data
./create_shm_directory.sh

./time_simulations_secuencial_vectorial 1 5001 100 
cp times.log Secuencial_vectorial.log
cp Times.jpeg Secuencial_vectorial.jpeg

#Delete old data
./create_shm_directory.sh

./time_simulations_OpenMP 1 5001 100 
cp times.log OpenMP.log
cp Times.jpeg OpenMP.jpeg

#Delete old data
./create_shm_directory.sh

./time_simulations_OpenMP_vectorial 1 5001 100 
cp times.log OpenMP_vectorial.log
cp Times.jpeg OpenMP_vectorial.jpeg

./create_shm_directory.sh

./time_simulations_cuda 1 10001 100 
cp times.log Cuda.log
cp Times.jpeg Cuda.jpeg

./create_shm_directory.sh

./time_treecode.sh 1 25001 100
cp times.log Treecode.log

