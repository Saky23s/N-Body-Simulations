#!/bin/bash 

./create_plots.sh 1 350 100 -v -t
cp times.log ./Results/logs/supercomputadora/Secuencial.log
cp Times.jpeg ./Results/img/supercomputadora/Secuencial.jpeg

./create_plots.sh 1 350 100 -ov -t
cp times.log ./Results/logs/supercomputadora/OpenMP.log
cp Times.jpeg ./Results/img/supercomputadora/OpenMP.jpeg

./create_plots.sh 1 350 100 -c -t
cp times.log ./Results/logs/supercomputadora/Cuda.log
cp Times.jpeg ./Results/img/supercomputadora/Cuda.jpeg

./create_plots.sh 1 350 100 -c2 -t
cp times.log ./Results/logs/supercomputadora/Cuda_V2.log
cp Times.jpeg ./Results/img/supercomputadora/Cuda_V2.jpeg

./time_til_N_seconds_treecode.sh 100 350 100 
cp times.log ./Results/logs/supercomputadora/TreeCode.log
cp Times.jpeg ./Results/img/supercomputadora/TreeCode.jpeg
