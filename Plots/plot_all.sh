#!/bin/bash 

#./create_plots.sh 1 3001 100 -s
#cp times.log ./Results/logs/Secuencial.log
#cp Times.jpeg ./Results/img/Secuencial.jpeg

#./create_plots.sh 1 5001 100 -v
#cp times.log ./Results/logs/Secuencial_vectorial.log
#cp Times.jpeg ./Results/img/Secuencial_vectorial.jpeg

#./create_plots.sh 1 10001 100 -o
#cp times.log ./Results/logs/OpenMP.log
#cp Times.jpeg ./Results/img/OpenMP.jpeg
#
#./create_plots.sh 1 10001 100 -ov
#cp times.log ./Results/logs/OpenMP_vectorial.log
#cp Times.jpeg ./Results/img/OpenMP_vectorial.jpeg

./create_plots.sh 1 10001 100 -c
cp times.log ./Results/logs/Cuda.log
cp Times.jpeg ./Results/img/Cuda.jpeg

#./time_treecode.sh 100 25001 100 
#cp times.log ./Results/logs/TreeCode.log
#cp Times.jpeg ./Results/img/TreeCode.jpeg
