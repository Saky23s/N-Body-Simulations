#!/bin/bash 

./create_plots.sh 4001 100 -o
cp times.log OpenMP.log
cp Times.jpeg OpenMP.jpeg

./create_plots.sh 4001 100 -c
cp times.log cuda.log
cp Times.jpeg cuda.jpeg

