#!/bin/bash 

./create_plots.sh 1 100 1 -o
cp times.log OpenMP.log
cp Times.jpeg OpenMP.jpeg

./create_plots.sh 1 100 1 -c
cp times.log cuda.log
cp Times.jpeg cuda.jpeg

