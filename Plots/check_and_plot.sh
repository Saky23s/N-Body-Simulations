#!/bin/bash 

./create_plots.sh 1 100001 100 -c
cp times.log ./Results/logs/Cuda.log
cp times.log ./Results/logs/Cuda.log
cd ../Metodo_Directo/Check

./check.sh 1 100001 1000 -c

cd ../../Plots