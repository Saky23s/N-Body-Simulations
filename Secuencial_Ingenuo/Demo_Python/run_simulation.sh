#!/bin/bash 

if [ -d data ]; then
  rm -rf data
fi
mkdir data
python3 Secuencial_Runge_Kutta.py $1 $2
cd ../../Graphics/
cargo run
cd ../Secuencial_Ingenuo/Demo_Python
