#!/bin/bash 

# Check if exactly two parameters are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <T> <Filepath>"
    exit 1
fi

#Delete old data
if [ -d ../Graphics/data ]; then
  rm -f ../Graphics/data/*.csv
  rm -f ../Graphics/data/*.bin
else
  mkdir ../Graphics/data
fi

#Create starting configuration data
cd ../Starting_Configurations
make >/dev/null
./graphic_starting_position $2

#Run simulation
cd ../Solucion_Ingenua/
rm simulation_cuda >/dev/null
make simulation_cuda >/dev/null
./simulation_cuda $1 $2

#Run Graphics
cd ../Graphics/
cargo run
cd ../Solucion_Ingenua/
