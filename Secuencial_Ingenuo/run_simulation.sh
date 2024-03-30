#!/bin/bash 

# Check if exactly two parameters are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <T> <Filepath>"
    exit 1
fi

#Delete old data
if [ -d ../Graphics/data ]; then
  rm -rf ../Graphics/data
fi
mkdir ../Graphics/data

#Create starting configuration data
cd ../Starting_Configurations
make >/dev/null
./graphic_starting_position $2

#Run simulation
cd ../Secuencial_Ingenuo/
make >/dev/null
./main $1 $2

#Run Graphics
cd ../Graphics/
cargo run
cd ../Secuencial_Ingenuo/
