#!/bin/bash 

# Check if exactly two parameters are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <T> <Filepath>"
    exit 1
fi

#Delete old data
if [ -d ../../Graphics/data ]; then
  rm -f ../../Graphics/data/*.csv
  rm -f ../../Graphics/data/*.bin
else
  mkdir ../../Graphics/data
fi


#Create starting configuration data
filepath="$2"
cd ../../Starting_Configurations
make >/dev/null
./graphic_starting_position ${filepath:3}  

#Run simulation
cd ../Secuencial_Ingenuo/Demo_Python/
python3 Secuencial_Runge_Kutta.py $1 $2

#Run Graphics
cd ../../Graphics/
cargo run
cd ../Secuencial_Ingenuo/Demo_Python
