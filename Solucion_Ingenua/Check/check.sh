#!/bin/bash 

# Check if exactly two parameters are provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <N>"
    exit 1
fi

#Delete old data
if [ -d ../../Graphics/data ]; then
  rm -f ../../Graphics/data/*.csv
  rm -f ../../Graphics/data/*.bin
else
  mkdir ../../Graphics/data
fi

#Generate a random starting position
rm -f generate_random
make generate_random  >/dev/null
./generate_random $1

cd ../
rm simulation_secuencial >/dev/null
make simulation_secuencial >/dev/null

./simulation_secuencial 1 ../Starting_Configurations/bin_files/random.bin

cp ../Graphics/data/1.bin Check/secuential.bin

#Delete old data
if [ -d ../Graphics/data ]; then
  rm -f ../Graphics/data/*.csv
  rm -f ../Graphics/data/*.bin
else
  mkdir ../Graphics/data
fi

rm simulation_cuda >/dev/null
make simulation_cuda >/dev/null
./simulation_cuda 1 ../Starting_Configurations/bin_files/random.bin

cp ../Graphics/data/1.bin Check/cuda.bin

cd Check/
rm -f compare
make compare  >/dev/null

./compare $1 secuential.bin cuda.bin
