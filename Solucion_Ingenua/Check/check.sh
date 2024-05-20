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
/bin/bash run_simulation_secuential.sh 1 ../Starting_Configurations/bin_files/random.bin

cp ../Graphics/data/1.bin Check/secuential.bin

/bin/bash run_simulation_cuda_OpenMP.sh 1 ../Starting_Configurations/bin_files/random.bin

cp ../Graphics/data/1.bin Check/cuda_OpenMP.bin

/bin/bash run_simulation_cuda.sh 1 ../Starting_Configurations/bin_files/random.bin

cp ../Graphics/data/1.bin Check/cuda.bin

cd Check/
rm -f compare
make compare  >/dev/null
./compare $1 secuential.bin cuda_OpenMP.bin
./compare $1 secuential.bin cuda.bin
