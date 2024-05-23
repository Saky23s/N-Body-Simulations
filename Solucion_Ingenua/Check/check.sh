#!/bin/bash 

RED='\033[41m'
GREEN='\033[42m'
YELLOW='\033[43m'
NC='\033[m'
CLEAR_LINE='\033[K'

# Check if exactly two parameters are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <start> <end> <step>"
    exit 1
fi


#Generate fresh executables
make clean *>/dev/null
make all  >/dev/null

cd ../

make clean *>/dev/null
make all  >/dev/null

cd Check/

for (( n=$1; n<=$2; n+=$3 ))
do 
  echo -ne "${YELLOW} TESTING ${NC} $n bodies...\r"
  
  #Delete old data
  if [ -d ../../Graphics/data ]; then
    rm -f ../../Graphics/data/*.csv >/dev/null
    rm -f ../../Graphics/data/*.bin >/dev/null
  else
    mkdir ../../Graphics/data >/dev/null
  fi

  ./generate_random $n >/dev/null

  cd ../
  ./simulation_secuencial 1 ../Starting_Configurations/bin_files/random.bin >/dev/null
  cp ../Graphics/data/1.bin Check/secuential.bin >/dev/null

  #Delete old data
  if [ -d ../Graphics/data ]; then
    rm -f ../Graphics/data/*.csv 2>/dev/null
    rm -f ../Graphics/data/*.bin 2>/dev/null
  else
    mkdir ../Graphics/data 2>/dev/null
  fi

  ./simulation_cuda 1 ../Starting_Configurations/bin_files/random.bin >/dev/null

  cp ../Graphics/data/1.bin Check/cuda.bin >/dev/null

  cd Check/
  ./compare $1 secuential.bin cuda.bin >/dev/null

  if [ $? == 1 ];
  then
    echo -e "${CLEAR_LINE}${GREEN} PASSED ${NC} $n bodies"
  else
    echo -e "${CLEAR_LINE} ${RED} FAILED ${NC} $n bodies"
    exit
  fi
done


