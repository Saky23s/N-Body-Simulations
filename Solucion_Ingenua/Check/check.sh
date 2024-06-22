#!/bin/bash 

RED='\033[41m'
GREEN='\033[42m'
YELLOW='\033[43m'
NC='\033[m'
CLEAR_LINE='\033[K'

# Check al least the tree must have parameters are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <start> <end> <step>"
    exit 0
fi


#Generate fresh executables
make clean *>/dev/null
make all  >/dev/null

cd ../

make clean *>/dev/null
make all  >/dev/null

cd Check/

cuda=false
Open_MP=false
secuential_optimizada=false

for arg in "$@" 
do
    case $arg in
        -all)
            cuda=true
            Open_MP=true
            ;;
        -co)
            cuda=true
            Open_MP=true
            ;;
        -oc)
            cuda=true
            Open_MP=true
            ;;
        -o) 
            
            Open_MP=true
            ;;
        -c) 
            cuda=true
            ;;
        -so) 
            secuential_optimizada=true
            ;;
        *)
            ;;
    esac
done

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

  if [ $cuda == true -a $Open_MP == true ]; then
    #Delete old data
    if [ -d ../Graphics/data ]; then
      rm -f ../Graphics/data/*.csv 2>/dev/null
      rm -f ../Graphics/data/*.bin 2>/dev/null
    else
      mkdir ../Graphics/data 2>/dev/null
    fi

    ./simulation_OpenMP 1 ../Starting_Configurations/bin_files/random.bin >/dev/null

    cp ../Graphics/data/1.bin Check/OpenMP.bin >/dev/null

    ./simulation_cuda 1 ../Starting_Configurations/bin_files/random.bin >/dev/null

    cp ../Graphics/data/1.bin Check/cuda.bin >/dev/null

    cd Check/
    ./compare $1 secuential.bin cuda.bin >/dev/null
    result_cuda=$?
    ./compare $1 secuential.bin OpenMP.bin >/dev/null
    result_OpenMP=$?
    if [ $result_cuda == 1 -a $result_OpenMP == 1 ];
    then
      echo -e "${CLEAR_LINE}${GREEN} PASSED ${NC} $n bodies"
    elif [ $result_cuda != 1 -a $result_OpenMP == 1 ]; then
      echo -e "${CLEAR_LINE}${RED} CUDA FAILED ${NC} $n bodies"
      exit 0
    elif [ $result_cuda == 1 -a $result_OpenMP != 1 ]; then
      echo -e "${CLEAR_LINE}${RED} OPEN_MP FAILED ${NC} $n bodies"
      exit 0
    else
      echo -e "${CLEAR_LINE}${RED} OPEN_MP & CUDA FAILED ${NC} $n bodies"
      exit 0
    fi
  elif [ $cuda == false -a $Open_MP == true ]; then
    #Delete old data
    if [ -d ../Graphics/data ]; then
      rm -f ../Graphics/data/*.csv 2>/dev/null
      rm -f ../Graphics/data/*.bin 2>/dev/null
    else
      mkdir ../Graphics/data 2>/dev/null
    fi

    ./simulation_OpenMP 1 ../Starting_Configurations/bin_files/random.bin >/dev/null

    cp ../Graphics/data/1.bin Check/OpenMP.bin >/dev/null

    cd Check/
    ./compare $1 secuential.bin OpenMP.bin >/dev/null

    if [ $? == 1 ];
    then
      echo -e "${CLEAR_LINE}${GREEN} PASSED ${NC} $n bodies"
    else
      echo -e "${CLEAR_LINE}${RED} FAILED ${NC} $n bodies"
      exit 0
    fi
  elif [ $cuda == false -a $Open_MP == false -a $secuential_optimizada == true ]; then
    #Delete old data
    if [ -d ../Graphics/data ]; then
      rm -f ../Graphics/data/*.csv 2>/dev/null
      rm -f ../Graphics/data/*.bin 2>/dev/null
    else
      mkdir ../Graphics/data 2>/dev/null
    fi

    ./simulacion_optimizada_secuencial 1 ../Starting_Configurations/bin_files/random.bin >/dev/null

    cp ../Graphics/data/1.bin Check/optimizada_secuential.bin >/dev/null

    cd Check/
    ./compare $1 secuential.bin optimizada_secuential.bin >/dev/null

    if [ $? == 1 ];
    then
      echo -e "${CLEAR_LINE}${GREEN} PASSED ${NC} $n bodies"
    else
      echo -e "${CLEAR_LINE}${RED} FAILED ${NC} $n bodies"
      exit 0
    fi
  else 
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
      echo -e "${CLEAR_LINE}${RED} FAILED ${NC} $n bodies"
      exit 0
    fi
  fi
done

exit 1



