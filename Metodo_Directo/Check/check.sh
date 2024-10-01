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

cuda=false
Open_MP=false
vectorial=false
vectorial_openmp=false

for arg in "$@" 
do
    case $arg in
        -all)
            cuda=true
            Open_MP=true
            vectorial=true
            vectorial_openmp=true
            ;;
        -o) 
            
            Open_MP=true
            ;;
        -c) 
            cuda=true
            ;;
        -v)
            vectorial=true
            ;;
        -vo) 
            
            vectorial_openmp=true
            ;;
        -ov) 
            
            vectorial_openmp=true
            ;;
        *)
            ;;
    esac
done

#Check at least one was selected for testing
if [ $vectorial == false -a $cuda == false -a $Open_MP == false -a $vectorial_openmp == false ]; then 
  echo "ERROR: Select an implementation to check"
  exit 1
fi

for (( n=$1; n<=$2; n+=$3 ))
do 

  cd Check/
  echo -ne "${YELLOW} PREPARING TEST FOR $n BODIES ${NC}\r"
  
  #Delete old data
  if [ -d /dev/shm/data ]; then
    rm -f /dev/shm/data/*.csv >/dev/null
    rm -f /dev/shm/data/*.bin >/dev/null
  else
    mkdir /dev/shm/data >/dev/null
  fi

  ./generate_random $n >/dev/null

  cd ../
  ./simulation_secuencial 0.1 ../Starting_Configurations/bin_files/random.bin >/dev/null
  cp /dev/shm/data/2.bin Check/secuential.bin >/dev/null

  #Testing for vectorial
  if [ $vectorial == true ]; then 

    echo -ne "${YELLOW} TESTING FOR VECTORIAL FOR $n BODIES ${NC}\r"

    #Delete old data
    if [ -d /dev/shm/data ]; then
      rm -f /dev/shm/data/*.csv 2>/dev/null
      rm -f /dev/shm/data/*.bin 2>/dev/null
    else
      mkdir /dev/shm/data 2>/dev/null
    fi

    ./simulation_secuencial_vectorial 0.1 ../Starting_Configurations/bin_files/random.bin >/dev/null

    cp /dev/shm/data/2.bin Check/vectorial.bin >/dev/null

    cd Check/
    ./compare $1 secuential.bin vectorial.bin

    if [ $? == 1 ];
    then
      echo -e "${CLEAR_LINE}${GREEN} PASSED FOR VECTORIAL FOR $n BODIES ${NC}"
    else
      echo -e "${CLEAR_LINE}${RED} FAILED FOR VECTORIAL FOR $n BODIES ${NC}"
      exit 0
    fi
    
    cd ../

  fi

  #Check for OpenMP
  if [ $Open_MP == true ]; then

    echo -ne "${YELLOW} TESTING FOR   OPENMP   FOR $n BODIES ${NC}\r"

    #Delete old data
    if [ -d /dev/shm/data ]; then
      rm -f /dev/shm/data/*.csv 2>/dev/null
      rm -f /dev/shm/data/*.bin 2>/dev/null
    else
      mkdir /dev/shm/data 2>/dev/null
    fi

    ./simulation_OpenMP 0.1 ../Starting_Configurations/bin_files/random.bin >/dev/null

    cp /dev/shm/data/2.bin Check/OpenMP.bin >/dev/null

    cd Check/
    ./compare $1 secuential.bin OpenMP.bin >/dev/null

    if [ $? == 1 ];
    then
      echo -e "${CLEAR_LINE}${GREEN} PASSED FOR   OPENMP  FOR $n BODIES ${NC}"
    else
      echo -e "${CLEAR_LINE}${RED} FAILED FOR   OPENMP  FOR $n BODIES ${NC}"
      exit 0
    fi

    cd ../

  fi
  #Testing for vectorial
  if [ $vectorial_openmp == true ]; then 

    echo -ne "${YELLOW} TESTING FOR  OPENMP V FOR $n BODIES ${NC}\r"

    #Delete old data
    if [ -d /dev/shm/data ]; then
      rm -f /dev/shm/data/*.csv 2>/dev/null
      rm -f /dev/shm/data/*.bin 2>/dev/null
    else
      mkdir /dev/shm/data 2>/dev/null
    fi

    ./simulation_OpenMP_vectorial 0.1 ../Starting_Configurations/bin_files/random.bin >/dev/null

    cp /dev/shm/data/2.bin Check/openmp_vectorial.bin >/dev/null

    cd Check/
    ./compare $1 secuential.bin openmp_vectorial.bin

    if [ $? == 1 ];
    then
      echo -e "${CLEAR_LINE}${GREEN} PASSED FOR  OPENMP V FOR $n BODIES ${NC}"
    else
      echo -e "${CLEAR_LINE}${RED} FAILED FOR  OPENMP V FOR $n BODIES ${NC}"
      exit 0
    fi
    
    cd ../

  fi
  #Testing for cuda
  if [ $cuda == true ]; then 
    
    echo -ne "${YELLOW} TESTING FOR    CUDA   FOR $n BODIES ${NC}\r"
    #Delete old data
    if [ -d /dev/shm/data ]; then
      rm -f /dev/shm/data/*.csv 2>/dev/null
      rm -f /dev/shm/data/*.bin 2>/dev/null
    else
      mkdir /dev/shm/data 2>/dev/null
    fi

    ./simulation_cuda 0.1 ../Starting_Configurations/bin_files/random.bin >/dev/null

    cp /dev/shm/data/2.bin Check/cuda.bin >/dev/null

    cd Check/
    ./compare $1 secuential.bin cuda.bin

    if [ $? == 1 ];
    then
      echo -e "${CLEAR_LINE}${GREEN} PASSED FOR    CUDA   FOR $n BODIES ${NC}"
    else
      echo -e "${CLEAR_LINE}${RED} FAILED FOR    CUDA   FOR $n BODIES ${NC}"
      exit 0
    fi

    cd ../

  fi
done

exit 1



