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

cd ../Original/

make clean *>/dev/null
make treecode  >/dev/null

cd ../Modified/

make clean *>/dev/null
make treecode  >/dev/null

cd ../../Starting_Configurations/
make clean *>/dev/null
make all  >/dev/null

cd ../Tree_Code/Check/

for (( n=$1; n<=$2; n+=$3 ))
do 
    echo -ne "${YELLOW} TESTING ${NC} $n bodies...\r"
    
    cd ../../Starting_Configurations/
    ./plummer_configuration $n

    cd ../Tree_Code/Check

    #Delete old data
    if [ -d /dev/shm/data ]; then
        rm -f /dev/shm/data/*.csv >/dev/null
        rm -f /dev/shm/data/*.bin >/dev/null
    else
        mkdir /dev/shm/data >/dev/null
    fi

    cd ../Original/
    ./treecode  >/dev/null
    cp /dev/shm/data/5.bin ../Check/original.bin >/dev/null

    #Delete old data
    if [ -d /dev/shm/data ]; then
        rm -f /dev/shm/data/*.csv 2>/dev/null
        rm -f /dev/shm/data/*.bin 2>/dev/null
    else
        mkdir /dev/shm/data 2>/dev/null
    fi

    cd ../Modified/
    ./treecode  >/dev/null

    cp /dev/shm/data/5.bin ../Check/modified.bin >/dev/null

    cd ../Check/
    ./compare $1 original.bin modified.bin >/dev/null

    if [ $? == 1 ];
    then
      echo -e "${CLEAR_LINE}${GREEN} PASSED ${NC} $n bodies"
    else
      echo -e "${CLEAR_LINE}${RED} FAILED ${NC} $n bodies"
      exit 0
    fi
done

exit 1



