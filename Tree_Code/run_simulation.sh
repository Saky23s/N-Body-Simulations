#!/bin/bash 
TYPE="secuencial";
WINDOW=false

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <T> <Filepath> {-s -w -h}"
    exit 1
fi

for arg in "$@" 
do
    case $arg in
        -s) 
            TYPE="secuencial"
            ;;
        -w)
            WINDOW=true
            ;;
        -help) # Explain arguments
            echo "Usage: $0 <T> <Filepath> {-s -w -h}"
            echo "-s (default): Run secuential"
            echo "-w: Show result as a window"
            echo "-help: Show help"
            exit
            ;;
        *)
            ;;
    esac
done

#Delete old data
if [ -d /dev/shm/data ]; then
  rm -f /dev/shm/data/*.csv
  rm -f /dev/shm/data/*.bin
else
  mkdir /dev/shm/data
fi

#Create starting configuration data
cd ../Starting_Configurations
make >/dev/null
./graphic_starting_position $2

#Run simulation
cd ../Tree_Code/
make clean >/dev/null
make >/dev/null
./treecode_secuencial $1 $2

#Run Graphics
cd ../Graphics/
if [ "$WINDOW" = true ]; then
    cargo run w
else
    ./create_mp4.sh
fi
cd ../Tree_Code/
make clean >/dev/null
