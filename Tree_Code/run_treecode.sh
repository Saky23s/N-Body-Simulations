#!/bin/bash 

# Check if exactly two parameters are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <T> <Filepath> {-w -help}"
    exit 1
fi

TYPE="modified";
WINDOW=false
for arg in "$@" 
do
    case $arg in
        -w)
            WINDOW=true
            ;;
        -help) # Explain arguments
            echo "Usage: $0 <T> <Filepath> {-w -help}"
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
cd ../Tree_Code/Modified
make clean
make 

path="../$2"
./treecode $1 $path

#Run Graphics
cd ../../Graphics/
if [ "$WINDOW" = true ]; then
    cargo run w
else
    ./create_mp4.sh
fi
cd ../Tree_Code/
