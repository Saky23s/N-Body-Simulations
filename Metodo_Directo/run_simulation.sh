#!/bin/bash 

# Check if exactly two parameters are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <T> <Filepath> {-c -o -s -h}"
    exit 1
fi

TYPE="secuencial";
WINDOW=false
for arg in "$@" 
do
    case $arg in
        -s) 
            TYPE="secuencial"
            ;;
        -o) 
            TYPE="OpenMP"
            ;;
        -c)
            TYPE="cuda"
            ;;
        
        -v)
            TYPE="secuencial_vectorial"
            ;;
        -vo)
            TYPE="OpenMP_vectorial"
            ;;
        -ov)
            TYPE="OpenMP_vectorial"
            ;;
        -w)
            WINDOW=true
            ;;
        -help) # Explain arguments
            echo "Usage: $0 <T> <Filepath> {-c -o -s}"
            echo "-c: Run with cuda"
            echo "-o: Run with OpenMP"
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
cd ../Metodo_Directo/
rm simulation_${TYPE} >/dev/null
make simulation_${TYPE} >/dev/null
./simulation_${TYPE} $1 $2

#Run Graphics
cd ../Graphics/
if [ "$WINDOW" = true ]; then
    cargo run w
else
    ./create_mp4.sh
fi
cd ../Metodo_Directo/
