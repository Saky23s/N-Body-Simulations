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
        -w)
            WINDOW=true
            ;;
        -h) # Explain arguments
            echo "Usage: $0 <T> <Filepath> {-c -o -s}"
            echo "-c: Run with cuda"
            echo "-o: Run with OpenMP"
            echo "-s (default): Run secuential"
            echo "-h: Show help"
            exit
            ;;
        *)
            ;;
    esac
done

#Delete old data
if [ -d ../Graphics/data ]; then
  rm -f ../Graphics/data/*.csv
  rm -f ../Graphics/data/*.bin
else
  mkdir ../Graphics/data
fi

#Create starting configuration data
cd ../Starting_Configurations
make >/dev/null
./graphic_starting_position $2

#Run simulation
cd ../Solucion_Ingenua/
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
cd ../Solucion_Ingenua/
