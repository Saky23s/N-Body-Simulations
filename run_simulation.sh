#!/bin/bash 

# Check if exactly two parameters are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <T> <Filepath> {-s -o -c -c2 -v -vo -ov -w -t -help}"
    exit 1
fi

#Array de argumentos vacios
array=()
treecode=false;
WINDOW=false
for arg in "$@" 
do
    case $arg in
        -s) 
            treecode=false;
            array+=" $arg "
            ;;
        -o) 
            treecode=false;
            array+=" $arg "
            ;;
        -c)
            treecode=false;
            array+=" $arg "
            ;;
        
        -c2)
            treecode=false;
            array+=" $arg "
            ;;
        -v)
            treecode=false;
            array+=" $arg "
            ;;
        -vo)
            treecode=false;
            array+=" $arg "
            ;;
        -ov)
            treecode=false;
            array+=" $arg "
            ;;
        -t)
            treecode=true;
            array+=" $arg "
            ;;
        -w)
            WINDOW=true
            array+=" $arg "
            ;;
        -help) # Explain arguments
            echo "Usage: $0 <T> <Filepath> {-s -o -c -c2 -v -vo -ov -w -help}"
            echo "-s (default): Run metodo directo secuential"
            echo "-o: Run metodo directo with OpenMP"
            echo "-c: Run metodo directo with cuda version 1"
            echo "-c2: Run metodo directo with cuda version 2"
            echo "-v: Run metodo directo secuential with march"
            echo "-ov or -vo: Run metodo directo OpenMP with march"
            echo "-t: Run treecode"
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

path="../$2"

if [ $treecode == "true" ]; then
    cd "Tree_Code/"
    ./run_treecode.sh $1 $path ${array[@]}
    cd "../"
else
    cd "./Metodo_Directo/"
    ./run_metodo_directo.sh $1 $path ${array[@]}
    cd "../"
fi
 
