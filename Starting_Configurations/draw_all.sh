#!/bin/bash 

make clean
make

for (( n=1000; n<=10000; n+=1000 ))
do 
    ./galaxy_style_configuration $n
    cd ../Metodo_Directo
    ./run_simulation.sh 2 ../Starting_Configurations/bin_files/galaxy.bin -c -w
    cd ../Graphics

    cp simulation.mp4 mp4/galaxy$n.mp4

    cd ../Starting_Configurations

done

