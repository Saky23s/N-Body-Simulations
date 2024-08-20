# Three-Body-Problem
Este software requiere `cargo`, `rustc`, `gcc` y un sistema operativo linux.
Para ejecutar cuda se necesita `nvcc`.

## Ejemplo de ejecucion secuencial
    cd Metodo_Directo
    ./run_simulation.sh 100 ../Starting_Configurations/csv_files/ocho.csv

## Ejemplo de ejecucion paralela con Open_MP
    cd Metodo_Directo
    ./run_simulation.sh 100 ../Starting_Configurations/csv_files/ocho.csv -o

## Ejemplo de ejecucion paralela con cuda
    cd Metodo_Directo
    ./run_simulation.sh 100 ../Starting_Configurations/csv_files/ocho.csv -c




