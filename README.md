# Three-Body-Problem
Este software requiere `cargo`, `rustc`, `gcc` y un sistema operativo linux.
Para ejecutar cuda se necesita `nvcc`.

## Ejemplo de ejecucion secuencial
    cd Solucion_Ingenua
    ./run_simulation_secuential.sh 100 ../Starting_Configurations/csv_files/ocho.csv

## Ejemplo de ejecucion paralela con Open_MP
    cd Solucion_Ingenua
    ./run_simulation_OpenMP.sh 100 ../Starting_Configurations/csv_files/ocho.csv

## Ejemplo de ejecucion paralela con cuda
    cd Solucion_Ingenua
    ./run_simulation_cuda.sh 100 ../Starting_Configurations/csv_files/ocho.csv

