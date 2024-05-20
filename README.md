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

# JUAN
Primero ejecuta con OpenMP

    cd Plots
    ./create_plots_openMP.sh 201 200

Esto deberia tardar un minuto aprox. Luego ejecuta con cuda
    
    ./create_plots_cuda.sh 201 200



