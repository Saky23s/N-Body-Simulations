# Three-Body-Problem
Este software requiere `cargo`, `rustc`, `gcc` y un sistema operativo linux.
Estas dependencias se pueden descargar automaticamente ejecutando:

    sudo ./install_dependencies.sh

Para ejecutar cuda se necesita `nvcc`.

## Ejemplo de ejecucion secuencial de metodo directo
    ./run_simulation.sh 100 Starting_Configurations/csv_files/ocho.csv -s -w

## Ejemplo de ejecucion paralela con Open_MP de metodo directo
    ./run_simulation.sh 100 Starting_Configurations/csv_files/ocho.csv -o -w

## Ejemplo de ejecucion paralela con cuda de metodo directo
    ./run_simulation.sh 100 Starting_Configurations/csv_files/ocho.csv -c -w

## Ejemplo de ejecucion con treecode
    ./run_simulation.sh 100 Starting_Configurations/csv_files/ocho.csv -t -w




