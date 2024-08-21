# Three-Body-Problem
Este software requiere `cargo`, `rustc`, `gcc` y un sistema operativo linux.
Estas dependencias se pueden descargar automaticamente ejecutando:
    sudo ./install_dependencies.sh

Para ejecutar cuda se necesita `nvcc`.

## Ejemplo de ejecucion secuencial de metodo directo
    cd Metodo_Directo
    ./run_simulation.sh 100 ../Starting_Configurations/csv_files/ocho.csv -w

## Ejemplo de ejecucion paralela con Open_MP de metodo directo
    cd Metodo_Directo
    ./run_simulation.sh 100 ../Starting_Configurations/csv_files/ocho.csv -o -w

## Ejemplo de ejecucion paralela con cuda de metodo directo
    cd Metodo_Directo
    ./run_simulation.sh 100 ../Starting_Configurations/csv_files/ocho.csv -c -w




