set term jpeg
set output "Times.jpeg"
set title "Comparación de tiempo de ejecución para Simulaciones de N cuerpos"
set xlabel "Número de cuerpos"
set ylabel "Segundos de ejecución"
set yrange [0:250]
set xrange [0:25000]
set label "dt=0.1, speed=0.1"  at screen 0.5, 0.92 font "Arial,8"
plot "./logs/Secuencial.log" using 1:2 with lines t "Secuencial", "./logs/Secuencial_vectorial.log" using 1:2 with lines t "Secuencial Vectorial",  "./logs/OpenMP.log" using 1:2 with lines t "OpenMP",  "./logs/OpenMP_vectorial.log" using 1:2 with lines t "OpenMP Vectorial", "./logs/Cuda.log" using 1:2 with lines t "Cuda", "./logs/Treecode.log" using 1:2 with lines t "Treecode"
