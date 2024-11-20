set term jpeg
set output "img/Treecode_vs_everyone.jpeg"
set title "Comparación de tiempo de ejecución para Simulaciones de N cuerpos"
set xlabel "Número de cuerpos"
set ylabel "Segundos de ejecución"
set key below
set yrange [0:250]
set xrange [0:25000]
set label "dt=0.1, speed=0.1"  at screen 0.5, 0.92 font "Arial,8"

plot "./logs/Secuencial_vectorial.log" using 1:2 with lines t "Secuencial" lt rgb "light-green","./logs/OpenMP_vectorial.log" using 1:2 with lines t "OpenMP" lt rgb "skyblue", "./logs/Cuda.log" using 1:2 with lines t "Cuda" lt rgb "magenta","./logs/Cuda_V2.log" using 1:2 with lines t "Cuda V2" lt rgb "purple", "./logs/Treecode.log" using 1:2 with lines t "Treecode" lt rgb "orange"