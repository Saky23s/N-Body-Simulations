set term jpeg
set output "Times.jpeg"
set title "Comparacion de tiempo de ejecucion para Simulaciones de N cuerpos"
set xlabel "Numero de cuerpos"
set ylabel "Segundos de ejecucion"
set xrange [0:2000]
plot "times_1to1000_secuential.log" using 1:2 with lines t "Secuencial", "times_1to2000_openmp.log" using 1:2 with lines t "OpenMP", "times_1to1000_cuda_openmp.log" using 1:2 with lines t "Cuda and OpenMP" 
