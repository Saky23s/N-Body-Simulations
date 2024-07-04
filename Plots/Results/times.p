set term jpeg
set output "Times.jpeg"
set title "Comparación de tiempo de ejecución para Simulaciones de N cuerpos"
set xlabel "Número de cuerpos"
set ylabel "Segundos de ejecución"
set xrange [0:4000]
set yrange [0:6000]
set label "dt=0.1, speed=0.1"  at screen 0.5, 0.92 font "Arial,8"
plot "times_1to1000_secuential.log" using 1:2 with lines t "Secuencial", "times_1to2000_OpenMPv1.log" using 1:2 with lines t "OpenMP_v1", "times_1to3500_OpenMPv2.log" using 1:2 with lines t "OpenMP_v2" , "times_1to4000_cuda.log" using 1:2 with lines t "CUDA"
