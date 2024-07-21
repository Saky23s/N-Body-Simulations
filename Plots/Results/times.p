set term jpeg
set output "Times.jpeg"
set title "Comparación de tiempo de ejecución para Simulaciones de N cuerpos"
set xlabel "Número de cuerpos"
set ylabel "Segundos de ejecución"
set xrange [0:1000]
set yrange [0:3600]
set label "dt=0.1, speed=0.1"  at screen 0.5, 0.92 font "Arial,8"
plot "times_1to1000_secuential.log" using 1:2 with lines t "Secuencial old", "../secuential.log" using 1:2 with lines t "Secuencial new"
