set term jpeg
set output "Results/img/Treecode.jpeg"
set title "Tiempo de ejecucion para Simulaciones de N cuerpos"
set xlabel "Numero de cuerpos"
set ylabel "Segundos de ejecucion"
set key below

set xrange [0:25000]

n2(x) = a * x * x + b
fit n2(x) "Results/logs/Treecode.log" using 1:2 via a,b

nlogn(x) = a * x * log(x*a) + b
fit nlogn(x) "Results/logs/Treecode.log" using 1:2 via a,b

set label "dt=0.1, speed=0.1"  at screen 0.5, 0.92 font "Arial,8"
plot "Results/logs/Treecode.log" using 1:2 with lines t "Treecode", nlogn(x)