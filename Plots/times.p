set term jpeg
set output "Times.jpeg"
set title "Tiempo de ejecucion para Simulaciones de N cuerpos"
set xlabel "Numero de cuerpos"
set ylabel "Segundos de ejecucion"
set key below

n2(x) = a * x * x + b
fit n2(x) "times.log" using 1:2 via a,b

set label "dt=0.1, speed=0.1"  at screen 0.5, 0.92 font "Arial,8"
plot "times.log" using 1:2 with lines t "Times", n2(x)