set term jpeg
set output "Times.jpeg"
set title "Tiempo de ejecucion para Simulaciones de N cuerpos"
set xlabel "Numero de cuerpos"
set ylabel "Tiempo de ejecucion"
n2(x) = a * x * x + b
fit n2(x) "times.log" using 1:2 via a,b
plot "times.log" using 1:2 with lines, n2(x)