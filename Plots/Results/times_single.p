set term jpeg
set output "img/Cuda.jpeg"
set title "Comparación de tiempo de ejecución para Simulaciones de N cuerpos"
set xlabel "Número de cuerpos"
set ylabel "Segundos de ejecución"
set key below
set yrange [0:300]
set xrange [0:10000]
set label "dt=0.1, speed=0.1"  at screen 0.5, 0.92 font "Arial,8"

n2(x) = a * x * x + b
fit n2(x) "logs/Cuda.log" using 1:2 via a,b

nlogn(x) = a * x * log(x*a) + b
#fit nlogn(x) "logs/Treecode.log" using 1:2 via a,b

plot "./logs/Cuda.log" using 1:2 with lines t "Cuda" lt rgb "magenta", n2(x)
#plot "./logs/Treecode.log" using 1:2 with lines t "Treecode" lt rgb "orange",  nlogn(x)