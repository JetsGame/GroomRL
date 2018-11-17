set term pdfcairo enhanced color size 12cm,12cm lw 2

set output 'SD-reward-test.pdf'

min(x,y) = (x < y) ? x : y
max(x,y) = (x > y) ? x : y


set view map
set xrange [0:8]
set yrange [-10:0]
set ylabel 'ln z'
set xlabel 'ln 1/{/Symbol D}'

# linear boundary
f(x,y,a,b,z)=exp(-a*x+b*(-z-y))
set title 'min(exp(-0.5*ln 1/{/Symbol D} + 0.5*(-6-ln z)),1)'
splot min(f(x,y,0.5,0.5,6),1) w pm3d

set title 'max(1-exp(-0.4*ln 1/{/Symbol D} + 0.4*(-4-ln z)),0)'
splot max(1-f(x,y,0.4,0.2,4),0) w pm3d

# circular boundary
g(x,y,b)=exp(b*x*(-y-8))
set title 'min(exp(ln 1/{/Symbol D} (-8-ln z)),1)'
splot min(g(x,y,1.0),1) with pm3d

set title 'max(1 - exp(0.1*ln 1/{/Symbol D} (-8-ln z)),0)'
splot max(1-g(x,y,0.1),0) with pm3d

