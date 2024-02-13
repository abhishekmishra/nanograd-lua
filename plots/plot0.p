# Image output of size 800x600
set terminal png size 800,600
# Output file name
set output 'plot0.png'
# Plot title
set title 'f(x)=3*(x^2) - 4*x + 5 over x=[-5, 5]'
# Plot the data
plot 'plot0-x-fx-range.data' with linespoints