import numpy as np
from pylab import *

A = loadtxt('frequency_equidistant.dat').transpose()
plot(A[0], A[1], label='Equidistant')

A = loadtxt('frequency_N8_B2.0.dat').transpose()
plot(A[0], A[1], 'o', label='N=8, B=2.0', markersize=8)

A = loadtxt('frequency_N16_B1.2.dat').transpose()
plot(A[0], A[1], 'o', label='N=16, B=1.2', markersize=8)

A = loadtxt('frequency_N16_B2.0.dat').transpose()
plot(A[0], A[1], 'o', label='N=16, B=2.0', markersize=8)

xlabel('Frequency [eV]', fontsize=18)
ylabel('Integrand', fontsize=18)
axis([0, 50, None, None])
legend(loc='lower right')
#show()
savefig('integration.png')
