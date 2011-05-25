import numpy as np
from pylab import *

A = np.loadtxt('rpa_Kr.dat').transpose()
xs = np.array([170 +i*100. for i in range(500)])
plot(A[0]**(-1.5), A[1], 'o', markersize=8, label='Calculated points')

plot(xs**(-1.5), -10.4805+4080.97*xs**(-1.5), label='Fit: -10.5+4081*E^(-1.5)')

t = [int(A[0,i]) for i in range(len(A[0]))]
xticks(A[0]**(-1.5), t)
xlabel('Cutoff energy [eV]', fontsize=16)
ylabel('Correlation energy', fontsize=16)
axis([0.,None,None,None])
title('RPA correlation energy of fcc Kr lattice at $V=40\,\AA^3$')
legend(loc='upper left')
#show()
savefig('extrapolate_Kr.png')
