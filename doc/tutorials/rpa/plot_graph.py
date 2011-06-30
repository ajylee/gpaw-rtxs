# creates: graphite.png

import matplotlib
matplotlib.use('Agg')
from pylab import *
import numpy as np

lda = np.loadtxt('lda_graph.dat').transpose()
pbe = np.loadtxt('pbe_graph.dat').transpose()
vdW = np.loadtxt('vdW_graph.dat').transpose()
hf= np.loadtxt('hf_graph.dat').transpose()
rpa = np.loadtxt('rpa_graph.dat').transpose()

plot(lda[0], (lda[1]-lda[1,-1])/4*1000, label='LDA')
plot(pbe[0], (pbe[1]-pbe[1,-1])/4*1000, label='PBE')
plot(vdW[0], (vdW[1]-vdW[1,-1])/4*1000, label='vdW-DF')
plot(rpa[0], (hf[1]-hf[1,-1]+rpa[1]-rpa[1,-1])/4*1000, 's',
     c='0.0', label='HF+RPA')
errorbar(3.426, -55, yerr=5.0, elinewidth=2.0, capsize=5, label='QMC')
errorbar(3.34, -44, yerr=14.0, elinewidth=2.0, capsize=5, c='0.5', label='Exp')

axis([2.8, 6.0,-65, 20])
legend(loc='lower right')
xlabel('d [A]')
ylabel('Binding energy per atom [meV]')
savefig('graphite.png')
#show()
