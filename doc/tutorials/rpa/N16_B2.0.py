from ase import *
from ase.parallel import paropen
from gpaw import *
from gpaw.mpi import serial_comm
from gpaw.xc.rpa_correlation_energy import RPACorrelation
import numpy as np

ecut = 50

calc = GPAW('N2.gpw', communicator=serial_comm, txt=None)

rpa = RPACorrelation(calc, txt='frequency_N16_B2.0.txt')

Es = rpa.get_E_q(ecut=ecut,
                 integrated=False,
                 q=[0,0,0],
                 direction=0)

f = paropen('frequency_N16_B2.0.dat', 'w')
for w, E in zip(rpa.w, Es):
    print >> f, w, E.real
f.close()
