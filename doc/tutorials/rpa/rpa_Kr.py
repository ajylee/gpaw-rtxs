from ase import *
from ase.parallel import paropen
from gpaw import *
from gpaw.mpi import serial_comm
from gpaw.xc.rpa_correlation_energy import RPACorrelation
import numpy as np

calc = GPAW('Kr_gs.gpw', communicator=serial_comm, txt=None)
rpa = RPACorrelation(calc, txt='new_rpa_Kr.txt')

f = paropen('new_rpa_Kr.dat', 'w')

for ecut in [150, 175, 200, 225, 250, 275, 300]:
    E_rpa = rpa.get_rpa_correlation_energy(ecut=ecut, 
                                           kcommsize=8, 
                                           directions=[[0, 1.]])
    print >> f, ecut, E_rpa

f.close()
