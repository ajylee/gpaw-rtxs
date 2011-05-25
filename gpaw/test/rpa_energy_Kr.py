from ase import *
from ase.structure import bulk
import numpy as np
from gpaw import *
from gpaw.mpi import serial_comm
from gpaw.test import equal
from gpaw.xc.rpa_correlation_energy import RPACorrelation

calc = GPAW(h=0.18, xc='LDA', kpts=(2,2,2),
            nbands=32, eigensolver='cg', 
            convergence={'bands': -5},
            communicator=serial_comm)

V = 30.
a0 = (4.*V)**(1/3.)
Kr = bulk('Kr', 'fcc', a=a0)

Kr.set_calculator(calc)
Kr.get_potential_energy()

ecut = 50.
rpa = RPACorrelation(calc)
E_rpa = rpa.get_rpa_correlation_energy(ecut=ecut,
                                       directions=[[0, 1.0]],
                                       gauss_legendre=8)

equal(E_rpa, -4.56, 0.1)
