from ase import *
from ase.structure import bulk
from ase.dft import monkhorst_pack
from gpaw import *
from gpaw.mpi import serial_comm
from gpaw.test import equal
from gpaw.xc.rpa_correlation_energy import RPACorrelation
import numpy as np

kpts = monkhorst_pack((2,2,2))
kpts += np.array([1/4., 1/4., 1/4.])

calc = GPAW(h=0.18,
            xc='LDA',
            kpts=kpts,
            #kpts=(2,2,2),
            nbands=30,
            eigensolver='cg', 
            convergence={'bands': -5},
            communicator=serial_comm)

V = 30.
a0 = (4.*V)**(1/3.)
Kr = bulk('Kr', 'fcc', a=a0)
Kr.set_calculator(calc)
Kr.get_potential_energy()

rpa = RPACorrelation(calc, qsym=True)
E_rpa = rpa.get_rpa_correlation_energy(ecut=50,
                                       nbands=25,
                                       directions=[[0, 1.0]],
                                       gauss_legendre=8)

equal(E_rpa, -4.79, 0.01)
