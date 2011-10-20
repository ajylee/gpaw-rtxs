from ase import *
from ase.units import Ha
from ase.structure import bulk
from ase.dft import monkhorst_pack
from gpaw import *
from gpaw.response.cell import get_primitive_cell, set_Gvectors
from gpaw.mpi import serial_comm
from gpaw.test import equal
from gpaw.xc.rpa_correlation_energy import RPACorrelation
import numpy as np

kpts = monkhorst_pack((2,2,2))
kpts += np.array([1/4., 1/4., 1/4.])

ecut = 50

calc = GPAW(h=0.18,
            xc='LDA',
            kpts=kpts,
            nbands=8,
            communicator=serial_comm)

Kr = bulk('Kr', 'fcc', a=5.0)
Kr.set_calculator(calc)
Kr.get_potential_energy()

acell = Kr.cell / Bohr
bcell = get_primitive_cell(acell)[1]
gpts = calc.get_number_of_grid_points()
bands_cut = set_Gvectors(acell, bcell, gpts,
                         [ecut/Ha, ecut/Ha, ecut/Ha])[0]
calc.set(fixdensity=True,
         nbands=bands_cut+10,
         eigensolver='cg', 
         convergence={'bands': bands_cut})
calc.get_potential_energy()

alda = RPACorrelation(calc, qsym=False)
E_rpa_noqsym = alda.get_rpa_correlation_energy(ecut=ecut,
                                               directions=[[0, 1.0]],
                                               gauss_legendre=8)

rpa = RPACorrelation(calc, qsym=True)
E_rpa_qsym = rpa.get_rpa_correlation_energy(ecut=ecut,
                                            directions=[[0, 1.0]],
                                            gauss_legendre=8)


equal(E_rpa_qsym, E_rpa_noqsym, 0.001)
equal(E_rpa_qsym, -4.38, 0.01)
