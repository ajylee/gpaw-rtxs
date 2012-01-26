from ase import *
from ase.structure import molecule
from ase.units import Ha, Bohr
from gpaw import *
from gpaw.xc.hybrid import HybridXC
from gpaw.response.cell import get_primitive_cell, set_Gvectors
from gpaw.mpi import serial_comm
from gpaw.test import equal
from gpaw.xc.rpa_correlation_energy import RPACorrelation
import numpy as np

N2 = molecule('N2', pbc=True)
N2.center(vacuum=2.0)

calc = GPAW(h=0.18, xc='PBE', communicator=serial_comm)
N2.set_calculator(calc)
E_n2_pbe = N2.get_potential_energy()
E_n2_hf = E_n2_pbe + calc.get_xc_difference(HybridXC('EXX'))

ecut = 25
acell = N2.cell / Bohr
bcell = get_primitive_cell(acell)[1]
gpts = calc.get_number_of_grid_points()
bands_cut = set_Gvectors(acell, bcell, gpts,
                         [ecut/Ha, ecut/Ha, ecut/Ha])[0]

calc.set(fixdensity=True,
         nbands=bands_cut+10,
         eigensolver='cg', 
         convergence={'bands': bands_cut})
calc.get_potential_energy()

rpa = RPACorrelation(calc)
E_n2_rpa = rpa.get_rpa_correlation_energy(ecut=ecut,
                                          directions=[[0, 2/3.], [2, 1/3.]],
                                          gauss_legendre=8)

# -------------------------------------------------------------------------

N = molecule('N', pbc=True)
N.set_cell(N2.cell)
N.center()
calc = GPAW(h=0.18, xc='PBE', communicator=serial_comm)
N.set_calculator(calc)
E_n_pbe = N.get_potential_energy()
E_n_hf = E_n_pbe + calc.get_xc_difference(HybridXC('EXX'))

calc.set(fixdensity=True,
         nbands=bands_cut+10,
         eigensolver='cg',
         convergence={'bands': bands_cut})
N.set_calculator(calc)
calc.get_potential_energy()

rpa = RPACorrelation(calc)
E_n_rpa = rpa.get_rpa_correlation_energy(ecut=ecut,
                                         directions=[[0, 1.0]],
                                         gauss_legendre=8)
print 'Atomization energies:'
print 'PBE: ', E_n2_pbe - 2*E_n_pbe
print 'HF: ',  E_n2_hf - 2*E_n_hf
print 'HF+RPA: ', E_n2_hf - 2*E_n_hf + E_n2_rpa - 2*E_n_rpa, '(Not converged!)'

equal(E_n2_rpa - 2*E_n_rpa, -1.87, 0.01)
