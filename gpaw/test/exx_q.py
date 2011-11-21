from ase import *
from ase.dft import monkhorst_pack
from ase.structure import bulk
from gpaw import *
from gpaw.test import equal
import numpy as np

a0 = 5.43
cell = bulk('Si', 'fcc', a=a0).get_cell()
Si = Atoms('Si2', cell=cell, pbc=True,
           scaled_positions=((0,0,0), (0.25,0.25,0.25)))

kpts = monkhorst_pack((2,2,2))
kpts += np.array([1/4., 1/4., 1/4.])

calc = GPAW(h=0.18,
            kpts=kpts,
            occupations=FermiDirac(0.001))
Si.set_calculator(calc)
E = Si.get_potential_energy()

from gpaw.xc.hybridq import HybridXC
exx = HybridXC('EXX')
E_q = E + calc.get_xc_difference(exx)

from gpaw.xc.hybridk import HybridXC
exx = HybridXC('EXX')
E_k = E + calc.get_xc_difference(exx)

print 'Hartree-Fock ACDF method    :', E_q
print 'Hartree-Fock Standard method:', E_k

#equal(E_q, E_k, 0.001)
#equal(E_q, -27.71, 0.01)