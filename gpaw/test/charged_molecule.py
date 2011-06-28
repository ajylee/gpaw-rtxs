# this test checks if the calculation of charged
# molecules is possible
#
import numpy as np

from ase import Atoms
from gpaw import GPAW, FermiDirac, Mixer
from gpaw.test import equal

gs=0.4
vac=3.0
d=2.0   # does not matter
# fixmagmom is not needed - but it speeds up the calculation
# about 20%
calc=GPAW(occupations=FermiDirac(width=0.1, fixmagmom=True),
        charge=1.0,
        h=gs,
        mixer=Mixer(beta=0.75, nmaxold=2, weight=50.0),
        convergence={'energy':0.5, 'eigenstates': 1.e-0,
                     'density': 1.e+1})
atoms = Atoms('Li2', positions=[(0,0,0),(0,0,d)], pbc=False)
atoms.center(vacuum=vac)
mm = [1] * 2
mm[0] = 1.
mm[1] = 0.
atoms.set_initial_magnetic_moments(mm)
atoms.set_calculator(calc)
atoms.get_potential_energy()
