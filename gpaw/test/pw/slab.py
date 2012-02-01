import numpy as np
from ase import Atoms
from gpaw import GPAW
from gpaw.wavefunctions.pw import PW
from gpaw.test import equal
from gpaw.mpi import world

a = 2.65
bulk = Atoms('Li', cell=(a, a, 3 * a), pbc=True)
k = 4
calc = GPAW(mode=PW(200),
            parallel={'band': min(world.size, 4)},
            idiotproof=0,
            kpts=(k, k, 1))
bulk.set_calculator(calc)
bulk.get_potential_energy()
