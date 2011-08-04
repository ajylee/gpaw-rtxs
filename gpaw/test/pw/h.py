from ase import Atoms
from ase.data.molecules import molecule
from gpaw import GPAW, FermiDirac
from gpaw.wavefunctions.pw import PW
from gpaw.mpi import world

if world.size <= 2:
    a = molecule('H', pbc=1)
    a.center(vacuum=2)
    a.set_calculator(GPAW(mode=PW(250)))
    a.get_potential_energy()
