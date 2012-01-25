from ase import Atoms
from ase.structure import molecule
from gpaw import GPAW, FermiDirac
from gpaw.wavefunctions.pw import PW
from gpaw.mpi import world

a = molecule('H', pbc=1)
a.center(vacuum=2)

comm = world.new_communicator([0])
e0 = 0.0
if world.rank == 0:
    a.calc = GPAW(mode=PW(250),
                  communicator=comm,
                  txt=None)
    e0 = a.get_potential_energy()
e0 = world.sum(e0)

if world.size == 1:
    bandpar = 1
else:
    bandpar = world.size // 2

a.calc = GPAW(mode=PW(250),
              basis='szp(dzp)',
              parallel=dict(band=bandpar),
              txt='%d.txt' % world.size)
e = a.get_potential_energy()
assert abs(e - e0) < 3e-5

