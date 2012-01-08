from ase import *
from gpaw import GPAW

xc = 'BEEF-vdW'
d = 0.8
h2 = Atoms('H2', [[0.0,0.0,0.0],[0.0,0.0,d]], pbc=False)
h2.center(vacuum=3.)
calc = GPAW(xc=xc,txt=None)
h2.set_calculator(calc)
e = h2.get_potential_energy()
assert e - -7.927315 < 1.e-6
f = h2.get_forces()
f0 = f[0].sum()
f1 = f[1].sum()
assert abs(f0 + f1) < 1.e-10
assert f0 - 1.95088181246 < 1.e-10
