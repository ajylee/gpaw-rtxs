from ase import *
from gpaw import GPAW
from gpaw.xc.bee import BEEF_Ensemble
import numpy as np

xc = 'BEEF-vdW'
d = 0.75

# H2 molecule
h2 = Atoms('H2',[[0.,0.,0.],[0.,0.,d]])
h2.center(vacuum=3.)
cell = h2.get_cell()
calc = GPAW(xc=xc)
h2.set_calculator(calc)
e_h2 = h2.get_potential_energy()
f = h2.get_forces()
ens = BEEF_Ensemble(calc)
de_h2 = ens.get_ensemble_energies()
del h2, calc, ens

# H atom
h = Atoms('H')
h.set_cell(cell)
h.center()
calc = GPAW(xc=xc, spinpol=True)
h.set_calculator(calc)
e_h = h.get_potential_energy()
ens = BEEF_Ensemble(calc)
de_h = ens.get_ensemble_energies()

# forces
f0 = f[0].sum()
f1 = f[1].sum()
assert abs(f0 + f1) < 1.e-10
assert abs(f0 - 0.534993531493) < 1.e-10

# binding energy
E_bind = 2*e_h - e_h2
dE_bind = 2*de_h[:] - de_h2[:]
dE_bind = np.std(dE_bind)
assert abs(E_bind - 4.8738652912) < 1.e-10
assert abs(dE_bind - 0.193965700193) < 1.e-10
