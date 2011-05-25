from ase import Atom, Atoms
from gpaw import GPAW
from gpaw.test import equal

a = 4.05
d = a / 2**0.5
bulk = Atoms([Atom('Al', (0, 0, 0)),
              Atom('Al', (0.5, 0.5, 0.5))],
             pbc=True)
bulk.set_cell((d, d, a), scale_atoms=True)
h = 0.25
calc = GPAW(h=h,
                  nbands=2*8,
                  kpts=(2, 2, 2),
                  convergence={'energy': 1e-5})
bulk.set_calculator(calc)
e0 = bulk.get_potential_energy()
niter0 = calc.get_number_of_iterations()
calc = GPAW(h=h,
                  nbands=2*8,
                  kpts=(2, 2, 2),
                  convergence={'energy': 1e-5},
                  eigensolver='cg')
bulk.set_calculator(calc)
e1 = bulk.get_potential_energy()
niter1 = calc.get_number_of_iterations()
equal(e0, e1, 3.6e-5)

energy_tolerance = 0.00004
niter_tolerance = 0
equal(e0, -6.97626, energy_tolerance)
assert 16 <= niter0 <= 22, niter0
equal(e1, -6.97627, energy_tolerance)
assert 12 <= niter1 <= 17, niter1
