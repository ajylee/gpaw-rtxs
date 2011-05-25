import numpy as np
from ase.structure import bulk
from gpaw import FermiDirac, MethfesselPaxton, MixerSum, \
         KohnShamConvergenceError, PoissonSolver
from gpaw.utilities.bulk2 import GPAWRunner

strains = np.linspace(0.98, 1.02, 9)
a0 = 2.84
atoms = bulk('Fe', 'bcc', a0, cubic=True)
atoms.set_initial_magnetic_moments([2.3, 2.3])
def f(name, dist, k, g):
    tag = '%s-%02d-%2d' % (name, k, g)
    r = GPAWRunner('Fe', atoms, strains, tag=tag)
    r.set_parameters(xc='PBE',
                     occupations=dist,
                     basis='dzp',
                     mixer=MixerSum(0.05, 5, 1),
                     eigensolver='cg',
                     maxiter=500,
                     poissonsolver=PoissonSolver(eps=1e-12), 
                     kpts=(k, k, k),
                     gpts=(g, g, g))
    try:
        r.run()
    except KohnShamConvergenceError:
        pass

for width in [0.05, 0.1, 0.15, 0.2]:
    for k in [4, 6, 8, 10, 12]:
        f('FD-%.2f' % width, FermiDirac(width), k, 12)
        f('MP-%.2f' % width, MethfesselPaxton(width), k, 12)
        #f('MP1-%.2f' % width, MethfesselPaxton(width, 1), k, 12)
for g in range(16, 32, 4):
    f('FD-%.2f' % 0.1, FermiDirac(0.1), 8, g)
