import numpy as np
from ase.tasks.bulk import BulkTask
from gpaw import FermiDirac, MethfesselPaxton, MixerSum, \
         KohnShamConvergenceError, PoissonSolver
from gpaw.factory import GPAWFactory

a0 = 2.84
def f(name, dist, k, g):
    tag = '%s-%02d-%2d' % (name, k, g)
    task = BulkTask(tag=tag, lattice_constant=a0, cubic=True,
                    kpts=(k, k, k), magmoms=[2.3],
                    fit=(5, 0.02))
    factory = GPAWFactory(xc='PBE',
                          occupations=dist,
                          basis='dzp',
                          mixer=MixerSum(0.05, 5, 1),
                          eigensolver='cg',
                          maxiter=500,
                          poissonsolver=PoissonSolver(eps=1e-12), 
                          gpts=(g, g, g))
    task.set_calculator_factory(factory)
    task.run('Fe')

for width in [0.05, 0.1, 0.15, 0.2]:
    for k in [4, 6, 8, 10, 12]:
        f('FD-%.2f' % width, FermiDirac(width), k, 12)
        f('MP-%.2f' % width, MethfesselPaxton(width), k, 12)
        #f('MP1-%.2f' % width, MethfesselPaxton(width, 1), k, 12)
for g in range(16, 32, 4):
    f('FD-%.2f' % 0.1, FermiDirac(0.1), 8, g)
