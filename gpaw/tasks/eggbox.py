import optparse

import numpy as np
from ase.atoms import Atoms

from gpaw import FermiDirac
from gpaw.tasks.convergence import ConvergenceTestTask


class EggboxTestTask(ConvergenceTestTask):
    def __init__(self, **kwargs):
        ConvergenceTestTask.__init__(self, **kwargs)
        
        self.taskname = 'eggbox'

    def calculate(self, name, atoms):
        g = int(name.split('-')[1])
        atoms.calc.set(gpts=(g, g, g),
                       occupations=FermiDirac(0.1),
                       kpts=[1, 1, 1])
        energies = []
        forces = []
        for i in range(25):
            x = self.L / g * i / 48
            atoms[0].x = x
            e = atoms.calc.get_potential_energy(atoms,
                                                force_consistent=True)
            f = atoms.get_forces()[0, 0]
            energies.append(e)
            forces.append(f)

        return {'energies': energies,
                'forces': forces}

    def analyse(self):
        self.summary_header = [('name', '')] + [
            ('dE(h=%.2f)' % (self.L / g), 'meV') for g in self.gs]

        for name, data in self.data.items():
            symbol, g = name.split('-')
            g = int(g)
            de = data['energies'].ptp()
            if symbol not in self.results:
                self.results[symbol] = [None] * len(self.gs)
            self.results[symbol][self.gs.index(g)] = de * 1000


if __name__ == '__main__':
    task = EggboxTestTask()
    args = task.parse_args()
    task.run(args)
