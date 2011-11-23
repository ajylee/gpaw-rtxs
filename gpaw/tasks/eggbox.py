import optparse

import numpy as np
from ase.atoms import Atoms

from gpaw import FermiDirac
from gpaw.tasks.convergence import ConvergenceTestTask


class EggboxTestTask(ConvergenceTestTask):
    taskname = 'eggbox'

    def __init__(self, **kwargs):
        """Calculate size of eggbox error.

        A single atom is translated from (0, 0, 0) to (h / 2, 0, 0) in
        25 steps in order to measure to eggbox error."""

        ConvergenceTestTask.__init__(self, **kwargs)
        
    def calculate(self, name, atoms):
        atoms.calc.set(occupations=FermiDirac(0.1),
                       kpts=[1, 1, 1])
        data = {}
        for g in self.gs:
            atoms.calc.set(gpts=(g, g, g))
            energies = []
            forces = []
            for i in range(25):
                x = self.L / g * i / 48
                atoms.positions[0] = x
                e = atoms.calc.get_potential_energy(atoms,
                                                    force_consistent=True)
                energies.append(e)
                forces.append(atoms.get_forces()[0,0])
            data[g] = (energies, forces)

        return data

    def analyse(self):
        self.summary_header = [('name', '')] + [
            ('dE(h=%.2f)' % (self.L / g), 'meV') for g in self.gs]

        for name, data in self.data.items():
            results = []
            for g in self.gs:
                de = data[str(g)][0].ptp()
                results.append(de * 1000)
            self.results[name] = results


if __name__ == '__main__':
    task = EggboxTestTask()
    args = task.parse_args()
    task.run(args)
