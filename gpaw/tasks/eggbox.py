import optparse

import numpy as np
from ase.tasks.task import Task
from ase.atoms import Atoms

from gpaw import FermiDirac


class EggboxTestTask(Task):
    def __init__(self, g1=20, g2=40, L=4.0, **kwargs):
        self.gs = range(g1, g2 + 1, 4)
        
        self.L = L

        Task.__init__(self, calcwrapper='gpaw', **kwargs)
        
        self.taskname = 'eggbox'

    def build_system(self, name):
        atoms = Atoms(name, pbc=True, cell=(self.L, self.L, self.L))
        return atoms

    def calculate(self, name, atoms):
        data = {}
        for g in self.gs:
            atoms.calc.set(gpts=(g, g, g),
                           occupations=FermiDirac(0.1),
                           kpts=[1, 1, 1])
            results = []
            for i in range(25):
                x = self.L / g * i / 48
                atoms[0].x = x
                e = atoms.calc.get_potential_energy(atoms,
                                                    force_consistent=True)
                f = atoms.get_forces()[0, 0]
                results.append((e, f))
            data[g] = results
        return data

    def analyse(self):
        self.summary_header = [('name', '')] + [
            ('dE(h=%.2f)' % (self.L / g), 'meV') for g in self.gs]

        for name, data in self.data.items():
            self.results[name] = [1000 * data[g][:, 0].ptp() for g in self.gs]
            
    def add_options(self, parser):
        Task.add_options(self, parser)

        egg = optparse.OptionGroup(parser, 'Eggbox')
        egg.add_option('-g', '--grid-point-range', default='20,40',
                       help='...')
        parser.add_option_group(egg)

    def parse(self, opts, args):
        Task.parse(self, opts, args)

        g1, g2 = (int(x) for x in opts.grid_point_range.split(','))
        self.gs = range(g1, g2 + 1, 4)


if __name__ == '__main__':
    task = EggboxTestTask()
    args = task.parse_args()
    task.run(args)
