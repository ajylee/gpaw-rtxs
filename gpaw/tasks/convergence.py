import optparse

import numpy as np
from ase.atoms import Atoms
from ase.tasks.task import Task
from ase.data import covalent_radii, atomic_numbers

from gpaw import FermiDirac


class ConvergenceTestTask(Task):
    def __init__(self, g1=20, g2=40, L=4.0, **kwargs):
        self.gs = range(g1, g2 + 1, 4)
        
        self.L = L

        Task.__init__(self, calcwrapper='gpaw', **kwargs)
        
        self.taskname = 'convergence'

    def build_system(self, name):
        atoms = Atoms(name * 2,
                      [(0, 0, 0), (1.0, 1.1, 1.2)],
                      pbc=True,
                      cell=(self.L, self.L, self.L))
        r = covalent_radii[atomic_numbers[name]]
        atoms.set_distance(0, 1, 2 * r, 0)
        
        return atoms

    def calculate(self, name, atoms):
        data = {}
        for g in self.gs:
            atoms.calc.set(gpts=(g, g, g),
                           occupations=FermiDirac(0.1),
                           kpts=[1, 1, 1])
            results = []
            e = atoms.get_potential_energy()
            data[g] = [e]

        del atoms[1]
        for g in self.gs:
            atoms.calc.set(gpts=(g, g, g),
                           occupations=FermiDirac(0.1),
                           kpts=[1, 1, 1])
            results = []
            e = atoms.get_potential_energy()
            data[g].append(e)

        return data

    def analyse(self):
        self.summary_header = [('name', '')] + [
            ('dE(h=%.2f)' % (self.L / g), 'meV') for g in self.gs]

        for name, data in self.data.items():
            self.results[name] = [1000 * (2 * data[g][1] - data[g][0])
                                  for g in self.gs]
            
    def add_options(self, parser):
        Task.add_options(self, parser)

        egg = optparse.OptionGroup(parser, 'Convergence')
        egg.add_option('-g', '--grid-point-range', default='20,40',
                       help='...')
        parser.add_option_group(egg)

    def parse(self, opts, args):
        Task.parse(self, opts, args)

        g1, g2 = (int(x) for x in opts.grid_point_range.split(','))
        self.gs = range(g1, g2 + 1, 4)


if __name__ == '__main__':
    task = ConvergenceTestTask()
    args = task.parse_args()
    task.run(args)
