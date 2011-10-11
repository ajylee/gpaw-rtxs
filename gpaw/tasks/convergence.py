import optparse

import numpy as np
from ase.atoms import Atoms
from ase.tasks.task import Task
from ase.io import string2index
from ase.data import covalent_radii

from gpaw import FermiDirac


class ConvergenceTestTask(Task):
    def __init__(self, g1=20, g2=40, L=4.0, **kwargs):
        self.gs = range(g1, g2 + 1, 4)
        
        self.L = L

        Task.__init__(self, calcwrapper='gpaw', **kwargs)
        
        self.taskname = 'convergence'

    def build_system(self, name):
        symbol = name.split('-')[0]
        return Atoms(symbol, pbc=True, cell=(self.L, self.L, self.L))

    def run(self, names):
        names = self.expand(names)
        newnames = []
        for g in self.gs:
            newnames.extend([name + '-' + str(g) for name in names])
        Task.run(self, newnames)

    def calculate(self, name, atoms):
        g = int(name.split('-')[1])
        atoms.calc.set(gpts=(g, g, g),
                       occupations=FermiDirac(0.1),
                       kpts=[1, 1, 1])
        e1 = atoms.get_potential_energy()
        
        atoms += atoms
        atoms[1].position = [1.0, 1.1, 1.2]
        r = covalent_radii[atoms[0].number]
        atoms.set_distance(0, 1, 2 * r, 0)

        e2 = atoms.get_potential_energy()
        
        return {'e1': e1, 'e2':e2}

    def analyse(self):
        self.summary_header = [('name', '')] + [
            ('dE(h=%.2f)' % (self.L / g), 'meV') for g in self.gs]

        for name, data in self.data.items():
            symbol, g = name.split('-')
            g = int(g)
            ea = 2 * data['e1'] - data['e2']
            if symbol not in self.results:
                self.results[symbol] = [None] * len(self.gs)
            self.results[symbol][self.gs.index(g)] = ea * 1000
            
    def add_options(self, parser):
        Task.add_options(self, parser)

        grp = optparse.OptionGroup(parser, self.taskname.title())
        grp.add_option('-g', '--grid-point-range', default='20:41:4',
                       help='...')
        parser.add_option_group(grp)

    def parse(self, opts, args):
        Task.parse(self, opts, args)

        self.gs = range(100)[string2index(opts.grid_point_range)]


if __name__ == '__main__':
    task = ConvergenceTestTask()
    args = task.parse_args()
    task.run(args)
