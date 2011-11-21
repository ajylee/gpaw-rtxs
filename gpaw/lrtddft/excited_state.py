"""Excited state as calculator object."""

from ase.units import Hartree
from ase.calculators.general import Calculator
from ase.calculators.test import numeric_forces
from gpaw import GPAW

class ExcitedState(Calculator):
    def __init__(self, lrtddft, index, d=0.001, txt=None):
        self.lrtddft = lrtddft
        self.calculator = self.lrtddft.calculator
        self.atoms = self.calculator.get_atoms()
        self.index = index
        self.d = d
        if txt is None:
            self.txt = self.lrtddft.txt
        
        print >> self.txt, 'ExcitedState', self.index
 
    def get_potential_energy(self, atoms=None):
        """Evaluate potential energy for the given excitation."""
        if atoms is not None:
            self.atoms = atoms
            self.update()
        return self.energy

    def update(self):
        E0 = self.calculator.get_potential_energy(self.atoms)
        lr = self.lrtddft
        self.lrtddft.forced_update()
        self.lrtddft.diagonalize()
        self.energy = E0 + self.lrtddft[self.index].energy * Hartree

    def get_forces(self, atoms):
        """Get finite-difference forces"""
        atoms.set_calculator(self)
        forces = numeric_forces(atoms, d=self.d)
        if self.txt:
            print >> self.txt, 'Excited state forces in eV/Ang:'
            symbols = self.atoms.get_chemical_symbols()
            for a, symbol in enumerate(symbols):
                print >> self.txt, ('%3d %-2s %10.5f %10.5f %10.5f' %
                                    ((a, symbol) + tuple(forces[a])))
        return forces

    def get_stress(self, atoms):
        """Return the stress for the current state of the Atoms."""
        raise NotImplementedError
