"""Excited state as calculator object."""

from ase.units import Hartree
from ase.calculators.general import Calculator
from ase.calculators.test import numeric_forces
from gpaw import GPAW
from gpaw.output import initialize_text_stream
from gpaw.mpi import rank

class ExcitedState(Calculator):
    def __init__(self, lrtddft, index, d=0.001, txt=None,
                 parallel=None):
        """ExcitedState object.

        parallel: Can be used to parallelize the numerical force calculation over
        images.
        """
        self.lrtddft = lrtddft
        self.calculator = self.lrtddft.calculator
        self.atoms = self.calculator.get_atoms()
        if type(index) == type(1):
            self.index = UnconstraintIndex(index)
        else:
            self.index = index
        self.d = d
        if txt is None:
            self.txt = self.lrtddft.txt
        else:
            rank = self.calculator.wfs.world.rank
            self.txt, firsttime = initialize_text_stream(txt, rank)
                                                              
        self.parallel = parallel

        self.energy = None
        self.forces = None
        
        print >> self.txt, 'ExcitedState', self.index
 
    def get_potential_energy(self, atoms=None):
        """Evaluate potential energy for the given excitation."""
        if atoms is None:
            atoms = self.atoms
            self.energy = self.calculate(atoms)
        if (self.energy is None) or atoms != self.atoms:  
            energy = self.calculate(atoms)
            return energy
        else:
            return self.energy

    def calculate(self, atoms):
        E0 = self.calculator.get_potential_energy(atoms)
        lr = self.lrtddft
        self.lrtddft.forced_update()
        self.lrtddft.diagonalize()
        index = self.index.apply(self.lrtddft)
        return E0 + self.lrtddft[index].energy * Hartree

    def get_forces(self, atoms):
        """Get finite-difference forces"""
        if (self.forces is None) or atoms != self.atoms:
            atoms.set_calculator(self)
            self.forces = numeric_forces(atoms, d=self.d, 
                                         parallel=self.parallel)
            if self.txt:
                print >> self.txt, 'Excited state forces in eV/Ang:'
                symbols = self.atoms.get_chemical_symbols()
                for a, symbol in enumerate(symbols):
                    print >> self.txt, ('%3d %-2s %10.5f %10.5f %10.5f' %
                                        ((a, symbol) + tuple(self.forces[a])))
        return self.forces

    def get_stress(self, atoms):
        """Return the stress for the current state of the Atoms."""
        raise NotImplementedError

    def set(self, **kwargs):
        self.calculator.set(**kwargs)

class UnconstraintIndex:
    def __init__(self, index):
        assert(type(index) == type(1))
        self.index = index
    def apply(self, *argv):
        return self.index

class MinimalOSIndex:
    """
    Constraint on minimal oscillator strength.

    direction:
        None: averaged (default)
        0, 1, 2: x, y, z
    """
    def __init__(self, fmin=0.02, direction=None):
        self.fmin = fmin
        self.direction = direction

    def apply(self, lrtddft):
        index = None
        i = 0
        fmax = 0.
        while i < len(lrtddft):
            ex = lrtddft[i]
            idir = 0
            if self.direction is not None:
                idir = 1 + self.direction
            f = ex.get_oscillator_strength()[idir]
            fmax = max(f, fmax)
            if f > self.fmin:
                return i
            i += 1
        error = 'The intensity constraint |f| > ' + str(self.fmin) + ' '
        error += 'can not be satisfied (max(f) = ' + str(fmax) + ').'
        raise RuntimeError(error)
        
