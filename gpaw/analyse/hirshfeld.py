import numpy as np

from ase import Atoms
from ase.units import Bohr
from gpaw.density import RealSpaceDensity
from gpaw.lfc import BasisFunctions
from gpaw.mixer import Mixer
from gpaw.setup import Setups
from gpaw.xc import XC
from gpaw.utilities.tools import coordinates

from gpaw.mpi import rank

class HirshfeldDensity(RealSpaceDensity):
    """Density as sum of atomic densities."""
    def __init__(self, calculator):
        self.calculator = calculator
        density = calculator.density
        par = self.calculator.input_parameters
        RealSpaceDensity.__init__(self, density.gd, density.finegd, 1, 0,
                                  stencil=par.stencils[1])

    def get_density(self, atom_indicees=None):
        """Get sum of atomic densities from the given atom list.

        All atoms are taken if the list is not given."""
 
        all_atoms = self.calculator.get_atoms()
        if atom_indicees is None:
            atom_indicees = range(len(all_atoms))

        density = self.calculator.density
        density.set_positions(all_atoms.get_scaled_positions() % 1.0,
                              self.calculator.wfs.rank_a)

        # select atoms
        atoms = []
        D_asp = {}
        rank_a = []
        all_D_asp = self.calculator.density.D_asp
        all_rank_a = self.calculator.density.rank_a
        for a in atom_indicees:
            if a in all_D_asp:
                D_asp[len(atoms)] = all_D_asp.get(a)
            atoms.append(all_atoms[a])
            rank_a.append(all_rank_a[a])
        atoms = Atoms(atoms, cell=all_atoms.get_cell())
        spos_ac = atoms.get_scaled_positions() % 1.0
        Z_a = atoms.get_atomic_numbers()

        par = self.calculator.input_parameters
        setups = Setups(Z_a, par.setups, par.basis, par.lmax, 
                        XC(par.xc), 
                        self.calculator.wfs.world)
        self.D_asp = D_asp

        # initialize 
        self.initialize(setups, 
                        self.calculator.timer,
                        np.zeros((len(atoms), 3)), False)
        self.set_mixer(None)
        self.set_positions(spos_ac, rank_a)
        basis_functions = BasisFunctions(self.gd,
                                         [setup.phit_j
                                          for setup in self.setups],
                                         cut=True)
        basis_functions.set_positions(spos_ac)
        self.initialize_from_atomic_densities(basis_functions)

        aed_sg, gd = self.get_all_electron_density(atoms, 
                                                   gridrefinement=2)

        return aed_sg[0], gd

class HirshfeldPartitioning:
    """Partion space according to the Hirshfeld method.

    After: F. L. Hirshfeld Theoret. Chim.Acta 44 (1977) 129-138
    """
    def __init__(self, calculator, density_cutoff=1.e-12):
        self.calculator = calculator
        self.density_cutoff = density_cutoff

    def initialize(self):
        self.atoms = self.calculator.get_atoms()
        self.hdensity = HirshfeldDensity(self.calculator)
        density_g, gd = self.hdensity.get_density()
        self.invweight_g = 0. * density_g
        density_ok = np.where(density_g > self.density_cutoff)
        self.invweight_g[density_ok] = 1.0 / density_g[density_ok]

    def get_calculator(self):
        return self.calculator
    
    def get_effective_volume_ratio(self, atom_index):
        """Effective volume to free volume ratio.

        After: Tkatchenko and Scheffler PRL 102 (2009) 073005, eq. (7)
        """
        atoms = self.atoms
        finegd = self.calculator.density.finegd

        den_g, gd = self.calculator.density.get_all_electron_density(atoms)
        den_g = den_g.sum(axis=0)
        assert(gd == finegd)
        denfree_g, gd = self.hdensity.get_density([atom_index])
        assert(gd == finegd)

        # the atoms r^3 grid
        position = self.atoms[atom_index].position / Bohr
        r_vg, r2_g = coordinates(finegd, origin=position)
        r3_g = r2_g * np.sqrt(r2_g)

        weight_g = denfree_g * self.invweight_g

        nom = finegd.integrate(r3_g * den_g * weight_g)
        denom = finegd.integrate(r3_g * denfree_g)

        return nom / denom

    def get_weight(self, atom_index):
        denfree_g, gd = self.hdensity.get_density([atom_index])
        weight_g = denfree_g * self.invweight_g
        return weight_g

    def get_charges(self):
        """Charge on the atom according to the Hirshfeld partitioning"""
        self.initialize()
        finegd = self.calculator.density.finegd
        
        den_g, gd = self.calculator.density.get_all_electron_density(self.atoms)
        den_g = den_g.sum(axis=0)

        charges = []
        for ia, atom in enumerate(self.atoms):
            weight_g = self.get_weight(ia)
            charge = atom.number - finegd.integrate(weight_g * den_g)
            charges.append(atom.number - finegd.integrate(weight_g * den_g))
        return charges

    def get_effective_volume_ratios(self):
        """Return the list of effective volume to free volume ratios."""
        self.initialize()
        ratios = []
        for a, atom in enumerate(self.atoms):
            ratios.append(self.get_effective_volume_ratio(a))
        return np.array(ratios)
        
