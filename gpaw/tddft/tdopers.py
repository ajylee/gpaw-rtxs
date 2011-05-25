# Written by Lauri Lehtovaara, 2007

"""This module implements classes for time-dependent variables and
operators."""

import numpy as np

from gpaw.external_potential import ExternalPotential
from gpaw.utilities import pack2, unpack
from gpaw.mpi import run
from gpaw.fd_operators import Laplace, Gradient
from gpaw.tddft.abc import *

# Hamiltonian
class TimeDependentHamiltonian:
    """Time-dependent Hamiltonian, H(t)
    
    This class contains information required to apply time-dependent
    Hamiltonian to a wavefunction.
    """
    
    def __init__(self, wfs, atoms, hamiltonian, td_potential):
        """Create the TimeDependentHamiltonian-object.
        
        The time-dependent potential object must (be None or) have a member
        function strength(self,time), which provides the strength of the
        time-dependent external potential to x-direction at the given time.
        
        Parameters
        ----------
        wfs: FDWaveFunctions
            time-independent grid-based wavefunctions
        hamiltonian: Hamiltonian
            time-independent Hamiltonian
        td_potential: TimeDependentPotential
            time-dependent potential
        """

        self.wfs = wfs
        self.hamiltonian = hamiltonian
        self.td_potential = td_potential
        self.time = self.old_time = 0
        
        # internal smooth potential
        self.vt_sG = hamiltonian.gd.zeros(hamiltonian.nspins)

        # Increase the accuracy of Poisson solver
        self.hamiltonian.poisson.eps = 1e-12

        # external potential
        #if hamiltonian.vext_g is None:
        #    hamiltonian.vext_g = hamiltonian.finegd.zeros()

        #self.ti_vext_g = hamiltonian.vext_g
        #self.td_vext_g = hamiltonian.finegd.zeros(n=hamiltonian.nspins)

        self.P = None

        self.spos_ac = atoms.get_scaled_positions() % 1.0
        self.absorbing_boundary = None
        

    def update(self, density, time):
        """Updates the time-dependent Hamiltonian.
    
        Parameters
        ----------
        density: Density
            the density at the given time  
            (TimeDependentDensity.get_density())
        time: float
            the current time

        """

        self.old_time = self.time = time
        self.hamiltonian.update(density)
        
    def half_update(self, density, time):
        """Updates the time-dependent Hamiltonian, in such a way, that a
        half of the old Hamiltonian is kept and the other half is updated.
        
        Parameters
        ----------
        density: Density
            the density at the given time 
            (TimeDependentDensity.get_density())
        time: float
            the current time

        """
        
        self.old_time = self.time
        self.time = time

        # copy old
        self.vt_sG[:] = self.hamiltonian.vt_sG
        self.dH_asp = {}
        for a, dH_sp in self.hamiltonian.dH_asp.items():
            self.dH_asp[a] = dH_sp.copy()
        # update
        self.hamiltonian.update(density)
        # average and difference
        self.hamiltonian.vt_sG[:], self.vt_sG[:] = \
            0.5*(self.hamiltonian.vt_sG + self.vt_sG), \
            self.hamiltonian.vt_sG - self.vt_sG
        for a, dH_sp in self.hamiltonian.dH_asp.items():
            dH_sp[:], self.dH_asp[a][:] = 0.5*(dH_sp + self.dH_asp[a]), \
                dH_sp - self.dH_asp[a] #pack/unpack is linear for real values

    def half_apply_local_potential(self, psit_nG, Htpsit_nG, s):
        """Apply the half-difference Hamiltonian operator to a set of vectors.
        
        Parameters:

        psit_nG: ndarray
            set of vectors to which the overlap operator is applied.
        psit_nG: ndarray, output
            resulting H applied to psit_nG vectors.
        s: int
            spin index of k-point object defined in kpoint.py.
        
        """
        # Does exactly the same as Hamiltonian.apply_local_potential
        # but uses the difference between vt_sG at time t and t+dt.
        vt_G = self.vt_sG[s]
        if psit_nG.ndim == 3:
            Htpsit_nG += psit_nG * vt_G
        else:
            for psit_G, Htpsit_G in zip(psit_nG, Htpsit_nG):
                Htpsit_G += psit_G * vt_G


    def half_apply(self, kpt, psit, hpsit, calculate_P_ani=True):
        """Applies the half-difference of the time-dependent Hamiltonian
        to the wavefunction psit of the k-point kpt.
        
        Parameters
        ----------
        kpt: Kpoint
            the current k-point (kpt_u[index_of_k-point])
        psit: List of coarse grid
            the wavefuntions (on coarse grid) 
            (kpt_u[index_of_k-point].psit_nG[indices_of_wavefunc])
        hpsit: List of coarse grid
            the resulting "operated wavefunctions" (H psit)
        calculate_P_ani: bool
            When True, the integrals of projector times vectors
            P_ni = <p_i | psit> are calculated.
            When False, existing P_uni are used

        """

        hpsit.fill(0.0)
        self.half_apply_local_potential(psit, hpsit, kpt.s)

        # Does exactly the same as last part of Hamiltonian.apply but
        # uses the difference between dH_asp at time t and t+dt.
        shape = psit.shape[:-3]
        P_axi = self.wfs.pt.dict(shape)

        if calculate_P_ani:
            self.wfs.pt.integrate(psit, P_axi, kpt.q)
        else:
            for a, P_ni in kpt.P_ani.items():
                P_axi[a][:] = P_ni

        for a, P_xi in P_axi.items():
            dH_ii = unpack(self.dH_asp[a][kpt.s])
            P_axi[a][:] = np.dot(P_xi, dH_ii)
        self.wfs.pt.add(hpsit, P_axi, kpt.q)

        if self.td_potential is not None:
            # FIXME: add half difference here... but maybe it's not important
            # as this will be used only for getting initial guess. So, should
            # not affect to the results, only to the speed of convergence.
            #raise NotImplementedError
            pass

    def apply(self, kpt, psit, hpsit, calculate_P_ani=True):
        """Applies the time-dependent Hamiltonian to the wavefunction psit of
        the k-point kpt.
        
        Parameters
        ----------
        kpt: Kpoint
            the current k-point (kpt_u[index_of_k-point])
        psit: List of coarse grid
            the wavefuntions (on coarse grid) 
            (kpt_u[index_of_k-point].psit_nG[indices_of_wavefunc])
        hpsit: List of coarse grid
            the resulting "operated wavefunctions" (H psit)
        calculate_P_ani: bool
            When True, the integrals of projector times vectors
            P_ni = <p_i | psit> are calculated.
            When False, existing P_uni are used

        """

        self.hamiltonian.apply(psit, hpsit, self.wfs, kpt, calculate_P_ani)

        # PAW correction
        if self.P is not None:
            self.P.add(psit, hpsit, self.wfs, kpt)

        # Absorbing boundary conditions

        # Imaginary potential
        if self.absorbing_boundary is not None \
               and self.absorbing_boundary.type == 'IPOT':
            hpsit[:] += self.absorbing_boundary.get_potential_matrix() * psit

        # Perfectly matched layers
        if self.absorbing_boundary is not None \
               and self.absorbing_boundary.type == 'PML':
            # Perfectly matched layer is applied as potential Vpml = Tpml-T
            # Where  T = -0.5*\nabla^{2}\psi  (Use latex for these equations)
            # See abc.py for details
            # This is probably not the most optimal approach and slows
            # the propagation.
            if self.lpsit is None:
                self.lpsit = self.hamiltonian.gd.empty( n=len(psit),
                                                        dtype=complex )
            self.laplace.apply(psit, self.lpsit, kpt.phase_cd)
            hpsit[:] -= (.5 * (self.absorbing_boundary.get_G()**2 - 1.0)
                         * self.lpsit)
            for i in range(3):
                self.gradient[i].apply(psit, self.lpsit, kpt.phase_cd)
                hpsit[:] -= (.5 * self.absorbing_boundary.get_G()
                             * self.absorbing_boundary.get_dG()[i]
                             * self.lpsit)


        # Time-dependent dipole field
        if self.td_potential is not None:
            #TODO on shaky ground here...
            strength = self.td_potential.strength
            ExternalPotential().add_linear_field(self.wfs, self.spos_ac,
                                                 psit, hpsit,
                                                 0.5 * strength(self.time) +
                                                 0.5 * strength(self.old_time),
                                                 kpt)

            
    def set_absorbing_boundary(self, absorbing_boundary):
        """ Sets up the absorbing boundary.            
            Parameters:
            absorbing_boundary: absorbing boundary object of any kind.  
        """
        
        self.absorbing_boundary = absorbing_boundary
        self.absorbing_boundary.set_up(self.hamiltonian.gd)
        if self.absorbing_boundary.type == 'PML':
            gd = self.hamiltonian.gd
            self.laplace = Laplace(gd, n=2, dtype=complex)
            self.gradient = np.array((Gradient(gd,0, n=2, dtype=complex),
                                       Gradient(gd,1, n=2, dtype=complex),
                                       Gradient(gd,2, n=2, dtype=complex)))
            self.lpsit=None


# AbsorptionKickHamiltonian
class AbsorptionKickHamiltonian:
    """Absorption kick Hamiltonian, p.r
    
    This class contains information required to apply absorption kick
    Hamiltonian to a wavefunction.
    """
    
    def __init__(self, wfs, atoms, strength=[0.0, 0.0, 1e-3]):
        """Create the AbsorptionKickHamiltonian-object.

        Parameters
        ----------
        wfs: FDWaveFunctions
            time-independent grid-based wavefunctions
        atoms: Atoms
            list of atoms
        strength: float[3]
            strength of the delta field to different directions

        """

        self.wfs = wfs
        self.spos_ac = atoms.get_scaled_positions() % 1.0
        
        # magnitude
        magnitude = np.sqrt(strength[0]*strength[0] 
                             + strength[1]*strength[1] 
                             + strength[2]*strength[2])
        # iterations
        self.iterations = int(round(magnitude / 1.0e-4))
        if self.iterations < 1:
            self.iterations = 1
        # delta p
        self.dp = strength / self.iterations

        # hamiltonian
        self.abs_hamiltonian = np.array([self.dp[0], self.dp[1], self.dp[2]])
        

    def update(self, density, time):
        """Dummy function = does nothing. Required to have correct interface.
        
        Parameters
        ----------
        density: Density or None
            the density at the given time or None (ignored)
        time: Float or None
            the current time (ignored)

        """
        pass
        
    def half_update(self, density, time):
        """Dummy function = does nothing. Required to have correct interface.
        
        Parameters
        ----------
        density: Density or None
            the density at the given time or None (ignored)
        time: float or None
            the current time (ignored)

        """
        pass
        
    def apply(self, kpt, psit, hpsit, calculate_P_ani=True):
        """Applies the absorption kick Hamiltonian to the wavefunction psit of
        the k-point kpt.
        
        Parameters
        ----------
        kpt: Kpoint
            the current k-point (kpt_u[index_of_k-point])
        psit: List of coarse grids
            the wavefuntions (on coarse grid) 
            (kpt_u[index_of_k-point].psit_nG[indices_of_wavefunc])
        hpsit: List of coarse grids
            the resulting "operated wavefunctions" (H psit)
        calculate_P_ani: bool
            When True, the integrals of projector times vectors
            P_ni = <p_i | psit> are calculated.
            When False, existing P_uni are used

        """
        hpsit[:] = 0.0

        #TODO on shaky ground here...
        ExternalPotential().add_linear_field(self.wfs, self.spos_ac,
                                             psit, hpsit,
                                             self.abs_hamiltonian, kpt)


# Overlap
class TimeDependentOverlap:
    """Time-dependent overlap operator S(t)
    
    This class contains information required to apply time-dependent
    overlap operator to a wavefunction.
    """
    
    def __init__(self, wfs):
        """Creates the TimeDependentOverlap-object.
        
        Parameters
        ----------
        wfs: FDWaveFunctions
            time-independent grid-based wavefunctions

        """
        self.wfs = wfs
        self.overlap = wfs.overlap

    def update_k_point_projections(self, kpt, psit=None):
        """Updates the projector function overlap integrals
        with the wavefunctions of a given k-point.
        
        Parameters
        ----------
        kpt: Kpoint
            the current k-point (kpt_u[index_of_k-point])
        psit: List of coarse grids (optional)
            the wavefuntions (on coarse grid) 
            (kpt_u[index_of_k-point].psit_nG[indices_of_wavefunc])

        """
        if psit is not None:
            self.wfs.pt.integrate(psit, kpt.P_ani, kpt.q)
        else:
            self.wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

    def update(self):
        """Updates the time-dependent overlap operator.
        
        Parameters
        ----------
        None

        """
        for kpt in self.wfs.kpt_u:
            self.update_k_point_projections(kpt)
    
    def half_update(self):
        """Updates the time-dependent overlap operator, in such a way,
        that a half of the old overlap operator is kept and the other half
        is updated. !Currently does nothing!

        Parameters
        ----------
        None

        """
        #for kpt in self.wfs.kpt_u:
        #    # copy old
        #    P_ani = {}
        #    for a,P_ni in kpt.P_ani.items():
        #        P_ani[a] = P_ni.copy()
        #    # update
        #    self.update_k_point_projections(kpt)
        #    # average
        #    for a,P_ni in P_ani.items():
        #        kpt.P_ani[a] += P_ni
        #        kpt.P_ani[a] *= .5

        # !!! FIX ME !!! update overlap operator/projectors/...
        pass
    
    def apply(self, kpt, psit, spsit, calculate_P_ani=True):
        """Apply the time-dependent overlap operator to the wavefunction
        psit of the k-point kpt.
        
        Parameters
        ----------
        kpt: Kpoint
            the current k-point (kpt_u[index_of_k-point])
        psit: List of coarse grids
            the wavefuntions (on coarse grid) 
            (kpt_u[index_of_k-point].psit_nG[indices_of_wavefunc])
        spsit: List of coarse grids
            the resulting "operated wavefunctions" (S psit)
        calculate_P_ani: bool
            When True, the integrals of projector times vectors
            P_ni = <p_i | psit> are calculated.
            When False, existing P_ani are used

        """
        self.overlap.apply(psit, spsit, self.wfs, kpt, calculate_P_ani)

    def apply_inverse(self, kpt, psit, sinvpsit, calculate_P_ani=True):
        """Apply the approximative time-dependent inverse overlap operator
        to the wavefunction psit of the k-point kpt.

        Parameters
        ----------
        kpt: Kpoint
            the current k-point (kpt_u[index_of_k-point])
        psit: List of coarse grids
            the wavefuntions (on coarse grid) 
            (kpt_u[index_of_k-point].psit_nG[indices_of_wavefunc])
        sinvpsit: List of coarse grids
            the resulting "operated wavefunctions" (S^(-1) psit)
        calculate_P_ani: bool
            When True, the integrals of projector times vectors
            P_ni = <p_i | psit> are calculated.
            When False, existing P_uni are used

        """
        self.overlap.apply_inverse(psit, sinvpsit, self.wfs, kpt,
                                   calculate_P_ani)


# DummyDensity
class DummyDensity:
    """Implements dummy (= does nothing) density for AbsorptionKick."""

    def __init__(self, wfs):
        """Placeholder Density object for AbsorptionKick.

        Parameters
        ----------
        wfs: FDWaveFunctions
            time-independent grid-based wavefunctions

        """
        self.wfs = wfs

    def update(self):
        pass

    def get_wavefunctions(self):
        return self.wfs

    def get_density(self):
        return None


# Density
class TimeDependentDensity(DummyDensity):
    """Time-dependent density rho(t)
    
    This class contains information required to get the time-dependent
    density.
    """
    
    def __init__(self, paw):
        """Creates the TimeDependentDensity-object.
        
        Parameters
        ----------
        paw: PAW
            the PAW-object
        """
        DummyDensity.__init__(self, paw.wfs)
        self.density = paw.density

    def update(self):
        """Updates the time-dependent density.
        
        Parameters
        ----------
        None

        """
        #for kpt in self.wfs.kpt_u:
        #    self.wfs.pt.integrate(kpt.psit_nG, kpt.P_ani)
        self.density.update(self.wfs)
       
    def get_density(self):
        """Returns the current density.
        
        Parameters
        ----------
        None

        """
        return self.density
