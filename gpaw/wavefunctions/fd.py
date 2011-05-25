import numpy as np

from gpaw.overlap import Overlap
from gpaw.fd_operators import Laplace
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.utilities import unpack
from gpaw.io.tar import TarFileReference
from gpaw.lfc import BasisFunctions
from gpaw.utilities.blas import axpy
from gpaw.transformers import Transformer
from gpaw.fd_operators import Gradient
from gpaw.band_descriptor import BandDescriptor
from gpaw import extra_parameters
from gpaw.wavefunctions.fdpw import FDPWWaveFunctions
from gpaw.hs_operators import MatrixOperator
from gpaw.preconditioner import Preconditioner
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.kpoint import KPoint
from gpaw.mpi import serial_comm


class FDWaveFunctions(FDPWWaveFunctions):
    def __init__(self, stencil, diagksl, orthoksl, initksl,
                 gd, nvalence, setups, bd,
                 dtype, world, kd, timer=None):
        FDPWWaveFunctions.__init__(self, diagksl, orthoksl, initksl,
                                   gd, nvalence, setups, bd,
                                   dtype, world, kd, timer)

        self.wd = self.gd  # wave function descriptor
        
        # Kinetic energy operator:
        self.kin = Laplace(self.gd, -0.5, stencil, self.dtype, allocate=False)

        self.matrixoperator = MatrixOperator(orthoksl)

    def set_setups(self, setups):
        self.pt = LFC(self.gd, [setup.pt_j for setup in setups],
                      self.kpt_comm, dtype=self.dtype, forces=True)
        FDPWWaveFunctions.set_setups(self, setups)

    def set_positions(self, spos_ac):
        if not self.kin.is_allocated():
            self.kin.allocate()
        FDPWWaveFunctions.set_positions(self, spos_ac)

    def summary(self, fd):
        fd.write('Mode: Finite-difference\n')
        
    def make_preconditioner(self, block=1):
        return Preconditioner(self.gd, self.kin, self.dtype, block)
    
    def apply_pseudo_hamiltonian(self, kpt, hamiltonian, psit_xG, Htpsit_xG):
        self.kin.apply(psit_xG, Htpsit_xG, kpt.phase_cd)
        hamiltonian.apply_local_potential(psit_xG, Htpsit_xG, kpt.s)

    def add_orbital_density(self, nt_G, kpt, n):
        if self.dtype == float:
            axpy(1.0, kpt.psit_nG[n]**2, nt_G)
        else:
            axpy(1.0, kpt.psit_nG[n].real**2, nt_G)
            axpy(1.0, kpt.psit_nG[n].imag**2, nt_G)

    def add_to_density_from_k_point_with_occupation(self, nt_sG, kpt, f_n):
        # Used in calculation of response part of GLLB-potential
        nt_G = nt_sG[kpt.s]
        if self.dtype == float:
            for f, psit_G in zip(f_n, kpt.psit_nG):
                axpy(f, psit_G**2, nt_G)
        else:
            for f, psit_G in zip(f_n, kpt.psit_nG):
                axpy(f, psit_G.real**2, nt_G)
                axpy(f, psit_G.imag**2, nt_G)

        # Hack used in delta-scf calculations:
        if hasattr(kpt, 'c_on'):
            assert self.bd.comm.size == 1
            d_nn = np.zeros((self.bd.mynbands, self.bd.mynbands),
                            dtype=complex)
            for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                d_nn += ne * np.outer(c_n.conj(), c_n)
            for d_n, psi0_G in zip(d_nn, kpt.psit_nG):
                for d, psi_G in zip(d_n, kpt.psit_nG):
                    if abs(d) > 1.e-12:
                        nt_G += (psi0_G.conj() * d * psi_G).real

    def calculate_kinetic_energy_density(self, grad_v):
        assert not hasattr(self.kpt_u[0], 'c_on')
        if isinstance(self.kpt_u[0].psit_nG, TarFileReference):
            raise RuntimeError('Wavefunctions have not been initialized.')

        taut_sG = self.gd.zeros(self.nspins)
        dpsit_G = self.gd.empty(dtype=self.dtype)
        for kpt in self.kpt_u:
            for f, psit_G in zip(kpt.f_n, kpt.psit_nG):
                for v in range(3):
                    grad_v[v](psit_G, dpsit_G, kpt.phase_cd)
                    axpy(0.5 * f, abs(dpsit_G)**2, taut_sG[kpt.s])

        self.kpt_comm.sum(taut_sG)
        self.band_comm.sum(taut_sG)
        return taut_sG
        
    def calculate_forces(self, hamiltonian, F_av):
        # Calculate force-contribution from k-points:
        F_av.fill(0.0)
        F_aniv = self.pt.dict(self.bd.mynbands, derivative=True)
        for kpt in self.kpt_u:
            self.pt.derivative(kpt.psit_nG, F_aniv, kpt.q)
            for a, F_niv in F_aniv.items():
                F_niv = F_niv.conj()
                F_niv *= kpt.f_n[:, np.newaxis, np.newaxis]
                dH_ii = unpack(hamiltonian.dH_asp[a][kpt.s])
                P_ni = kpt.P_ani[a]
                F_vii = np.dot(np.dot(F_niv.transpose(), P_ni), dH_ii)
                F_niv *= kpt.eps_n[:, np.newaxis, np.newaxis]
                dO_ii = hamiltonian.setups[a].dO_ii
                F_vii -= np.dot(np.dot(F_niv.transpose(), P_ni), dO_ii)
                F_av[a] += 2 * F_vii.real.trace(0, 1, 2)

            # Hack used in delta-scf calculations:
            if hasattr(kpt, 'c_on'):
                assert self.bd.comm.size == 1
                self.pt.derivative(kpt.psit_nG, F_aniv, kpt.q) #XXX again
                d_nn = np.zeros((self.bd.mynbands, self.bd.mynbands),
                                dtype=complex)
                for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                    d_nn += ne * np.outer(c_n.conj(), c_n)
                for a, F_niv in F_aniv.items():
                    F_niv = F_niv.conj()
                    dH_ii = unpack(hamiltonian.dH_asp[a][kpt.s])
                    Q_ni = np.dot(d_nn, kpt.P_ani[a])
                    F_vii = np.dot(np.dot(F_niv.transpose(), Q_ni), dH_ii)
                    F_niv *= kpt.eps_n[:, np.newaxis, np.newaxis]
                    dO_ii = hamiltonian.setups[a].dO_ii
                    F_vii -= np.dot(np.dot(F_niv.transpose(), Q_ni), dO_ii)
                    F_av[a] += 2 * F_vii.real.trace(0, 1, 2)

        self.bd.comm.sum(F_av, 0)

        if self.bd.comm.rank == 0:
            self.kpt_comm.sum(F_av, 0)

    def ibz2bz(self, atoms):
        """Transform wave functions in IBZ to the full BZ."""

        assert self.kd.comm.size == 1

        # New k-point descriptor for full BZ:
        kd = KPointDescriptor(self.kd.bzk_kc, nspins=self.nspins)
        kd.set_symmetry(atoms, self.setups, None)
        kd.set_communicator(serial_comm)

        self.pt = LFC(self.gd, [setup.pt_j for setup in self.setups],
                      dtype=self.dtype)
        self.pt.set_k_points(kd.ibzk_kc)
        self.pt.set_positions(atoms.get_scaled_positions())

        self.initialize_wave_functions_from_restart_file()

        weight = 2.0 / kd.nspins / kd.nbzkpts
        
        # Build new list of k-points:
        kpt_u = []
        for s in range(self.nspins):
            for k in range(kd.nbzkpts):
                # Index of symmetry related point in the IBZ
                ik = self.kd.kibz_k[k]
                r, u = self.kd.get_rank_and_index(s, ik)
                assert r == 0
                kpt = self.kpt_u[u]
            
                phase_cd = np.exp(2j * np.pi * self.gd.sdisp_cd *
                                  kd.bzk_kc[k, :, np.newaxis])

                # New k-point:
                kpt2 = KPoint(weight, s, k, k, phase_cd)
                kpt2.f_n = kpt.f_n / kpt.weight / kd.nbzkpts * 2 / self.nspins
                kpt2.eps_n = kpt.eps_n.copy()
                
                # Transform wave functions using symmetry operation:
                Psit_nG = self.gd.collect(kpt.psit_nG)
                if Psit_nG is not None:
                    Psit_nG = Psit_nG.copy()
                    for Psit_G in Psit_nG:
                        Psit_G[:] = self.kd.transform_wave_function(Psit_G, k)
                kpt2.psit_nG = self.gd.empty(self.bd.nbands, dtype=self.dtype)
                self.gd.distribute(Psit_nG, kpt2.psit_nG)

                # Calculate PAW projections:
                kpt2.P_ani = self.pt.dict(len(kpt.psit_nG))
                self.pt.integrate(kpt2.psit_nG, kpt2.P_ani, k)
                
                kpt_u.append(kpt2)

        self.kd = kd
        self.kpt_u = kpt_u

    def estimate_memory(self, mem):
        FDPWWaveFunctions.estimate_memory(self, mem)
        self.kin.estimate_memory(mem.subnode('Kinetic operator'))
