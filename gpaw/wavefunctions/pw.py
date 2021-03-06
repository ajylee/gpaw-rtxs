# -*- coding: utf-8 -*-
from math import pi

import numpy as np
import ase.units as units

from gpaw.lfc import BaseLFC, LocalizedFunctionsCollection as LFC
from gpaw.wavefunctions.fdpw import FDPWWaveFunctions
from gpaw.hs_operators import MatrixOperator
import gpaw.fftw as fftw
from gpaw.lcao.overlap import fbt
from gpaw.spline import Spline
from gpaw.spherical_harmonics import Y
from gpaw.utilities import unpack, _fact as fac
from gpaw.utilities.blas import rk, r2k, gemm
from gpaw.density import Density
from gpaw.hamiltonian import Hamiltonian
from gpaw.blacs import BlacsGrid, BlacsDescriptor, Redistributor
from gpaw.matrix_descriptor import MatrixDescriptor
from gpaw.band_descriptor import BandDescriptor


class PWDescriptor:
    def __init__(self, ecut, gd, dtype=float, fftwflags=fftw.FFTW_MEASURE):

        assert gd.pbc_c.all() and gd.comm.size == 1

        self.ecut = ecut
        self.gd = gd

        N_c = gd.N_c
        self.comm = gd.comm

        assert 0.5 * pi**2 / (gd.h_cv**2).sum(1).max() >= ecut

        self.dtype = dtype

        if dtype == float:
            Nr_c = N_c.copy()
            Nr_c[2] = N_c[2] // 2 + 1
            i_Qc = np.indices(Nr_c).transpose((1, 2, 3, 0))
            i_Qc[..., :2] += N_c[:2] // 2
            i_Qc[..., :2] %= N_c[:2]
            i_Qc[..., :2] -= N_c[:2] // 2
            self.tmp_Q = fftw.empty(Nr_c, complex)
            self.tmp_R = self.tmp_Q.view(float)[:, :, :N_c[2]]
        else:
            i_Qc = np.indices(N_c).transpose((1, 2, 3, 0))
            i_Qc += N_c // 2
            i_Qc %= N_c
            i_Qc -= N_c // 2
            self.tmp_Q = fftw.empty(N_c, complex)
            self.tmp_R = self.tmp_Q

        self.fftplan = fftw.FFTPlan(self.tmp_R, self.tmp_Q, -1, fftwflags)
        self.ifftplan = fftw.FFTPlan(self.tmp_Q, self.tmp_R, 1, fftwflags)

        # Calculate reciprocal lattice vectors:
        B_cv = 2.0 * pi * gd.icell_cv
        i_Qc.shape = (-1, 3)
        G_Qv = np.dot(i_Qc, B_cv)
        G2_Q = (G_Qv**2).sum(axis=1)

        # Map from vectors inside sphere to fft grid:
        mask_Q = (G2_Q <= 2 * ecut)
        if self.dtype == float:
            mask_Q &= ((i_Qc[:, 2] > 0) |
                       (i_Qc[:, 1] > 0) |
                       ((i_Qc[:, 0] >= 0) & (i_Qc[:, 1] == 0)))
        self.Q_G = np.arange(len(G2_Q))[mask_Q]
        self.G_Gv = G_Qv[self.Q_G]

        self.n_c = self.Q_G #??????? # used by hs_operators.py XXX
        self.i1bzk_qc = []

    def g2(self, i1bzk_qc):
        # Did we already do this one?
        if (len(self.i1bzk_qc) == len(i1bzk_qc) and
            (self.i1bzk_qc == i1bzk_qc).all()):
            return self.G2_qG

        # No.
        self.i1bzk_qc = i1bzk_qc
        B_cv = 2.0 * pi * self.gd.icell_cv
        K_qv = np.dot(i1bzk_qc, B_cv)
        self.G2_qG = np.zeros((len(K_qv), len(self.Q_G)))
        for q, K_v in enumerate(K_qv):
            self.G2_qG[q] = ((self.G_Gv + K_v)**2).sum(1)
        return self.G2_qG

    def estimate_memory(self, mem):
        mem.subnode('Arrays',
                    self.Q_G.nbytes +
                    self.G_Gv.nbytes +
                    self.tmp_R.nbytes)

    def __len__(self):
        return len(self.Q_G)

    def bytecount(self, dtype=float):
        return self.Q_G.nbytes
    
    def zeros(self, x=(), dtype=None):
        a_xG = self.empty(x, dtype)
        a_xG.fill(0.0)
        return a_xG
    
    def empty(self, x=(), dtype=None):
        if dtype is not None:
            assert dtype == self.dtype
        if isinstance(x, int):
            x = (x,)
        shape = x + self.Q_G.shape
        return np.empty(shape, complex)
    
    def fft(self, f_R):
        """Fast Fourier transform.

        Returns c(G) for G<Gc::
  
                   __
                  \        -iG.R
            c(G) = ) f(R) e
                  /__
                   R
        """

        self.tmp_R[:] = f_R
        self.fftplan.execute()
        return self.tmp_Q.ravel()[self.Q_G]

    def ifft(self, c_G):
        """Inverse fast Fourier transform.

        Returns::
  
                      __
                   1 \        iG.R
            f(R) = -  ) c(G) e
                   N /__
                      G
        """

        self.tmp_Q[:] = 0.0
        self.tmp_Q.ravel()[self.Q_G] = c_G
        if self.dtype == float:
            t = self.tmp_Q[:, :, 0]
            n, m = self.gd.N_c[:2] // 2 - 1
            t[0,      -m:] = t[0,      m:0:-1].conj()
            t[n:0:-1, -m:] = t[-n:,    m:0:-1].conj()
            t[-n:,    -m:] = t[n:0:-1, m:0:-1].conj()
            t[-n:,    0  ] = t[n:0:-1, 0     ].conj()
        self.ifftplan.execute()
        return self.tmp_R * (1.0 / self.tmp_R.size)

    def integrate(self, a_xg, b_yg=None,
                  global_integral=True, hermitian=False,
                  _transposed_result=None):
        """Integrate function(s) over domain.

        a_xg: ndarray
            Function(s) to be integrated.
        b_yg: ndarray
            If present, integrate a_xg.conj() * b_yg.
        global_integral: bool
            If the array(s) are distributed over several domains, then the
            total sum will be returned.  To get the local contribution
            only, use global_integral=False.
        hermitian: bool
            Result is hermitian.
        _transposed_result: ndarray
            Long story.  Don't use this unless you are a method of the
            MatrixOperator class ..."""
        
        if b_yg is None:
            # Only one array:
            assert self.dtype == float
            return a_xg[..., 0].real * self.gd.dv

        A_xg = a_xg.reshape((-1, len(self)))
        B_yg = b_yg.reshape((-1, len(self)))

        alpha = self.gd.dv / self.gd.N_c.prod()

        if self.dtype == float:
            alpha *= 2
            A_xg = A_xg.view(float)
            B_yg = B_yg.view(float)

        if _transposed_result is None:
            result_yx = np.zeros((len(B_yg), len(A_xg)), self.dtype)
        else:
            result_yx = _transposed_result

        if a_xg is b_yg:
            rk(alpha, A_xg, 0.0, result_yx)
        elif hermitian:
            r2k(0.5 * alpha, A_xg, B_yg, 0.0, result_yx)
        else:
            gemm(alpha, A_xg, B_yg, 0.0, result_yx, 'c')
        
        if self.dtype == float:
            correction_yx = np.outer(B_yg[:, 0], A_xg[:, 0])
            if hermitian:
                result_yx -= 0.25 * alpha * (correction_yx + correction_yx.T)
            else:
                result_yx -= 0.5 * alpha * correction_yx

        xshape = a_xg.shape[:-1]
        yshape = b_yg.shape[:-1]
        result = result_yx.T.reshape(xshape + yshape)
        
        if result.ndim == 0:
            return result.item()
        else:
            return result

    def interpolate(self, a_R, pd):
        a_Q = self.tmp_Q
        b_Q = pd.tmp_Q

        e0, e1, e2 = 1 - self.gd.N_c % 2  # even or odd size
        a0, a1, a2 = pd.gd.N_c // 2 - self.gd.N_c // 2
        b0, b1, b2 = self.gd.N_c + (a0, a1, a2)

        if self.dtype == float:
            b2 = (b2 - a2) // 2 + 1
            a2 = 0
            axes = (0, 1)
        else:
            axes = (0, 1, 2)

        self.tmp_R[:] = a_R
        self.fftplan.execute()
        b_Q[:] = 0.0
        b_Q[a0:b0, a1:b1, a2:b2] = np.fft.fftshift(a_Q, axes=axes)

        if e0:
            b_Q[a0, a1:b1, a2:b2] *= 0.5
            b_Q[b0, a1:b1, a2:b2] = b_Q[a0, a1:b1, a2:b2]
            b0 += 1
        if e1:
            b_Q[a0:b0, a1, a2:b2] *= 0.5
            b_Q[a0:b0, b1, a2:b2] = b_Q[a0:b0, a1, a2:b2]
            b1 += 1
        if self.dtype == complex:
            if e2:
                b_Q[a0:b0, a1:b1, a2] *= 0.5
                b_Q[a0:b0, a1:b1, b2] = b_Q[a0:b0, a1:b1, a2]
        else:
            if e2:
                b_Q[a0:b0, a1:b1, b2 - 1] *= 0.5

        b_Q[:] = np.fft.ifftshift(b_Q, axes=axes)
        pd.ifftplan.execute()
        return pd.tmp_R * (1.0 / self.tmp_R.size), a_Q.ravel()[self.Q_G]

    def restrict(self, a_R, pd):
        a_Q = pd.tmp_Q
        b_Q = self.tmp_Q

        e0, e1, e2 = 1 - pd.gd.N_c % 2  # even or odd size
        a0, a1, a2 = self.gd.N_c // 2 - pd.gd.N_c // 2
        b0, b1, b2 = pd.gd.N_c // 2 + self.gd.N_c // 2 + 1

        if self.dtype == float:
            b2 = pd.gd.N_c[2] // 2 + 1
            a2 = 0
            axes = (0, 1)
        else:
            axes = (0, 1, 2)

        self.tmp_R[:] = a_R
        self.fftplan.execute()
        b_Q[:] = np.fft.fftshift(b_Q, axes=axes)

        if e0:
            b_Q[a0, a1:b1, a2:b2] += b_Q[b0 - 1, a1:b1, a2:b2]
            b_Q[a0, a1:b1, a2:b2] *= 0.5
            b0 -= 1
        if e1:
            b_Q[a0:b0, a1, a2:b2] += b_Q[a0:b0, b1 - 1, a2:b2]
            b_Q[a0:b0, a1, a2:b2] *= 0.5
            b1 -= 1
        if self.dtype == complex and e2:
            b_Q[a0:b0, a1:b1, a2] += b_Q[a0:b0, a1:b1, b2 - 1]
            b_Q[a0:b0, a1:b1, a2] *= 0.5
            b2 -= 1

        a_Q[:] = b_Q[a0:b0, a1:b1, a2:b2]
        a_Q[:] = np.fft.ifftshift(a_Q, axes=axes)
        a_G = a_Q.ravel()[pd.Q_G] / 8
        pd.ifftplan.execute()
        return pd.tmp_R * (1.0 / self.tmp_R.size), a_G

    def map(self, pd):
        N_c = np.array(self.tmp_Q.shape)
        N3_c = pd.tmp_Q.shape
        Q2_G = self.Q_G
        Q2_Gc = np.empty((len(Q2_G), 3), int)
        Q2_Gc[:, 0], r_G = divmod(Q2_G, N_c[1] * N_c[2])
        Q2_Gc.T[1:] = divmod(r_G, N_c[2])
        Q2_Gc[:, :2] += N_c[:2] // 2
        Q2_Gc[:, :2] %= N_c[:2]
        Q2_Gc[:, :2] -= N_c[:2] // 2
        Q2_Gc[:, :2] %= N3_c[:2]
        Q3_G = Q2_Gc[:, 2] + N3_c[2] * (Q2_Gc[:, 1] + N3_c[1] * Q2_Gc[:, 0])
        G3_Q = np.empty(N3_c, int).ravel()
        G3_Q[pd.Q_G] = np.arange(len(pd))
        return G3_Q[Q3_G]

    def gemm(self, alpha, psit_nG, C_mn, beta, newpsit_mG):
        """Helper function for MatrixOperator class."""
        if self.dtype == float:
            psit_nG = psit_nG.view(float)
            newpsit_mG = newpsit_mG.view(float)
        gemm(alpha, psit_nG, C_mn, beta, newpsit_mG)


class Preconditioner:
    """Preconditioner for KS equation.

    From:

      Teter, Payne and Allen, Phys. Rev. B 40, 12255 (1989)

    as modified by:

      Kresse and Furthmüller, Phys. Rev. B 54, 11169 (1996)
    """

    def __init__(self, G2_qG, pd):
        self.G2_qG = G2_qG
        self.pd = pd

    def calculate_kinetic_energy(self, psit_xG, kpt):
        G2_G = self.G2_qG[kpt.q]
        return [self.pd.integrate(0.5 * G2_G * psit_G, psit_G)
                for psit_G in psit_xG]

    def __call__(self, R_xG, kpt, ekin_x):
        G2_G = self.G2_qG[kpt.q]
        PR_xG = np.empty_like(R_xG)
        for PR_G, R_G, ekin in zip(PR_xG, R_xG, ekin_x):
            x_G = 1 / ekin / 3 * G2_G
            a_G = 27.0 + x_G * (18.0 + x_G * (12.0 + x_G * 8.0))
            PR_G[:] = 4.0 / 3 / ekin * R_G * a_G / (a_G + 16.0 * x_G**4)
        return PR_xG


class PWWaveFunctions(FDPWWaveFunctions):
    def __init__(self, ecut, fftwflags,
                 diagksl, orthoksl, initksl,
                 gd, nvalence, setups, bd, dtype,
                 world, kd, timer):
        self.ecut =  ecut
        self.fftwflags = fftwflags

        FDPWWaveFunctions.__init__(self, diagksl, orthoksl, initksl,
                                   gd, nvalence, setups, bd, dtype,
                                   world, kd, timer)
        
        self.orthoksl.gd = self.pd
        self.matrixoperator = MatrixOperator(self.orthoksl)

    def empty(self, n=(), dtype=float, global_array=False, realspace=False):
        if realspace:
            return self.gd.empty(n, dtype, global_array)
        else:
            return self.pd.empty(n, dtype)

    def integrate(self, a_xg, b_yg=None, global_integral=True):
        return self.pd.integrate(a_xg, b_yg, global_integral)

    def bytes_per_wave_function(self):
        return self.gd.bytecount(self.dtype)

    def set_setups(self, setups):
        self.timer.start('PWDescriptor')
        self.pd = PWDescriptor(self.ecut, self.gd, self.dtype, self.fftwflags)
        self.timer.stop('PWDescriptor')

        self.G2_qG = self.pd.g2(self.kd.i1bzk_qc)

        self.pt = PWLFC([setup.pt_j for setup in setups], self.pd, self.kd)

        FDPWWaveFunctions.set_setups(self, setups)

    def summary(self, fd):
        fd.write('Wave functions: Plane wave expansion\n')
        fd.write('      Cutoff energy: %.3f eV\n' %
                 (self.pd.ecut * units.Hartree))
        if self.dtype == float:
            fd.write('      Number of coefficients: %d (reduced to %d)\n' %
                     (len(self.pd) * 2 + 1, len(self.pd)))
        else:
            fd.write('      Number of coefficients: %d\n' % len(self.pd))
        if fftw.FFTPlan is fftw.FFTPlanB:
            fd.write("      Using Numpy's FFT\n")
        else:
            fd.write("      Using FFTW library\n")

    def make_preconditioner(self, block=1):
        return Preconditioner(self.G2_qG, self.pd)

    def apply_pseudo_hamiltonian(self, kpt, hamiltonian, psit_xG, Htpsit_xG):
        """Apply the non-pseudo Hamiltonian i.e. without PAW corrections."""
        Htpsit_xG[:] = 0.5 * self.G2_qG[kpt.q] * psit_xG
        for psit_G, Htpsit_G in zip(psit_xG, Htpsit_xG):
            psit_R = self.pd.ifft(psit_G)
            Htpsit_G += self.pd.fft(psit_R * hamiltonian.vt_sG[kpt.s])

    def add_to_density_from_k_point_with_occupation(self, nt_sR, kpt, f_n):
        nt_R = nt_sR[kpt.s]
        for f, psit_G in zip(f_n, kpt.psit_nG):
            nt_R += f * abs(self.pd.ifft(psit_G))**2

    def _get_wave_function_array(self, u, n, realspace=True):
        psit_G = FDPWWaveFunctions._get_wave_function_array(self, u, n,
                                                            realspace)
        if not realspace:
            return psit_G

        kpt = self.kpt_u[u]
        if self.kd.gamma:
            return self.pd.ifft(psit_G)
        else:
            N_c = self.gd.N_c
            k_c = self.kd.i1bzk_kc[kpt.k]
            eikr_R = np.exp(2j * pi * np.dot(np.indices(N_c).T, k_c / N_c).T)
            return self.pd.ifft(psit_G) * eikr_R

    def write(self, writer, write_wave_functions=False):
        writer['Mode'] = 'pw'
        writer['PlaneWaveCutoff'] = self.ecut

        if not write_wave_functions:
            return

        writer.dimension('nplanewaves', len(self.pd))
        writer.add('PseudoWaveFunctions',
                   ('nspins', 'nibzkpts', 'nbands', 'nplanewaves'),
                   dtype=complex)

        for s in range(self.nspins):
            for k in range(self.nibzkpts):
                for n in range(self.bd.nbands):
                    psit_G = self.get_wave_function_array(n, k, s,
                                                          realspace=False)
                    writer.fill(psit_G, s, k, n)

    def hs(self, ham, q=-1, s=0, md=None):
        assert self.dtype == complex

        n = len(self.pd)
        N = self.pd.tmp_R.size

        if md is None:
            H_GG = np.zeros((n, n), complex)
            S_GG = np.zeros((n, n), complex)
            G1 = 0
            G2 = n
        else:
            H_GG = md.zeros(dtype=complex)
            S_GG = md.zeros(dtype=complex)
            G1, G2 = md.my_blocks(S_GG).next()[:2]

        H_GG.ravel()[G1::n + 1] = (0.5 * self.pd.gd.dv / N *
                                   self.pd.G2_qG[q, G1:G2])

        for G in range(G1, G2):
            x_G = self.pd.zeros()
            x_G[G] = 1.0
            H_GG[G - G1] += (self.pd.gd.dv / N *
                             self.pd.fft(ham.vt_sG[s] * self.pd.ifft(x_G)))

        S_GG.ravel()[G1::n + 1] = self.pd.gd.dv / N

        f_IG = self.pt.expand(q)
        nI = len(f_IG)
        dH_II = np.zeros((nI, nI))
        dS_II = np.zeros((nI, nI))
        I1 = 0
        for a in self.pt.my_atom_indices:
            dH_ii = unpack(ham.dH_asp[a][s])
            dS_ii = self.setups[a].dO_ii
            I2 = I1 + len(dS_ii)
            dH_II[I1:I2, I1:I2] = dH_ii / N**2
            dS_II[I1:I2, I1:I2] = dS_ii / N**2
            I1 = I2

        H_GG += np.dot(f_IG.T[G1:G2].conj(), np.dot(dH_II, f_IG))
        S_GG += np.dot(f_IG.T[G1:G2].conj(), np.dot(dS_II, f_IG))
        
        return H_GG, S_GG
        
    def diagonalize_full_hamiltonian(self, ham, atoms, occupations, txt,
                                     nbands=None,
                                     scalapack=None):
        npw = len(self.pd)

        if nbands is None:
            nbands = npw

        self.bd = bd = BandDescriptor(nbands, self.bd.comm)

        if scalapack:
            nprow, npcol, b = scalapack
            bg = BlacsGrid(bd.comm, bd.comm.size, 1)
            mynpw = -(-npw // bd.comm.size)
            md = BlacsDescriptor(bg, npw, npw, mynpw, npw)

            bg2 = BlacsGrid(bd.comm, nprow, npcol)
            md2 = BlacsDescriptor(bg2, npw, npw, b, b)

            md3 = BlacsDescriptor(bg, npw, npw, bd.mynbands, npw)
        else:
            md = md2 = MatrixDescriptor(npw, npw)
            nprow = npcol = 1

        assert bd.comm.size == nprow * npcol

        self.pt.set_positions(atoms.get_scaled_positions())
        self.kpt_u[0].P_ani = None
        self.allocate_arrays_for_projections(self.pt.my_atom_indices)

        eps_n = np.empty(npw)
        myslice = bd.get_slice()

        for kpt in self.kpt_u:
            H_GG, S_GG = self.hs(ham, kpt.q, kpt.s, md)
            if scalapack:
                r = Redistributor(bd.comm, md, md2)
                H_GG = r.redistribute(H_GG)
                S_GG = r.redistribute(S_GG)

            psit_nG = md2.empty(dtype=complex)
            md2.general_diagonalize_dc(H_GG, S_GG, psit_nG, eps_n)
            del H_GG, S_GG

            kpt.eps_n = eps_n[myslice].copy()

            if scalapack:
                r = Redistributor(bd.comm, md2, md3)
                psit_nG = r.redistribute(psit_nG)

            kpt.psit_nG = psit_nG[:bd.mynbands].copy()
            del psit_nG

            self.pt.integrate(kpt.psit_nG, kpt.P_ani)

            f_n = np.zeros_like(kpt.eps_n)
            f_n[:len(kpt.f_n)] = kpt.f_n
            kpt.f_n = None

        occupations.calculate(self)
        
    def initialize_from_lcao_coefficients(self, basis_functions, mynbands):
        N_c = self.gd.N_c

        psit_nR = self.gd.empty(mynbands, self.dtype)

        for kpt in self.kpt_u:
            if self.kd.gamma:
                emikr_R = 1.0
            else:
                k_c = self.kd.i1bzk_kc[kpt.k]
                emikr_R = np.exp(-2j * pi *
                                  np.dot(np.indices(N_c).T, k_c / N_c).T)

            psit_nR[:] = 0.0
            basis_functions.lcao_to_grid(kpt.C_nM, psit_nR, kpt.q)
            kpt.C_nM = None

            kpt.psit_nG = self.pd.empty(self.bd.mynbands)
            for n in range(mynbands):
                kpt.psit_nG[n] = self.pd.fft(psit_nR[n] * emikr_R)

    def random_wave_functions(self, mynao):
        rs = np.random.RandomState(self.world.rank)
        weight_G = 1.0 / (1.0 + (self.pd.G_Gv**2).sum(1))
        for kpt in self.kpt_u:
            psit_nG = kpt.psit_nG[mynao:]
            psit_nG.real = rs.uniform(-1, 1, psit_nG.shape) * weight_G
            psit_nG.imag = rs.uniform(-1, 1, psit_nG.shape) * weight_G

    def estimate_memory(self, mem):
        FDPWWaveFunctions.estimate_memory(self, mem)
        self.pd.estimate_memory(mem.subnode('PW-descriptor'))
        mem.subnode('G2', self.G2_qG.nbytes)


def ft(spline):
    l = spline.get_angular_momentum_number()
    rc = 50.0
    N = 2**10
    assert spline.get_cutoff() <= rc

    dr = rc / N
    r_r = np.arange(N) * dr
    dk = pi / 2 / rc
    k_q = np.arange(2 * N) * dk
    f_r = spline.map(r_r) * (4 * pi)

    f_q = fbt(l, f_r, r_r, k_q)
    f_q[1:] /= k_q[1:]**(2 * l + 1)
    f_q[0] = (np.dot(f_r, r_r**(2 + 2 * l)) *
              dr * 2**l * fac[l] / fac[2 * l + 1])

    return Spline(l, k_q[-1], f_q)


class PWLFC(BaseLFC):
    def __init__(self, spline_aj, pd, kd=None):
        """Reciprocal-space plane-wave localized function collection."""

        self.pd = pd
        self.kd = kd

        if kd is None:
            k_qc = np.zeros((1, 3))
        else:
            k_qc = kd.i1bzk_qc

        self.G2_qG = pd.g2(k_qc)

        self.lf_aj = []
        cache = {}
        lmax = 0

        self.nbytes = 0

        # Fourier transform functions:
        for a, spline_j in enumerate(spline_aj):
            self.lf_aj.append([])
            for spline in spline_j:
                l = spline.get_angular_momentum_number()
                if spline not in cache:
                    f = ft(spline)
                    G_qG = self.G2_qG**0.5
                    f_qG = f.map(G_qG) * G_qG**l
                    cache[spline] = f_qG
                    self.nbytes += f_qG.size * 8
                else:
                    f_qG = cache[spline]
                self.lf_aj[a].append((l, f_qG))
                lmax = max(lmax, l)
            self.nbytes += len(pd) * 8  # self.emiGR_Ga
        
        self.dtype = pd.dtype

        B_cv = 2.0 * pi * self.pd.gd.icell_cv
        self.K_qv = np.dot(k_qc, B_cv)

        # Spherical harmonics:
        self.Y_qLG = np.empty((len(self.K_qv), (lmax + 1)**2, len(pd)))
        for q, K_v in enumerate(self.K_qv):
            G_Gv = pd.G_Gv + K_v
            G_Gv[1:] /= self.G2_qG[q, 1:, None]**0.5
            if self.G2_qG[q, 0] > 0:
                G_Gv[0] /= self.G2_qG[q, 0]**0.5
            for L in range((lmax + 1)**2):
                self.Y_qLG[q, L] = Y(L, *G_Gv.T)

        # These are set later in set_potitions():
        self.eikR_qa = None
        self.emiGR_Ga = None
        self.my_atom_indices = None
        self.indices = None

        self.nbytes += self.G2_qG.size * (lmax + 1)**2 * 8  # self.Y_qLG

    def estimate_memory(self, mem):
        mem.subnode('Arrays', self.nbytes)

    def get_function_count(self, a):
        return sum(2 * l + 1 for l, f_qG in self.lf_aj[a])

    def __iter__(self):
        I = 0
        for a in self.my_atom_indices:
            j = 0
            i1 = 0
            for l, f_qG in self.lf_aj[a]:
                i2 = i1 + 2 * l + 1
                yield a, j, i1, i2, I + i1, I + i2
                i1 = i2
                j += 1
            I += i2

    def set_positions(self, spos_ac):
        if self.kd is None or self.kd.gamma:
            self.eikR_qa = np.ones((1, len(spos_ac)))
        else:
            self.eikR_qa = np.exp(2j * pi * np.dot(self.kd.i1bzk_qc, spos_ac.T))

        pos_av = np.dot(spos_ac, self.pd.gd.cell_cv)
        self.emiGR_Ga = np.exp(-1j * np.dot(self.pd.G_Gv, pos_av.T))
        self.my_atom_indices = np.arange(len(spos_ac))
        self.indices = []
        I1 = 0
        for a in self.my_atom_indices:
            I2 = I1 + self.get_function_count(a)
            self.indices.append((a, I1, I2))
            I1 = I2
        self.nI = I2

    def expand(self, q=-1):
        f_IG = self.pd.empty(self.nI)
        for a, j, i1, i2, I1, I2 in self:
            l, f_qG = self.lf_aj[a][j]
            f_IG[I1:I2] = (self.emiGR_Ga[:, a] * f_qG[q] * (-1.0j)**l *
                           self.Y_qLG[q, l**2:(l + 1)**2])
        return f_IG

    def add(self, a_xG, c_axi=1.0, q=-1):
        if isinstance(c_axi, float):
            assert q == -1, a_xG.dims == 1
            a_xG += (c_axi / self.pd.gd.dv) * self.expand(-1).sum(0)
            return

        c_xI = np.empty(a_xG.shape[:-1] + (self.nI,), self.pd.dtype)
        f_IG = self.expand(q)
        for a, I1, I2 in self.indices:
            c_xI[..., I1:I2] = c_axi[a] * self.eikR_qa[q][a].conj()

        c_xI = c_xI.reshape((-1, self.nI))
        a_xG = a_xG.reshape((-1, len(self.pd)))

        if self.pd.dtype == float:
            f_IG = f_IG.view(float)
            a_xG = a_xG.view(float)

        gemm(1.0 / self.pd.gd.dv, f_IG, c_xI, 1.0, a_xG)

    def integrate(self, a_xG, c_axi, q=-1):
        c_xI = np.zeros(a_xG.shape[:-1] + (self.nI,), self.pd.dtype)
        f_IG = self.expand(q)

        b_xI = c_xI.reshape((-1, self.nI))
        a_xG = a_xG.reshape((-1, len(self.pd)))

        alpha = 1.0 / self.pd.gd.N_c.prod()
        if self.pd.dtype == float:
            alpha *= 2
            f_IG[:, 0] *= 0.5
            f_IG = f_IG.view(float)
            a_xG = a_xG.view(float)
            
        gemm(alpha, f_IG, a_xG, 0.0, b_xI, 'c')
        for a, I1, I2 in self.indices:
            c_axi[a][:] = self.eikR_qa[q][a] * c_xI[..., I1:I2]

    def derivative(self, a_xG, c_axiv, q=-1):
        c_xI = np.zeros(a_xG.shape[:-1] + (self.nI,), self.pd.dtype)
        f_IG = self.expand(q)

        K_v = self.K_qv[q]

        b_xI = c_xI.reshape((-1, self.nI))
        a_xG = a_xG.reshape((-1, len(self.pd)))

        alpha = 1.0 / self.pd.gd.N_c.prod()
        if self.pd.dtype == float:
            for v in range(3):
                gemm(2 * alpha,
                      (f_IG * 1.0j * self.pd.G_Gv[:, v]).view(float),
                      a_xG.view(float),
                      0.0, b_xI, 'c')
                for a, I1, I2 in self.indices:
                    c_axiv[a][..., v] = c_xI[..., I1:I2]
        else:
            for v in range(3):
                gemm(-alpha,
                      f_IG * (self.pd.G_Gv[:, v] + K_v[v]),
                      a_xG,
                      0.0, b_xI, 'c')
                for a, I1, I2 in self.indices:
                    c_axiv[a][..., v] = (1.0j * self.eikR_qa[q][a] *
                                         c_xI[..., I1:I2])


class PW:
    def __init__(self, ecut=340, fftwflags=fftw.FFTW_MEASURE):
        """Plane-wave basis mode.

        ecut: float
            Plane-wave cutoff in eV.
        fftwflags: int
            Flags for making FFTW plan (default is FFTW_MEASURE)."""

        self.ecut = ecut / units.Hartree
        self.fftwflags = fftwflags

    def __call__(self, diagksl, orthoksl, initksl, *args):
        wfs = PWWaveFunctions(self.ecut, self.fftwflags,
                              diagksl, orthoksl, initksl, *args)
        return wfs

    def __eq__(self, other):
        return (isinstance(other, PW) and self.ecut == other.ecut)

    def __ne__(self, other):
        return not self == other


class ReciprocalSpaceDensity(Density):
    def __init__(self, gd, finegd, nspins, charge, collinear=True):
        Density.__init__(self, gd, finegd, nspins, charge, collinear)

        self.ecut2 = 0.5 * pi**2 / (self.gd.h_cv**2).sum(1).max()
        self.pd2 = PWDescriptor(self.ecut2, self.gd)
        self.ecut3 = 0.5 * pi**2 / (self.finegd.h_cv**2).sum(1).max()
        self.pd3 = PWDescriptor(self.ecut3, self.finegd)

        self.G3_G = self.pd2.map(self.pd3)

    def initialize(self, setups, timer, magmom_av, hund):
        Density.initialize(self, setups, timer, magmom_av, hund)

        spline_aj = []
        for setup in setups:
            if setup.nct is None:
                spline_aj.append([])
            else:
                spline_aj.append([setup.nct])
        self.nct = PWLFC(spline_aj, self.pd2)

        self.ghat = PWLFC([setup.ghat_l for setup in setups], self.pd3)

    def set_positions(self, spos_ac, rank_a=None):
        Density.set_positions(self, spos_ac, rank_a)
        self.nct_q = self.pd2.zeros()
        self.nct.add(self.nct_q, 1.0 / self.nspins)
        self.nct_G = self.pd2.ifft(self.nct_q)

    def interpolate(self, comp_charge=None):
        """Interpolate pseudo density to fine grid."""
        if comp_charge is None:
            comp_charge = self.calculate_multipole_moments()

        if self.nt_sg is None:
            self.nt_sg = self.finegd.empty(self.nspins * self.ncomp**2)
            self.nt_sQ = self.pd2.empty(self.nspins * self.ncomp**2)

        for nt_G, nt_Q, nt_g in zip(self.nt_sG, self.nt_sQ, self.nt_sg):
            nt_g[:], nt_Q[:] = self.pd2.interpolate(nt_G, self.pd3)

    def calculate_pseudo_charge(self):
        self.nt_Q = self.nt_sQ[:self.nspins].sum(axis=0)
        self.rhot_q = self.pd3.zeros()
        self.rhot_q[self.G3_G] = self.nt_Q * 8
        self.ghat.add(self.rhot_q, self.Q_aL)
        self.rhot_q[0] = 0.0


class ReciprocalSpaceHamiltonian(Hamiltonian):
    def __init__(self, gd, finegd, pd2, pd3, nspins, setups, timer, xc,
                 vext=None, collinear=True):
        Hamiltonian.__init__(self, gd, finegd, nspins, setups, timer, xc,
                             vext, collinear)

        self.vbar = PWLFC([[setup.vbar] for setup in setups], pd2)
        self.pd2 = pd2
        self.pd3 = pd3

        class PS:
            def initialize(self):
                pass
            def get_stencil(self):
                return '????'
            def estimate_memory(self, mem):
                pass

        self.poisson = PS()
        self.npoisson = 0

    def summary(self, fd):
        Hamiltonian.summary(self, fd)
        fd.write('Interpolation: FFT\n')
        fd.write('Poisson solver: FFT\n')

    def set_positions(self, spos_ac, rank_a=None):
        Hamiltonian.set_positions(self, spos_ac, rank_a)
        self.vbar_Q = self.pd2.zeros()
        self.vbar.add(self.vbar_Q)

    def update_pseudo_potential(self, density):
        ebar = self.pd2.integrate(self.vbar_Q, density.nt_sQ.sum(0))

        self.timer.start('Poisson')
        # npoisson is the number of iterations:
        #self.npoisson = 0
        self.vHt_q = 4 * pi * density.rhot_q.copy()
        self.vHt_q[1:] /= self.pd3.G2_qG[0, 1:]
        epot = 0.5 * self.pd3.integrate(self.vHt_q, density.rhot_q)
        self.timer.stop('Poisson')

        # Calculate atomic hamiltonians:
        W_aL = {}
        for a in density.D_asp:
            W_aL[a] = np.empty((self.setups[a].lmax + 1)**2)
        density.ghat.integrate(self.vHt_q, W_aL)

        self.vt_Q = self.vbar_Q + self.vHt_q[density.G3_G] / 8
        self.vt_sG[:] = self.pd2.ifft(self.vt_Q)
        
        self.timer.start('XC 3D grid')
        vxct_sg = self.finegd.zeros(self.nspins)
        exc = self.xc.calculate(self.finegd, density.nt_sg, vxct_sg)

        for vt_G, vxct_g in zip(self.vt_sG, vxct_sg):
            vxc_G, vxc_Q = self.pd3.restrict(vxct_g, self.pd2)
            vt_G += vxc_G
            self.vt_Q += vxc_Q / self.nspins
        self.timer.stop('XC 3D grid')

        ekin = 0.0
        for vt_G, nt_G in zip(self.vt_sG, density.nt_sG):
            ekin -= self.gd.integrate(vt_G, nt_G)
        ekin += self.gd.integrate(self.vt_sG, density.nct_G).sum()

        eext = 0.0

        return ekin, epot, ebar, eext, exc, W_aL

    def calculate_forces2(self, dens, ghat_aLv, nct_av, vbar_av):
        dens.ghat.derivative(self.vHt_q, ghat_aLv)
        dens.nct.derivative(self.vt_Q, nct_av)
        self.vbar.derivative(dens.nt_sQ.sum(0), vbar_av)
