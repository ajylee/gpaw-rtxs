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
from gpaw.utilities import _fact as fac
from gpaw.utilities.blas import gemm


class PWDescriptor:
    def __init__(self, ecut, gd, ibzk_qc=[(0, 0, 0)],
                 fftwflags=fftw.FFTW_MEASURE):
        assert gd.pbc_c.all() and gd.comm.size == 1

        self.ecut = ecut

        assert 0.5 * pi**2 / (gd.h_cv**2).sum(1).max() >= ecut
        
        # Calculate reciprocal lattice vectors:
        N_c = gd.N_c
        i_Qc = np.indices(N_c).transpose((1, 2, 3, 0))
        i_Qc += N_c // 2
        i_Qc %= N_c
        i_Qc -= N_c // 2
        B_cv = 2.0 * pi * gd.icell_cv
        G_Qv = np.dot(i_Qc, B_cv).reshape((-1, 3))
        G2_Q = (G_Qv**2).sum(axis=1)
        self.Q_G = np.arange(len(G2_Q))[G2_Q <= 2 * ecut]
        K_qv = np.dot(ibzk_qc, B_cv)
        self.G_Gv = G_Qv[self.Q_G]
        self.G2_qG = np.zeros((len(ibzk_qc), len(self.Q_G)))
        for q, K_v in enumerate(K_qv):
            self.G2_qG[q] = ((self.G_Gv + K_v)**2).sum(1)
        
        self.gd = gd
        self.dv = gd.dv / N_c.prod()
        self.comm = gd.comm

        self.n_c = self.Q_G  # used by hs_operators.py XXX

        self.tmp_R = fftw.empty(N_c, complex)
        self.fftplan = fftw.FFTPlan(self.tmp_R, -1, fftwflags)
        self.ifftplan = fftw.FFTPlan(self.tmp_R, 1, fftwflags)

    def __len__(self):
        return len(self.Q_G)

    def bytecount(self, dtype=float):
        return len(self) * np.array(1, dtype).itemsize
    
    def zeros(self, n=(), dtype=float):
        assert dtype == complex
        if isinstance(n, int):
            n = (n,)
        shape = n + self.Q_G.shape
        return np.zeros(shape, complex)
    
    def empty(self, n=(), dtype=float):
        assert dtype == complex
        if isinstance(n, int):
            n = (n,)
        shape = n + self.Q_G.shape
        return np.empty(shape, complex)
    
    def fft(self, a_R):
        self.tmp_R[:] = a_R
        self.fftplan.execute()
        return self.tmp_R.ravel()[self.Q_G]

    def ifft(self, a_G):
        self.tmp_R[:] = 0.0
        self.tmp_R.ravel()[self.Q_G] = a_G
        self.ifftplan.execute()
        return self.tmp_R * (1.0 / self.tmp_R.size)


class Preconditioner:
    def __init__(self, pd):
        self.pd = pd
        self.allocated = True

    def __call__(self, R_G, kpt):
        return R_G / (1.0 + self.pd.G2_qG[kpt.q])


class PWWaveFunctions(FDPWWaveFunctions):
    def __init__(self, ecut, fftwflags, real_space_projections,
                 diagksl, orthoksl, initksl,
                 gd, nvalence, setups, bd,
                 world, kd, timer):
        self.ecut =  ecut / units.Hartree
        self.fftwflags = fftwflags
        self.real_space_projections = real_space_projections

        # Set dtype=complex and gamma=False:
        kd.gamma = False
        FDPWWaveFunctions.__init__(self, diagksl, orthoksl, initksl,
                                   gd, nvalence, setups, bd, complex,
                                   world, kd, timer)
        
        orthoksl.gd = self.pd
        self.matrixoperator = MatrixOperator(orthoksl)
        self.wd = self.pd        

    def set_setups(self, setups):
        self.timer.start('PWDescriptor')
        self.pd = PWDescriptor(self.ecut, self.gd, self.kd.ibzk_qc,
                               self.fftwflags)
        self.timer.stop('PWDescriptor')
        if self.real_space_projections:
            pt = LFC(self.gd, [setup.pt_j for setup in setups],
                     self.kpt_comm, dtype=self.dtype, forces=True)
            self.pt = RealSpacePWLFC(pt, self.pd)
        else:
            self.pt = PWLFC([setup.pt_j for setup in setups], self.pd)
        FDPWWaveFunctions.set_setups(self, setups)

    def summary(self, fd):
        fd.write('Mode: Plane waves (%d, ecut=%.3f eV)\n' %
                 (len(self.pd), self.pd.ecut * units.Hartree))
        fd.write('      Doing projections in %s space.\n' %
                 ['reciprocal', 'real'][self.real_space_projections])
        
    def make_preconditioner(self, block=1):
        return Preconditioner(self.pd)

    def apply_pseudo_hamiltonian(self, kpt, hamiltonian, psit_xG, Htpsit_xG):
        """Apply the non-pseudo Hamiltonian i.e. without PAW corrections."""
        Htpsit_xG[:] = 0.5 * self.pd.G2_qG[kpt.q] * psit_xG
        for psit_G, Htpsit_G in zip(psit_xG, Htpsit_xG):
            psit_R = self.pd.ifft(psit_G)
            Htpsit_G += self.pd.fft(psit_R * hamiltonian.vt_sG[kpt.s])

    def add_to_density_from_k_point_with_occupation(self, nt_sR, kpt, f_n):
        nt_R = nt_sR[kpt.s]
        for f, psit_G in zip(f_n, kpt.psit_nG):
            nt_R += f * abs(self.pd.ifft(psit_G))**2

    def initialize_wave_functions_from_basis_functions(self, basis_functions,
                                                       density, hamiltonian,
                                                       spos_ac):
        FDPWWaveFunctions.initialize_wave_functions_from_basis_functions(
            self, basis_functions, density, hamiltonian, spos_ac)

        for kpt in self.kpt_u:
            psit_nG = self.pd.empty(self.bd.mynbands, complex)
            for n, psit_R in enumerate(kpt.psit_nG):
                psit_nG[n] = self.pd.fft(psit_R)
            kpt.psit_nG = psit_nG


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
    def __init__(self, spline_aj, pd):
        """Reciprocal-space plane-wave localized function collection."""

        self.pd = pd

        self.lf_aj = []
        cache = {}
        self.lmax = 0

        # Fourier transform functions:
        for a, spline_j in enumerate(spline_aj):
            self.lf_aj.append([])
            for spline in spline_j:
                l = spline.get_angular_momentum_number()
                if spline not in cache:
                    f = ft(spline)
                    G_qG = pd.G2_qG**0.5
                    f_qG = f.map(G_qG) * G_qG**l
                    cache[spline] = f_qG
                else:
                    f_qG = cache[spline]
                self.lf_aj[a].append((l, f_qG))
                self.lmax = max(self.lmax, l)
        
        self.dtype = complex

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

    def set_k_points(self, k_qc):
        self.k_qc = k_qc
        B_cv = 2.0 * pi * self.pd.gd.icell_cv
        K_qv = np.dot(k_qc, B_cv)

        self.Y_qLG = np.empty((len(K_qv), (self.lmax + 1)**2, len(self.pd)))
        for q, K_v in enumerate(K_qv):
            G_Gv = self.pd.G_Gv + K_v
            G_Gv[1:] /= self.pd.G2_qG[q, 1:, None]**0.5
            if self.pd.G2_qG[q, 0] > 0:
                G_Gv[0] /= self.pd.G2_qG[q, 0]**0.5
            for L in range((self.lmax + 1)**2):
                self.Y_qLG[q, L] = Y(L, *G_Gv.T)

    def set_positions(self, spos_ac):
        self.eikR_qa = np.exp(2j * pi * np.dot(self.k_qc, spos_ac.T))
        pos_av = np.dot(spos_ac, self.pd.gd.cell_cv)
        self.emiGR_Ga = np.exp(-1j * np.dot(self.pd.G_Gv, pos_av.T))
        self.my_atom_indices = np.arange(len(spos_ac))

    def expand(self, q):
        nI = sum(self.get_function_count(a) for a in self.my_atom_indices)
        f_IG = self.pd.empty(nI, complex)
        for a, j, i1, i2, I1, I2 in self:
            l, f_qG = self.lf_aj[a][j]
            f_IG[I1:I2] = (self.emiGR_Ga[:, a] * f_qG[q] *
                           self.Y_qLG[q, l**2:(l + 1)**2])
        return f_IG

    def add(self, a_xG, c_axi, q):
        assert a_xG.ndim == 2
        nI = sum(self.get_function_count(a) for a in self.my_atom_indices)
        c_xI = np.empty((len(a_xG), nI), complex)
        f_IG = self.expand(q)
        for a, j, i1, i2, I1, I2 in self:
            l = self.lf_aj[a][j][0]
            c_xI[:, I1:I2] = (c_axi[a][:, i1:i2] * (-1.0j)**l *
                              self.eikR_qa[q][a].conj())

        gemm(1.0 / self.pd.gd.dv, f_IG, c_xI, 1.0, a_xG)

    def integrate(self, a_xG, c_axi, q):
        assert a_xG.ndim == 2
        nI = sum(self.get_function_count(a) for a in self.my_atom_indices)
        c_xI = np.zeros((len(a_xG), nI), complex)
        f_IG = self.expand(q)
        gemm(1.0 / self.pd.gd.N_c.prod(), f_IG, a_xG, 0.0, c_xI, 'c')
        for a, j, i1, i2, I1, I2 in self:
            l = self.lf_aj[a][j][0]
            c_axi[a][:, i1:i2] = (1.0j**l * self.eikR_qa[q][a] * c_xI[:, I1:I2])


class RealSpacePWLFC:
    def __init__(self, lfc, pd):
        """Real-space plane-wave localized function collection."""
        self.lfc = lfc
        self.pd = pd

    def dict(self, shape=(), derivative=False, zero=False):
        return self.lfc.dict(shape, derivative, zero)

    def set_positions(self, spos_ac):
        self.lfc.set_positions(spos_ac)
        self.my_atom_indices = self.lfc.my_atom_indices
        
    def set_k_points(self, ibzk_qc):
        self.lfc.set_k_points(ibzk_qc)
        N_c = self.pd.gd.N_c
        self.expikr_qR = np.exp(2j * pi * np.dot(np.indices(N_c).T,
                                                    (ibzk_qc / N_c).T).T)

    def add(self, a_xG, c_axi, q):
        a_R = self.pd.tmp_R
        xshape = a_xG.shape[:-1]
        for x in np.indices(xshape).reshape((len(xshape), -1)).T:
            c_ai = {}
            for a, c_xi in c_axi.items():
                c_ai[a] = c_xi[x]
            a_R[:] = 0.0
            self.lfc.add(a_R, c_ai, q)
            a_xG[x] += self.pd.fft(a_R / self.expikr_qR[q])

    def integrate(self, a_xG, c_axi, q):
        c_ai = self.dict()
        xshape = a_xG.shape[:-1]
        for x in np.indices(xshape).reshape((len(xshape), -1)).T:
            a_R = self.pd.ifft(a_xG[x]) * self.expikr_qR[q]
            self.lfc.integrate(a_R, c_ai, q)
            for a, c_i in c_ai.items():
                c_axi[a][x] = c_i

    def derivative(self, a_xG, c_axiv, q):
        c_aiv = self.dict(derivative=True)
        xshape = a_xG.shape[:-1]
        for x in np.indices(xshape).reshape((len(xshape), -1)).T:
            a_R = self.pd.ifft(a_xG[x]) * self.expikr_qR[q]
            self.lfc.derivative(a_R, c_aiv, q)
            for a, c_iv in c_aiv.items():
                c_axiv[a][x] = c_iv


class PW:
    def __init__(self, ecut=340, fftwflags=fftw.FFTW_MEASURE,
                 real_space_projections=False):
        self.ecut = ecut
        self.fftwflags = fftwflags
        self.real_space_projections = real_space_projections

    def __call__(self, diagksl, orthoksl, initksl, *args):
        wfs = PWWaveFunctions(self.ecut, self.fftwflags,
                              self.real_space_projections,
                              diagksl, orthoksl, initksl, *args)
        return wfs
