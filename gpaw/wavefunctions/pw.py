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
from gpaw.utilities.blas import rk, r2k, gemm


class PWDescriptor:
    def __init__(self, ecut, gd, dtype=float, fftwflags=fftw.FFTW_MEASURE):

        assert gd.pbc_c.all() and gd.comm.size == 1

        self.ecut = ecut
        self.gd = gd

        N_c = gd.N_c
        self.comm = gd.comm

        assert 0.5 * pi**2 / (gd.h_cv**2).sum(1).max() > ecut

        self.dtype = dtype

        if dtype == float:
            Nr_c = N_c.copy()
            Nr_c[2] = N_c[2] // 2 + 1
            i_Qc = np.indices(Nr_c).transpose((1, 2, 3, 0))
            i_Qc[..., :2] += N_c[:2] // 2
            i_Qc[..., :2] %= N_c[:2]
            i_Qc[..., :2] -= N_c[:2] // 2
            self.tmp_Q = fftw.empty(Nr_c, complex)
            self.tmp_R = self.tmp_Q.view(float)[:, :, :-2]
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
        mask_Q = G2_Q <= 2 * ecut
        if self.dtype == float:
            mask_Q &= ((i_Qc[:, 2] > 0) |
                       (i_Qc[:, 1] > 0) |
                       ((i_Qc[:, 0] >= 0) & (i_Qc[:, 1] == 0)))
        self.Q_G = np.arange(len(G2_Q))[mask_Q]
        self.G_Gv = G_Qv[self.Q_G]

        self.n_c = self.Q_G #??????? # used by hs_operators.py XXX
        self.ibzk_qc = []

    def g2(self, ibzk_qc):
        # Did we already do this one?
        if (len(self.ibzk_qc) == len(ibzk_qc) and
            (self.ibzk_qc == ibzk_qc).all()):
            return self.G2_qG

        # No.
        self.ibzk_qc = ibzk_qc
        B_cv = 2.0 * pi * self.gd.icell_cv
        K_qv = np.dot(ibzk_qc, B_cv)
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
    
    def zeros(self, x=(), dtype=float):
        a_xG = self.empty(x, dtype)
        a_xG.fill(0.0)
        return a_xG
    
    def empty(self, x=(), dtype=float):
        assert dtype == self.dtype
        if isinstance(x, int):
            x = (x,)
        shape = x + self.Q_G.shape
        return np.empty(shape, complex)
    
    def fft(self, a_R):
        self.tmp_R[:] = a_R
        self.fftplan.execute()
        return self.tmp_Q.ravel()[self.Q_G]

    def ifft(self, a_G):
        self.tmp_Q[:] = 0.0
        self.tmp_Q.ravel()[self.Q_G] = a_G
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
        
        xshape = a_xg.shape[:-1]
        
        alpha = self.gd.dv / self.gd.N_c.prod()

        if b_yg is None:
            # Only one array:
            assert self.dtype == float
            return a_xg[..., 0].real * alpha

        A_xg = a_xg.reshape((-1, len(self)))
        B_yg = b_yg.reshape((-1, len(self)))

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
            result_yx -= 0.5 * alpha * np.outer(B_yg[:, 0], A_xg[:, 0])

        yshape = b_yg.shape[:-1]
        result = result_yx.T.reshape(xshape + yshape)
        
        if result.ndim == 0:
            return result.item()
        else:
            return result

    def gemm(self, alpha, psit_nG, C_mn, beta, newpsit_mG):
        """Helper function for MatrixOperator class."""
        if self.dtype == float:
            psit_nG = psit_nG.view(float)
            newpsit_mG = newpsit_mG.view(float)
        gemm(alpha, psit_nG, C_mn, beta, newpsit_mG)


class Preconditioner:
    def __init__(self, G2_qG):
        self.G2_qG = G2_qG
        self.allocated = True

    def __call__(self, R_G, kpt):
        return R_G / (1.0 + self.G2_qG[kpt.q])


class PWWaveFunctions(FDPWWaveFunctions):
    def __init__(self, ecut, fftwflags,
                 diagksl, orthoksl, initksl,
                 gd, nvalence, setups, bd, dtype,
                 world, kd, timer):
        self.ecut =  ecut / units.Hartree
        self.fftwflags = fftwflags

        #kd.gamma = False
        FDPWWaveFunctions.__init__(self, diagksl, orthoksl, initksl,
                                   gd, nvalence, setups, bd, dtype,
                                   world, kd, timer)
        
        self.orthoksl.gd = self.pd
        self.matrixoperator = MatrixOperator(self.orthoksl)
        self.wd = self.pd

    def set_setups(self, setups):
        self.timer.start('PWDescriptor')
        self.pd = PWDescriptor(self.ecut, self.gd, self.dtype, self.fftwflags)
        self.timer.stop('PWDescriptor')

        self.G2_qG = self.pd.g2(self.kd.ibzk_qc)

        self.pt = PWLFC([setup.pt_j for setup in setups], self.pd, self.kd)

        FDPWWaveFunctions.set_setups(self, setups)

    def summary(self, fd):
        fd.write('Mode: Plane waves (%d, ecut=%.3f eV)\n' %
                 (len(self.pd), self.pd.ecut * units.Hartree))
        
    def make_preconditioner(self, block=1):
        return Preconditioner(self.G2_qG)

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

    def initialize_wave_functions_from_basis_functions(self, basis_functions,
                                                       density, hamiltonian,
                                                       spos_ac):
        FDPWWaveFunctions.initialize_wave_functions_from_basis_functions(
            self, basis_functions, density, hamiltonian, spos_ac)

        for kpt in self.kpt_u:
            psit_nG = self.pd.empty(self.bd.mynbands, self.dtype)
            for n, psit_R in enumerate(kpt.psit_nG):
                psit_nG[n] = self.pd.fft(psit_R)
            kpt.psit_nG = psit_nG

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
            k_qc = kd.ibzk_qc

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
            self.eikR_qa = np.exp(2j * pi * np.dot(self.kd.ibzk_qc, spos_ac.T))

        pos_av = np.dot(spos_ac, self.pd.gd.cell_cv)
        self.emiGR_Ga = np.exp(-1j * np.dot(self.pd.G_Gv, pos_av.T))
        self.my_atom_indices = np.arange(len(spos_ac))

    def expand(self, q):
        nI = sum(self.get_function_count(a) for a in self.my_atom_indices)
        f_IG = self.pd.empty(nI, self.pd.dtype)
        for a, j, i1, i2, I1, I2 in self:
            l, f_qG = self.lf_aj[a][j]
            f_IG[I1:I2] = (self.emiGR_Ga[:, a] * f_qG[q] * (-1.0j)**l *
                           self.Y_qLG[q, l**2:(l + 1)**2])
        return f_IG

    def add(self, a_xG, c_axi, q=-1):
        nI = sum(self.get_function_count(a) for a in self.my_atom_indices)
        c_xI = np.empty(a_xG.shape[:-1] + (nI,), self.pd.dtype)
        f_IG = self.expand(q)
        for a, j, i1, i2, I1, I2 in self:
            l = self.lf_aj[a][j][0]
            c_xI[..., I1:I2] = c_axi[a][..., i1:i2] * self.eikR_qa[q][a].conj()

        c_xI = c_xI.reshape((-1, nI))
        a_xG = a_xG.reshape((-1, len(self.pd)))

        if self.pd.dtype == float:
            f_IG = f_IG.view(float)
            a_xG = a_xG.view(float)

        gemm(1.0 / self.pd.gd.dv, f_IG, c_xI, 1.0, a_xG)

    def integrate(self, a_xG, c_axi, q=-1):
        nI = sum(self.get_function_count(a) for a in self.my_atom_indices)
        c_xI = np.zeros(a_xG.shape[:-1] + (nI,), self.pd.dtype)
        f_IG = self.expand(q)

        b_xI = c_xI.reshape((-1, nI))
        a_xG = a_xG.reshape((-1, len(self.pd)))

        alpha = 1.0 / self.pd.gd.N_c.prod()
        if self.pd.dtype == float:
            alpha *= 2
            f_IG[:, 0] *= 0.5
            f_IG = f_IG.view(float)
            a_xG = a_xG.view(float)
            
        gemm(alpha, f_IG, a_xG, 0.0, b_xI, 'c')
        for a, j, i1, i2, I1, I2 in self:
            l = self.lf_aj[a][j][0]
            c_axi[a][..., i1:i2] = self.eikR_qa[q][a] * c_xI[..., I1:I2]

    def derivative(self, a_xG, c_axiv, q=-1):
        nI = sum(self.get_function_count(a) for a in self.my_atom_indices)
        c_xI = np.zeros(a_xG.shape[:-1] + (nI,), self.pd.dtype)
        f_IG = self.expand(q)

        K_v = self.K_qv[q]

        b_xI = c_xI.reshape((-1, nI))
        a_xG = a_xG.reshape((-1, len(self.pd)))

        alpha = 1.0 / self.pd.gd.N_c.prod()
        if self.pd.dtype == float:
            for v in range(3):
                gemm(2 * alpha,
                      (f_IG * 1.0j * self.pd.G_Gv[:, v]).view(float),
                      a_xG.view(float),
                      0.0, b_xI, 'c')
                for a, j, i1, i2, I1, I2 in self:
                    l = self.lf_aj[a][j][0]
                    c_axiv[a][..., i1:i2, v] = c_xI[..., I1:I2]
        else:
            for v in range(3):
                gemm(-alpha,
                      f_IG * (self.pd.G_Gv[:, v] + K_v[v]),
                      a_xG,
                      0.0, b_xI, 'c')
                for a, j, i1, i2, I1, I2 in self:
                    l = self.lf_aj[a][j][0]
                    c_axiv[a][..., i1:i2, v] = (1.0j * self.eikR_qa[q][a] *
                                                c_xI[..., I1:I2])


class PW:
    def __init__(self, ecut=340, fftwflags=fftw.FFTW_MEASURE):
        """Plane-wave basis mode.

        ecut: float
            Plane-wave cutoff in eV.
        fftwflags: int
            Flags for making FFTW plan (default is FFTW_MEASURE)."""

        self.ecut = ecut
        self.fftwflags = fftwflags

    def __call__(self, diagksl, orthoksl, initksl, *args):
        wfs = PWWaveFunctions(self.ecut, self.fftwflags,
                              diagksl, orthoksl, initksl, *args)
        return wfs


class ReciprocalSpaceDensity(Density):
    def initialize(self, setups, timer, magmom_av, hund):
        Density.initialize(self, setups, timer, magmom_av, hund)

        spline_aj = []
        for setup in setups:
            if setup.nct is None:
                spline_aj.append([])
            else:
                spline_aj.append([setup.nct])
        self.nct = PWLFC(self.gd, spline_aj,
                         integral=[setup.Nct for setup in setups],
                         forces=True, cut=True)
        self.ghat = PWLFC(self.finegd, [setup.ghat_l for setup in setups],
                          integral=sqrt(4 * pi), forces=True)


class ReciprocalSpaceHamiltonian(Hamiltonian):
    def __init__(self, gd, finegd, nspins, setups, timer, xc,
                 vext=None, collinear=True):
        Hamiltonian.__init__(self, gd, finegd, nspins, setups, timer, xc,
                             vext, collinear)

        self.vbar = PWLFC(self.finegd, [[setup.vbar] for setup in setups])

    def set_positions(self, spos_ac, rank_a=None):
        Hamiltonian.set_positions(self, spos_ac, rank_a)
        if self.vbar_g is None:
            self.vbar_g = self.finegd.empty()
        self.vbar_g[:] = 0.0
        self.vbar.add(self.vbar_g)

    def update_pseudo_potential(self, density):
        self.timer.start('vbar')
        Ebar = self.finegd.integrate(self.vbar_g, density.nt_g,
                                     global_integral=False)

        vt_g = self.vt_sg[0]
        vt_g[:] = self.vbar_g
        self.timer.stop('vbar')

        Eext = 0.0
        if self.vext is not None:
            assert self.collinear
            vt_g += self.vext.get_potential(self.finegd)
            Eext = self.finegd.integrate(vt_g, density.nt_g,
                                         global_integral=False) - Ebar

        self.vt_sg[1:self.nspins] = vt_g

        self.vt_sg[self.nspins:] = 0.0
            
        self.timer.start('XC 3D grid')
        Exc = self.xc.calculate(self.finegd, density.nt_sg, self.vt_sg)
        Exc /= self.gd.comm.size
        self.timer.stop('XC 3D grid')

        self.timer.start('Poisson')
        # npoisson is the number of iterations:
        self.npoisson = self.poisson.solve(self.vHt_g, density.rhot_g,
                                           charge=-density.charge)
        self.timer.stop('Poisson')

        self.timer.start('Hartree integrate/restrict')
        Epot = 0.5 * self.finegd.integrate(self.vHt_g, density.rhot_g,
                                           global_integral=False)
        Ekin = 0.0
        s = 0
        for vt_g, vt_G, nt_G in zip(self.vt_sg, self.vt_sG, density.nt_sG):
            if s < self.nspins:
                vt_g += self.vHt_g
            self.restrict(vt_g, vt_G)
            if s < self.nspins:
                Ekin -= self.gd.integrate(vt_G, nt_G - density.nct_G,
                                          global_integral=False)
            else:
                Ekin -= self.gd.integrate(vt_G, nt_G, global_integral=False)
            s += 1
                
        self.timer.stop('Hartree integrate/restrict')
            
        # Calculate atomic hamiltonians:
        self.timer.start('Atomic')
        W_aL = {}
        for a in density.D_asp:
            W_aL[a] = np.empty((self.setups[a].lmax + 1)**2)
        density.ghat.integrate(self.vHt_g, W_aL)

        return Ekin, Epot, Ebar, Eext, Exc, W_aL
