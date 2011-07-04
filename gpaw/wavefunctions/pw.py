import numpy as np
import ase.units as units

from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.wavefunctions.fdpw import FDPWWaveFunctions
from gpaw.hs_operators import MatrixOperator
import gpaw.fftw as fftw


class PWDescriptor:
    def __init__(self, ecut, gd, ibzk_qc=[(0, 0, 0)],
                 fftwflags=fftw.FFTW_MEASURE):
        assert gd.pbc_c.all() and gd.comm.size == 1

        self.ecut = ecut

        assert 0.5 * np.pi**2 / (gd.h_cv**2).sum(1).max() >= ecut
        
        # Calculate reciprocal lattice vectors:
        N_c = gd.N_c
        i_Qc = np.indices(N_c).transpose((1, 2, 3, 0))
        i_Qc += N_c // 2
        i_Qc %= N_c
        i_Qc -= N_c // 2
        B_cv = 2.0 * np.pi * gd.icell_cv
        G_Qv = np.dot(i_Qc, B_cv).reshape((-1, 3))
        G2_Q = (G_Qv**2).sum(axis=1)
        self.Q_G = np.arange(len(G2_Q))[G2_Q <= 2 * ecut]
        K_qv = np.dot(ibzk_qc, B_cv)
        G_Gv = G_Qv[self.Q_G]
        self.G2_qG = np.zeros((len(ibzk_qc), len(self.Q_G)))
        for q, K_v in enumerate(K_qv):
            self.G2_qG[q] = ((G_Gv + K_v)**2).sum(1)
        
        self.gd = gd
        self.dv = gd.dv / N_c.prod()
        self.comm = gd.comm

        self.n_c = self.Q_G  # used by hs_operators.py XXX

        self.tmp_R = fftw.empty(N_c, complex)
        self.fftplan = fftw.FFTPlan(self.tmp_R, -1, fftwflags)
        self.ifftplan = fftw.FFTPlan(self.tmp_R, 1, fftwflags)

    def bytecount(self, dtype=float):
        return len(self.Q_G) * np.array(1, dtype).itemsize
    
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
    def __init__(self, ecut, fftwflags,
                 diagksl, orthoksl, initksl,
                 gd, nvalence, setups, bd,
                 world, kd, timer):
        self.ecut =  ecut / units.Hartree
        self.fftwflags = fftwflags

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
        pt = LFC(self.gd, [setup.pt_j for setup in setups],
                 self.kpt_comm, dtype=self.dtype, forces=True)
        self.pt = PWLFC(pt, self.pd)
        FDPWWaveFunctions.set_setups(self, setups)

    def summary(self, fd):
        fd.write('Mode: Plane waves (%d, ecut=%.3f eV)\n' %
                 (len(self.pd.Q_G), self.pd.ecut * units.Hartree))
        
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


class PWLFC:
    def __init__(self, lfc, pd):
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
        self.expikr_qR = np.exp(2j * np.pi * np.dot(np.indices(N_c).T,
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
    def __init__(self, ecut=340, fftwflags=fftw.FFTW_MEASURE):
        self.ecut = ecut
        self.fftwflags = fftwflags

    def __call__(self, diagksl, orthoksl, initksl, *args):
        wfs = PWWaveFunctions(self.ecut, self.fftwflags,
                              diagksl, orthoksl, initksl, *args)
        return wfs
