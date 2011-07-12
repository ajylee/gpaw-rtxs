import sys
import numpy as np
from math import pi, sqrt
from time import time, ctime
from ase.units import Bohr, Hartree
from gpaw.utilities.blas import gemmdot, gemv, scal, axpy
from gpaw.response.base import BASECHI
from gpaw.fd_operators import Gradient
from datetime import timedelta

class SIGMA(BASECHI):

    def __init__(
                 self,
                 calc=None,
                 nbands=None,
                 bands=None,
                 kpoints=None,
                 w=None,
                 q=np.array([0.,0.,0.]),
                 eshift=None,
                 ecut=150.,
                 G_plus_q=False,
                 eta=0.1,
                 rpad=np.array([1,1,1]),
                 ftol=1e-5,
                 txt=None,
                 optical_limit=False
                ):

        BASECHI.__init__(self, calc, nbands, w, q, eshift, ecut,
                         G_plus_q, eta, rpad, ftol, txt, optical_limit)

        self.bands = bands
        self.kpoints = kpoints
        self.w = w


    def initialize(self):

        self.printtxt('')
        self.printtxt('-----------------------------------------------')
        self.printtxt('GW calculation started at:')
        self.printtxt(ctime())

        BASECHI.initialize(self)

        calc = self.calc
        self.kd = kd = calc.wfs.kd

        # frequency points init
        self.Nw = np.shape(self.w)[0] - 1
        self.dw = (self.w.max() - self.w.min()) / self.Nw
        self.w /= Hartree
        self.dw /= Hartree

        self.nkptout = np.shape(self.kpoints)[0]
        self.nbandsout = np.shape(self.bands)[0]


    def get_self_energy(self, W_wGG):

        self.initialize()

        Sigma_kn = np.zeros((self.nkptout, self.nbandsout), dtype=complex)
        Z_kn = np.zeros((self.nkptout, self.nbandsout), dtype=float)
        rho_G = np.zeros(self.npw, dtype=complex)

        i = 0
        for k in self.kpoints:

            for kq in range(np.shape(self.kd.bzk_kc)[0]):
                if ((np.abs(self.kd.bzk_kc[kq] - self.kd.bzk_kc[k] - self.q_c)) < 1e-5).all():
                    ibzkpt1 = self.kd.bz2ibz_k[k]
                    ibzkpt2 = self.kd.bz2ibz_k[kq]

                    j = 0
                    for n in self.bands:

                        for m in range(self.nbands):
                            check_focc = self.f_kn[ibzkpt2, m] > self.ftol
                            if check_focc:
                                occ = -1
                            else:
                                occ = 1

                            rho_G = self.density_matrix(n, m, k, kq)
                            rho_GG = np.outer(rho_G, rho_G.conj())

                            if m==n:
                                if (np.abs(self.q_c) < 1e-5).all():
                                    q_c = np.array([0.0001, 0., 0.])
                                    q_v = np.dot(q_c, self.bcell_cv)
                                    W_wGG[:,0,0] *= (q_v*q_v).sum()
                                    W_wGG[:,0,0] *= 2./pi*(6*pi**2/self.vol)**(1./3.)*self.vol

                            w0 = self.e_kn[ibzkpt2,m] - self.e_kn[ibzkpt1,n]
                            pm = occ*np.sign(self.e_kn[ibzkpt1,n] - self.e_kn[ibzkpt2,m])
                            w0_id = np.abs(int(w0 / self.dw))
                            w1 = w0_id * self.dw
                            w2 = (w0_id + 1) * self.dw

                            w1_w = np.zeros(self.Nw, dtype=complex)
                            w2_w = np.zeros(self.Nw, dtype=complex)
                            if pm == 1:
                                for iw in range(self.Nw):
                                    w = iw * self.dw
                                    w1_w[iw] = 1. / (w1 + w + 1j*self.eta) + 1. / (w1 - w + 1j*self.eta)
                                    w2_w[iw] = 1. / (w2 + w + 1j*self.eta) + 1. / (w2 - w + 1j*self.eta)
                            if pm == -1:
                                for iw in range(self.Nw):
                                    w = iw * self.dw
                                    w1_w[iw] = 1. / (w1 + w - 1j*self.eta) + 1. / (w1 - w - 1j*self.eta)
                                    w2_w[iw] = 1. / (w2 + w - 1j*self.eta) + 1. / (w2 - w - 1j*self.eta)

                            Cw1_GG = 1j/(2*pi) * gemmdot(w1_w, W_wGG, beta = 0.) * self.dw
                            Cw2_GG = 1j/(2*pi) * gemmdot(w2_w, W_wGG, beta = 0.) * self.dw

                            Sw1 = 1. / self.vol * np.sum(Cw1_GG * rho_GG)
                            Sw2 = 1. / self.vol * np.sum(Cw2_GG * rho_GG)

                            Sw0 = (w2-np.abs(w0))/self.dw * Sw1 + (np.abs(w0)-w1)/self.dw * Sw2

                            Sigma_kn[i][j] = Sigma_kn[i][j] + np.sign(self.e_kn[ibzkpt1,n] - self.e_kn[ibzkpt2,m])*Sw0
                            Z_kn[i][j] = Z_kn[i][j] + 1./(1 - np.real((Sw2 - Sw1)/(w2 - w1)))
                        j+=1
            i+=1
        return np.real(Sigma_kn), Z_kn/self.nbands


    def density_matrix(self,n,m,k,kq):

        ibzk_kc = self.ibzk_kc
        bzk_kc = self.bzk_kc
        gd = self.gd
        kd = self.kd

        phi_aGp = self.get_phi_aGp()

        expqr_g = self.expqr_g
        q_v = self.qq_v
        optical_limit = self.optical_limit

        ibzkpt1 = kd.bz2ibz_k[k]
        ibzkpt2 = kd.bz2ibz_k[kq]
        
        psitold_g = self.get_wavefunction(ibzkpt1, n, True)
        psit1_g = kd.transform_wave_function(psitold_g, k)
        
        psitold_g = self.get_wavefunction(ibzkpt2, m, True)
        psit2_g = kd.transform_wave_function(psitold_g, kq)

        # FFT
        tmp_g = psit1_g.conj()* psit2_g * expqr_g
        rho_g = np.fft.fftn(tmp_g) * self.vol / self.nG0

        # Here, planewave cutoff is applied
        rho_G = np.zeros(self.npw, dtype=complex)
        for iG in range(self.npw):
            index = self.Gindex_G[iG]
            rho_G[iG] = rho_g[index[0], index[1], index[2]]

        if optical_limit:
            d_c = [Gradient(gd, i, n=4, dtype=complex).apply for i in range(3)]
            dpsit_g = gd.empty(dtype=complex)
            tmp = np.zeros((3), dtype=complex)
            phase_cd = np.exp(2j * pi * gd.sdisp_cd * bzk_kc[kq, :, np.newaxis])
            for ix in range(3):
                d_c[ix](psit2_g, dpsit_g, phase_cd)
                tmp[ix] = gd.integrate(psit1_g.conj() * dpsit_g)
            rho_G[0] = -1j * np.dot(q_v, tmp)

        # PAW correction
        pt = self.pt
        P1_ai = pt.dict()
        pt.integrate(psit1_g, P1_ai, k)
        P2_ai = pt.dict()
        pt.integrate(psit2_g, P2_ai, kq)
                            
        for a, id in enumerate(self.calc.wfs.setups.id_a):
            P_p = np.outer(P1_ai[a].conj(), P2_ai[a]).ravel()
            gemv(1.0, phi_aGp[a], P_p, 1.0, rho_G)
    
        if optical_limit:
            if n==m:
                rho_G[0] = 1.
            elif np.abs(self.e_kn[ibzkpt2, m] - self.e_kn[ibzkpt1, n]) < 1e-5:
                rho_G[0] = 0.
            else:
                rho_G[0] /= (self.e_kn[ibzkpt2, m] - self.e_kn[ibzkpt1, n])

        return rho_G
