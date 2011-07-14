import numpy as np
import copy
import sys
from math import pi, sqrt
from time import time, ctime
from datetime import timedelta
from ase.parallel import paropen
from ase.units import Hartree, Bohr
from gpaw import GPAW
from gpaw.response.parallel import parallel_partition
from gpaw.mpi import world, rank, size, serial_comm
from gpaw.response.df import DF
from gpaw.response.cell import get_primitive_cell, set_Gvectors
from gpaw.utilities import devnull
from gpaw.xc.hybridk import HybridXC
from gpaw.xc.tools import vxc
from gpaw.response.base import BASECHI

class GW(BASECHI):

    def __init__(
                 self,
                 file=None,
                 bands=None,
                 kpoints=None,
                 w=None,
                 ecut=150.,
                 eta=0.1,
                 txt=None,
                ):

        BASECHI.__init__(self, calc=file, w=w, ecut=ecut, eta=eta, txt=txt)

        self.bands = bands
        self.kpoints = kpoints

    def initialize(self):

        self.printtxt('')
        self.printtxt('-----------------------------------------------')
        self.printtxt('GW calculation started at:')
        self.printtxt(ctime())

        BASECHI.initialize(self)
        self.kd = self.calc.wfs.kd

        # frequency points init
        self.dw = self.w_w[1] - self.w_w[0]
        assert ((self.w_w[1:] - self.w_w[:-1] - self.dw) < 1e-10).all() # make sure its linear w grid
        assert self.w_w.max() == self.w_w[-1]
        assert self.w_w.min() == self.w_w[0]
        
        self.dw /= Hartree
        self.w_w  /= Hartree
        self.wmax = self.w_w[-1]
        self.wmin = self.w_w[0] 
        self.wcut = self.wmax + 5. / Hartree
        self.Nw  = int(self.wmax / self.dw) + 1
        self.NwS = int(self.wcut / self.dw) + 1

        self.nkptout = np.shape(self.kpoints)[0]
        self.nbandsout = np.shape(self.bands)[0]


    def get_QP_spectrum(self):

        self.initialize()
        calc = self.calc
        kd = self.kd
        nkpt = kd.nbzkpts
        nikpt = kd.nibzkpts
        nbands = self.nbands

        starttime = time()
        self.printtxt("------------------------------------------------")
        self.printtxt('starting calculation at %s' %(ctime(starttime)))

        # k-points init
        self.printtxt("------------------------------------------------")
        self.printtxt('calculate matrix elements for k = :')

        if (self.kpoints == None):
            nkptout = nikpt
            kptout = range(nikpt)
            for k in kptout:
                self.printtxt(kd.ibzk_kc[k])
        else:
            nkptout = np.shape(self.kpoints)[0]
            kptout = self.kpoints
            for k in kptout:
                self.printtxt(kd.bzk_kc[k])

        # bands init
        if (self.bands == None):
            nbandsout = nbands
            bandsout = range(nbands)
        else:
            nbandsout = np.shape(self.bands)[0]
            bandsout = self.bands
        self.printtxt("------------------------------------------------")
        self.printtxt('calculate matrix elements for n = :')
        self.printtxt(bandsout)

        self.printtxt("------------------------------------------------")
        self.printtxt("broadening (eV):")
        self.printtxt(self.eta*Hartree)
        self.printtxt("plane wave cut-off (eV):")
        self.printtxt(self.ecut*Hartree)
        self.printtxt("frequency range (eV):")
        self.printtxt('%.2f - %.2f in %.2f' %(self.wmin*Hartree, self.wmax*Hartree, self.dw*Hartree))
        self.printtxt("------------------------------------------------")
        self.printtxt("number of bands:")
        self.printtxt(nbands)
        self.printtxt("number of k-points:")
        self.printtxt(nkpt)
        self.printtxt("number of irreducible k-points:")
        self.printtxt(nikpt)
        self.printtxt("------------------------------------------------")

        bzq_kc = kd.get_bz_q_points()
        nqpt = np.shape(bzq_kc)[0]

        nG = calc.get_number_of_grid_points()
        acell_cv = self.acell_cv
        bcell_cv = self.bcell_cv
        vol = self.vol
        npw = self.npw
        Gvec_Gc = self.Gvec_Gc
        Nw = self.Nw

        Sigma_kn = np.zeros((nkptout, nbandsout), dtype=float)
        Z_kn = np.zeros((nkptout, nbandsout), dtype=float)

        qcomm = world
        nq, nq_local, q_start, q_end = parallel_partition(nqpt, world.rank, world.size, reshape=False)

        self.printtxt("calculating Sigma")
        self.printtxt("------------------------------------------------")

        for iq in range(q_start, q_end):
            q = bzq_kc[iq]

            q_G = np.zeros(npw, dtype=float)
            dfinv_wGG = np.zeros((Nw, npw, npw), dtype = complex)
            W_wGG = np.zeros((Nw, npw, npw), dtype = complex)
            tmp_GG = np.eye(npw, npw)

            self.printtxt('%i calculating q = [%f %f %f]' %(iq, q[0], q[1], q[2]))
            self.printtxt("------------------------------------------------")

            if (np.abs(q) < 1e-5).all():
                q0 = np.array([1e-10, 0., 0.])
                optical_limit = True
            else:
                q0 = q
                optical_limit = False

            qG = np.dot(q[np.newaxis,:] + Gvec_Gc,(bcell_cv).T)
            q_G = 1. / np.sqrt((qG*qG).sum(axis=1))

            df = DF(
                    calc=calc,
                    q=q0,
                    w=self.w_w.copy() * Hartree,
                    nbands=nbands,
                    optical_limit=False,
                    hilbert_trans=True,
                    full_response=True,
                    xc='RPA',
                    eta=copy.copy(self.eta)*Hartree,
                    ecut=copy.copy(self.ecut)*Hartree,
                    txt='df_q_' + str(iq) + '.out',
                    comm=serial_comm
                   )

            dfinv_wGG = df.get_inverse_dielectric_matrix(xc='RPA')

            for iw in range(Nw):
                W_wGG[iw] =  4*pi * ((q_G[:,np.newaxis] * (dfinv_wGG[iw] - tmp_GG)) * q_G[np.newaxis,:])
                if optical_limit:
                    W_wGG[iw,0,0:] = 0.
                    W_wGG[iw,0:,0] = 0.

            del q_G, dfinv_wGG

            S, Z = self.get_self_energy(df, W_wGG)

            Sigma_kn += S
            Z_kn += Z

            del q0, q, df

        qcomm.barrier()
        qcomm.sum(Sigma_kn)
        qcomm.sum(Z_kn)

        Z_kn /= nkpt
        Sigma_kn /= nkpt

        self.printtxt("calculating V_XC")
        self.printtxt("------------------------------------------------")
        v_xc = vxc(calc)

#        calc.set(parallel={'domain': 1})

        self.printtxt("calculating E_XX")
        alpha = 5.0
        exx = HybridXC('EXX', alpha=alpha)
        calc.get_xc_difference(exx)

        e_kn = np.zeros((nkptout, nbandsout), dtype=float)
        v_kn = np.zeros((nkptout, nbandsout), dtype=float)
        e_xx = np.zeros((nkptout, nbandsout), dtype=float)

        i = 0
        for k in kptout:
            j = 0
            ik = kd.bz2ibz_k[k]
            for n in bandsout:
                e_kn[i][j] = calc.get_eigenvalues(kpt=ik)[n] / Hartree
                v_kn[i][j] = v_xc[0][ik][n] / Hartree
                e_xx[i][j] = exx.exx_skn[0][ik][n]
                j += 1
            i += 1

        self.printtxt("------------------------------------------------")
        self.printtxt("LDA eigenvalues are (eV):")
        self.printtxt("------------------------------------------------")
        self.printtxt(e_kn*Hartree)
        self.printtxt("------------------------------------------------")
        self.printtxt("LDA exchange-correlation contributions are (eV):")
        self.printtxt("------------------------------------------------")
        self.printtxt(v_kn*Hartree)
        self.printtxt("------------------------------------------------")
        self.printtxt("exact exchange contributions are (eV):")
        self.printtxt("------------------------------------------------")
        self.printtxt(e_xx*Hartree)
        self.printtxt("------------------------------------------------")
        self.printtxt("correlation contributions are (eV):")
        self.printtxt("------------------------------------------------")
        self.printtxt(Sigma_kn*Hartree)
        self.printtxt("------------------------------------------------")
        self.printtxt("renormalization factors are:")
        self.printtxt("------------------------------------------------")
        self.printtxt(Z_kn)

        Sigma_kn = e_kn + Z_kn * (Sigma_kn + e_xx - v_kn)

        totaltime = round(time() - starttime)

        self.printtxt("------------------------------------------------")
        self.printtxt("GW calculation finished!")
        self.printtxt('in %s' %(timedelta(seconds=totaltime)))
        self.printtxt("------------------------------------------------")
        self.printtxt("Quasi-particle energies are (eV):")
        self.printtxt("------------------------------------------------")
        self.printtxt(Sigma_kn*Hartree)



    def get_self_energy(self, df, W_wGG):

#        self.initialize()

        Sigma_kn = np.zeros((self.nkptout, self.nbandsout), dtype=complex)
        Z_kn = np.zeros((self.nkptout, self.nbandsout), dtype=float)
        rho_G = np.zeros(self.npw, dtype=complex)

        Cplus_wGG = np.zeros((self.NwS, self.npw, self.npw), dtype=complex)
        Cminus_wGG = np.zeros((self.NwS, self.npw, self.npw), dtype=complex)

        for iw in range(self.NwS):
            w1 = iw * self.dw
            for jw in range(self.Nw):
                w2 = jw * self.dw
                Cplus_wGG[iw] += W_wGG[jw] * (1. / (w1 + w2 + 1j*self.eta) + 1. / (w1 - w2 + 1j*self.eta))
                Cminus_wGG[iw] += W_wGG[jw] * (1. / (w1 + w2 - 1j*self.eta) + 1. / (w1 - w2 - 1j*self.eta))

        Cplus_wGG *= 1j/(2*pi) * self.dw
        Cminus_wGG *= 1j/(2*pi) * self.dw

        i = 0
        for k in self.kpoints:

            kq = df.kq_k[k]
            ibzkpt1 = df.kd.bz2ibz_k[k]
            ibzkpt2 = df.kd.bz2ibz_k[kq]

            j = 0
            for n in self.bands:

                for m in range(self.nbands):
                    check_focc = self.f_kn[ibzkpt2, m] > 1e-3
                    if check_focc:
                        pm = -1
                    else:
                        pm = 1

                    if not self.e_kn[ibzkpt2,m] - self.e_kn[ibzkpt1,n] == 0:
                        pm *= np.sign(self.e_kn[ibzkpt1,n] - self.e_kn[ibzkpt2,m])

                    rho_G = self.density_matrix(n, m, k, kq, df.phi_aGp)
                    rho_GG = np.outer(rho_G, rho_G.conj())

                    w0 = self.e_kn[ibzkpt2,m] - self.e_kn[ibzkpt1,n]
                    w0_id = np.abs(int(w0 / self.dw))
                    w1 = w0_id * self.dw
                    w2 = (w0_id + 1) * self.dw

                    if pm == 1:
                        Sw1 = 1. / self.vol * np.sum(Cplus_wGG[w0_id] * rho_GG)
                        Sw2 = 1. / self.vol * np.sum(Cplus_wGG[w0_id + 1] * rho_GG)
                    if pm == -1:
                        Sw1 = 1. / self.vol * np.sum(Cminus_wGG[w0_id] * rho_GG)
                        Sw2 = 1. / self.vol * np.sum(Cminus_wGG[w0_id + 1] * rho_GG)

                    Sw0 = (w2-np.abs(w0))/self.dw * Sw1 + (np.abs(w0)-w1)/self.dw * Sw2

                    Sigma_kn[i][j] = Sigma_kn[i][j] + np.sign(self.e_kn[ibzkpt1,n] - self.e_kn[ibzkpt2,m])*Sw0
                    Z_kn[i][j] = Z_kn[i][j] + 1./(1 - np.real((Sw2 - Sw1)/(w2 - w1)))

                j+=1
            i+=1
        return np.real(Sigma_kn), Z_kn/self.nbands
