import numpy as np
from math import pi, sqrt
from time import time, ctime
from datetime import timedelta
from ase.parallel import paropen
from ase.units import Hartree, Bohr
from gpaw.mpi import world, rank, size, serial_comm
from gpaw.xc.hybridk import HybridXC
from gpaw.xc.tools import vxc
from gpaw.response.parallel import parallel_partition
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

        self.vcut = None
        self.bands = bands
        self.kpoints = kpoints

    def initialize(self):

        self.printtxt('-----------------------------------------------')
        self.printtxt('GW calculation started at: \n')
        self.printtxt(ctime())
        self.starttime = time()
        
        BASECHI.initialize(self)
        calc = self.calc
        self.kd = kd = self.calc.wfs.kd
        self.nkpt = kd.nbzkpts

        # q point init
        self.bzq_kc = kd.get_bz_q_points()
        self.ibzq_qc = self.bzq_kc # q point symmetry is not used at the moment.
        self.nqpt = np.shape(self.bzq_kc)[0]
        self.qcomm = world
        nq, self.nq_local, self.q_start, self.q_end = parallel_partition(
                                  self.nqpt, world.rank, world.size, reshape=False)
        
        
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


        # GW kpoints init
        if (self.kpoints == None):
            self.gwnkpt = self.nkpt
            self.gwkpt_k = range(self.nkpt)
        else:
            self.gwnkpt = np.shape(self.kpoints)[0]
            self.gwkpt_k = self.kpoints

        # GW bands init
        if (self.bands == None):
            self.gwnband = self.nbands
            self.gwbands_n = range(self.nbands)
        else:
            self.gwnband = np.shape(self.bands)[0]
            self.gwbands_n = self.bands

        # print init
        self.print_gw_init()
        

    def get_QP_spectrum(self):
        
        self.initialize()
        
        self.printtxt("calculating Sigma")

        Sigma_kn = np.zeros((self.gwnkpt, self.gwnband), dtype=float)
        Z_kn = np.zeros((self.gwnkpt, self.gwnband), dtype=float)

        t0 = time()
        for iq in range(self.q_start, self.q_end):

            # get screened interaction. 
            df, W_wGG = self.screened_interaction_kernel(iq, static=False)
            # to be checked.
            q = self.bzq_kc[iq]
            if (np.abs(q) < 1e-5).all():
                W_wGG[:,0,0:] = 0.
                W_wGG[:,0:,0] = 0.

            # get self energy
            S, Z = self.get_self_energy(df, W_wGG)
            Sigma_kn += S
            Z_kn += Z
            
            del df
            self.timing(iq, t0, self.nq_local, 'iq')

        self.qcomm.barrier()
        self.qcomm.sum(Sigma_kn)
        self.qcomm.sum(Z_kn)

        Z_kn /= self.nkpt
        Sigma_kn /= self.nkpt

        # exact exchange
        e_kn, v_kn, e_xx = self.get_exx() # not, e_kn is different from self.e_kn
        Sigma_kn = e_kn + Z_kn * (Sigma_kn + e_xx - v_kn)

        # finish
        self.print_gw_finish(e_kn, v_kn, e_xx, Sigma_kn, Z_kn)


    def get_self_energy(self, df, W_wGG):

        Sigma_kn = np.zeros((self.gwnkpt, self.gwnband), dtype=complex)
        Z_kn = np.zeros((self.gwnkpt, self.gwnband), dtype=float)
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


    def get_exx(self):

        self.printtxt("calculating Exact exchange and E_XC ")
        v_xc = vxc(self.calc)

        alpha = 5.0
        exx = HybridXC('EXX', alpha=alpha)
        self.calc.get_xc_difference(exx)

        e_kn = np.zeros((self.gwnkpt, self.gwnband), dtype=float)
        v_kn = np.zeros((self.gwnkpt, self.gwnband), dtype=float)
        e_xx = np.zeros((self.gwnkpt, self.gwnband), dtype=float)

        i = 0
        for k in self.gwkpt_k:
            j = 0
            ik = self.kd.bz2ibz_k[k]
            for n in self.gwbands_n:
                e_kn[i][j] = self.calc.get_eigenvalues(kpt=ik)[n] / Hartree
                v_kn[i][j] = v_xc[0][ik][n] / Hartree
                e_xx[i][j] = exx.exx_skn[0][ik][n]
                j += 1
            i += 1

        return e_kn, v_kn, e_xx


    def print_gw_init(self):

        self.printtxt("Number of IBZ k-points       : %d" %(self.kd.nibzkpts))
        self.printtxt("Frequency range (eV)         : %.2f - %.2f in %.2f" %(self.wmin*Hartree, self.wmax*Hartree, self.dw*Hartree))
        self.printtxt('')
        self.printtxt('Calculate matrix elements for k = :')
        for k in self.gwkpt_k:
            self.printtxt(self.kd.bzk_kc[k])
        self.printtxt('')
        self.printtxt('Calculate matrix elements for n = %s' %(self.gwbands_n))


    def print_gw_finish(self, e_kn, v_kn, e_xx, Sigma_kn, Z_kn):

        self.printtxt("------------------------------------------------")
        self.printtxt("LDA eigenvalues are (eV): ")
        self.printtxt("%s \n" %(e_kn*Hartree))
        self.printtxt("LDA exchange-correlation contributions are (eV): ")
        self.printtxt("%s \n" %(v_kn*Hartree))
        self.printtxt("Exact exchange contributions are (eV): ")
        self.printtxt("%s \n" %(e_xx*Hartree))
        self.printtxt("Self energy contributions are (eV):")
        self.printtxt("%s \n" %(Sigma_kn*Hartree))
        self.printtxt("Renormalization factors are:")
        self.printtxt("%s \n" %(Z_kn))

        totaltime = round(time() - self.starttime)
        self.printtxt("GW calculation finished in %s " %(timedelta(seconds=totaltime)))
        self.printtxt("------------------------------------------------")
        self.printtxt("Quasi-particle energies are (eV): ")
        self.printtxt(Sigma_kn*Hartree)
