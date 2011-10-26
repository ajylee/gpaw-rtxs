import numpy as np
from math import pi, sqrt
from time import time, ctime
from datetime import timedelta
from ase.parallel import paropen
from ase.units import Hartree, Bohr
from gpaw.mpi import world, rank, size, serial_comm
from gpaw.utilities.blas import gemmdot
from gpaw.xc.hybridk import HybridXC
from gpaw.xc.tools import vxc
from gpaw.response.parallel import parallel_partition
from gpaw.response.base import BASECHI

class GW(BASECHI):

    def __init__(
                 self,
                 file=None,
                 nbands=None,
                 bands=None,
                 kpoints=None,
                 w=None,
                 ecut=150.,
                 eta=0.1,
                 hilbert_trans=False,
                 wpar=False,
                 vcut=None,
                 txt=None,
                ):

        BASECHI.__init__(self, calc=file, nbands=nbands, w=w, ecut=ecut, eta=eta, txt=txt)

        self.vcut = vcut
        self.bands = bands
        self.kpoints = kpoints
        self.hilbert_trans = hilbert_trans
        self.wpar = wpar

    def initialize(self):

        self.printtxt('-----------------------------------------------')
        self.printtxt('GW calculation started at: \n')
        self.printtxt(ctime())
        self.starttime = time()
        
        BASECHI.initialize(self)
        calc = self.calc
        self.kd = kd = self.calc.wfs.kd
        self.nkpt = kd.nbzkpts
        self.nikpt = kd.nibzkpts

        # q point init
        self.bzq_kc = kd.get_bz_q_points()
        self.ibzq_qc = self.bzq_kc # q point symmetry is not used at the moment.
        self.nqpt = np.shape(self.bzq_kc)[0]
        
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
#        self.Nw  = int(self.wmax / self.dw) + 1
#        self.NwS = int(self.wcut / self.dw) + 1

        # GW kpoints init
        if (self.kpoints == None):
            self.gwnkpt = self.nikpt
            self.gwkpt_k = kd.ibz2bz_k
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

        self.alpha = 1j/(2*pi) * self.dw / (self.vol * self.nkpt)

        # print init
        self.print_gw_init()
        
        # parallel init
        if self.hilbert_trans and self.wpar:
            self.printtxt("-----")
            self.printtxt("frequency parallelization not yet implemented for method 2, switch to method 1")
            self.printtxt("-----")
            self.hilbert_trans = False
        if not self.wpar:
            self.qcomm = world
            nq, self.nq_local, self.q_start, self.q_end = parallel_partition(
                                      self.nqpt, world.rank, world.size, reshape=False)
            self.kcommsize = None
            self.parallel = serial_comm
        else:
            self.qcomm = serial_comm
            self.nq_local = self.nqpt
            self.q_start = 0
            self.q_end = self.nqpt
            self.kcommsize = 1
            self.parallel = None

    def get_QP_spectrum(self):
        
        self.initialize()
        
        self.printtxt("calculating Sigma")

        Sigma_kn = np.zeros((self.gwnkpt, self.gwnband), dtype=float)
        dSigma_kn = np.zeros((self.gwnkpt, self.gwnband), dtype=float)
        Z_kn = np.zeros((self.gwnkpt, self.gwnband), dtype=float)

        t0 = time()
        t_w = 0
        t_selfenergy = 0
        for iq in range(self.q_start, self.q_end):
            if iq >= self.nqpt:
                continue
            t1 = time()
            # get screened interaction. 
            df, W_wGG = self.screened_interaction_kernel(iq, static=False, comm=self.parallel, kcommsize=self.kcommsize)
            t2 = time()
            t_w += t2 - t1

            # get self energy
            S, dS = self.get_self_energy(df, W_wGG)
            t3 = time() - t2
            t_selfenergy += t3
            
            Sigma_kn += S
            dSigma_kn += dS
            
            del df
            self.timing(iq, t0, self.nq_local, 'iq')

        self.printtxt('W_wGG takes %f seconds' %(t_w))
        self.printtxt('Self energy takes %f  seconds' %(t_selfenergy))

        self.qcomm.barrier()
        self.qcomm.sum(Sigma_kn)
        self.qcomm.sum(dSigma_kn)

        Z_kn = 1. / (1. - dSigma_kn)

        # exact exchange
        t0 = time()
        e_kn, v_kn, e_xx = self.get_exx() # note, e_kn is different from self.e_kn
        self.printtxt('EXX takes %f seconds' %(time()-t0))

        QP_kn = e_kn + Z_kn * (Sigma_kn + e_xx - v_kn)
        self.QP_kn = QP_kn

        # finish
        self.print_gw_finish(e_kn, v_kn, e_xx, Sigma_kn, Z_kn, QP_kn)


    def get_self_energy(self, df, W_wGG):

        Sigma_kn = np.zeros((self.gwnkpt, self.gwnband), dtype=float)
        dSigma_kn = np.zeros((self.gwnkpt, self.gwnband), dtype=float)
        E_f = self.calc.get_fermi_level() / Hartree

        wcomm = df.wcomm

        # prepare optical limit for both methods
        if df.optical_limit:
            tmp_wG = np.zeros((df.Nw_local, self.npw), dtype=complex)
            q = np.array([0.0001,0,0])
            for jG in range(1, self.npw):
                qG = np.dot(q+self.Gvec_Gc[jG], self.bcell_cv)
                tmp_wG[:,jG] = self.dfinvG0_wG[:,jG] / np.sqrt(np.inner(qG,qG))
                const = 1./pi*self.vol*(6*pi**2/self.vol)**(2./3.)
            tmp_wG *= const * self.nkpt**(1./3.)
            tmp_w = 2./pi*(6*pi**2/self.vol)**(1./3.) * self.dfinvG0_wG[:,0] * self.vol
            tmp_w *= self.nkpt**(2./3.)

        if not self.hilbert_trans: #method 1
            Wbackup_wG0 = W_wGG[:,:,0].copy()
            Wbackup_w0G = W_wGG[:,0,:].copy()

        else: #method 2, perform Hilbert transform
            nG = np.shape(W_wGG)[1]
            coords = np.zeros(wcomm.size, dtype=int)
            nG_local = nG**2 // wcomm.size
            if wcomm.rank == wcomm.size - 1:
                  nG_local = nG**2 - (wcomm.size - 1) * nG_local
            wcomm.all_gather(np.array([nG_local]), coords)
            W_Wg = SliceAlongFrequency(W_wGG, coords, wcomm)

            Cplus_Wg = np.zeros_like(W_Wg)
            Cminus_Wg = np.zeros_like(W_Wg)
            Cplus_wG0 = np.zeros((2, nG), dtype=complex)
            Cminus_wG0 = np.zeros((2, nG), dtype=complex)

            if df.optical_limit:
                tmp_WG = np.zeros((df.Nw, nG), dtype=complex)
                wcomm.all_gather(tmp_wG, tmp_WG)
                del tmp_wG
                tmp_W = np.zeros(df.Nw, dtype=complex)
                wcomm.all_gather(tmp_w, tmp_W)
                del tmp_w

            w2_w = np.arange(df.Nw) * self.dw
            for iw in range(df.Nw):
                w1 = iw * self.dw
                w1_w = 1. / (w1 + w2_w + 1j*self.eta) + 1. / (w1 - w2_w + 1j*self.eta)
                Cplus_Wg[iw] = gemmdot(w1_w, W_Wg, beta=0.0)
                Cminus_Wg[iw] = gemmdot(w1_w.conj(), W_Wg, beta=0.0)
                # special Hilbert transform optical limit:
                if df.optical_limit:
                    if iw < 2:
                        Cplus_wG0[iw] = gemmdot(w1_w, tmp_WG, beta=0.0)
                        Cminus_wG0[iw] = gemmdot(w1_w.conj(), tmp_WG, beta=0.0)
                        Cplus_wG0[iw,0] = gemmdot(w1_w, tmp_W, beta=0.0)
                        Cminus_wG0[iw,0] = gemmdot(w1_w.conj(), tmp_W, beta=0.0)

        for i, k in enumerate(self.gwkpt_k): # k is bzk index
            if df.optical_limit:
                kq_c = df.kd.bzk_kc[k]
            else:
                kq_c = df.kd.bzk_kc[k] - df.q_c  # k - q
            
            kq = df.kd.where_is_q(kq_c, df.kd.bzk_kc)            
            assert df.kq_k[kq] == k
            ibzkpt1 = df.kd.bz2ibz_k[k]
            ibzkpt2 = df.kd.bz2ibz_k[kq]

            for j, n in enumerate(self.bands):
                for m in range(self.nbands):

                    if self.f_kn[ibzkpt2,m] < self.ftol: #self.e_kn[ibzkpt2, m] > E_f :
                        sign = 1.
                    else:
                        sign = -1.

                    rho_G = df.density_matrix(m, n, kq)

                    if not self.hilbert_trans: #method 1
                        W_wGG[:,:,0] = Wbackup_wG0
                        W_wGG[:,0,:] = Wbackup_w0G

                        if df.optical_limit:
                            if n==m:
                                W_wGG[:,:,0] = tmp_wG
                                W_wGG[:,0,:] = tmp_wG.conj()
                                W_wGG[:,0,0] = tmp_w
                            else:
                                # to be checked.
                                W_wGG[:,0,0:] = 0.
                                W_wGG[:,0:,0] = 0.

                        # perform W_wGG * np.outer(rho_G.conj(), rho_G).sum(GG)
                        W_wG = gemmdot(W_wGG, rho_G, beta=0.0)
                        C_wlocal = gemmdot(W_wG, rho_G, alpha=self.alpha, beta=0.0,trans='c')
                        del W_wG, rho_G

                        C_w = np.zeros(df.Nw, dtype=complex)
                        wcomm.all_gather(C_wlocal, C_w)
                        del C_wlocal

                        # w1 = w - epsilon_m,k-q + i*eta * sgn(epsilon_m,k-q, E_f)
                        w1 = self.e_kn[ibzkpt1, n] - self.e_kn[ibzkpt2,m] + 1j*self.eta*sign
                        w2_w = np.arange(df.Nw) * self.dw

                        # calculate self energy
                        w1_w = 1./(w1 - w2_w) + 1./(w1 + w2_w)
                        Pw = np.sum(w1_w)*self.dw - self.dw/w1
                        Sigma_kn[i,j] += np.real(gemmdot(C_w, w1_w, beta=0.0))

                        # calculate derivate of self energy with respect to w
                        w1_w = 1./(w1 - w2_w)**2 + 1./(w1 + w2_w)**2
                        dSigma_kn[i,j] -= np.real(gemmdot(C_w, w1_w, beta=0.0))

                    else: #method 2
                        if not self.e_kn[ibzkpt2,m] - self.e_kn[ibzkpt1,n] == 0:
                            sign *= np.sign(self.e_kn[ibzkpt1,n] - self.e_kn[ibzkpt2,m])

                        # find points on frequency grid
                        w0 = self.e_kn[ibzkpt1,n] - self.e_kn[ibzkpt2,m]
                        w0_id = np.abs(int(w0 / self.dw))
                        w1 = w0_id * self.dw
                        w2 = (w0_id + 1) * self.dw

                        # choose plus or minus, treat optical limit:
                        if sign == 1:
                            C_Wg = Cplus_Wg[w0_id:w0_id+2] # only two grid points needed for each w0
                        if sign == -1:
                            C_Wg = Cminus_Wg[w0_id:w0_id+2] # only two grid points needed for each w0

                        C_wGG = np.zeros((2, nG**2), dtype=complex)
                        wcomm.all_gather(C_Wg, C_wGG)
                        del C_Wg
                        C_wGG.reshape(2, nG, nG)

                        if df.optical_limit:
                            if n==m:
                                if sign == 1:
                                    C_wGG[:,0,:] = Cminus_wG0.conj()
                                    C_wGG[:,:,0] = Cplus_wG0
                                if sign == -1:
                                    C_wGG[:,0,:] = Cplus_wG0.conj()
                                    C_wGG[:,:,0] = Cminus_wG0
                            else:
                                C_wGG[:,0,0:] = 0.
                                C_wGG[:,0:,0] = 0.

                        # special treat of w0 = 0 (degenerate states):
                        if w0_id == 0:
                            Cplustmp_GG = Cplus_wGG[1]
                            Cminustmp_GG = Cminus_wGG[1]
                            if df.optical_limit:
                                if n==m:
                                    Cplustmp_GG[0,:] = Cminus_wG0.conj()[1]
                                    Cplustmp_GG[:,0] = Cplus_wG0[1]
                                    Cminustmp_GG[0,:] = Cplus_wG0.conj()[1]
                                    Cminustmp_GG[:,0] = Cminus_wG0[1]
                                else:
                                    Cplustmp_GG[0,:] = 0.
                                    Cplustmp_GG[:,0] = 0.
                                    Cminustmp_GG[0,:] = 0.
                                    Cminustmp_GG[:,0] = 0.

                        # perform C_wGG * np.outer(rho_G.conj(), rho_G).sum(GG)

                        if w0_id == 0:
                            Sw0_G = gemmdot(C_wGG[0], rho_G, beta=0.0)
                            Sw0 = np.real(gemmdot(Sw0_G, rho_G, alpha=self.alpha, beta=0.0, trans='c'))
                            Sw1_G = gemmdot(Cplustmp_GG, rho_G, beta=0.0)
                            Sw1 = np.real(gemmdot(Sw1_G, rho_G, alpha=self.alpha, beta=0.0, trans='c'))
                            Sw2_G = gemmdot(Cminustmp_GG, rho_G, beta=0.0)
                            Sw2 = np.real(gemmdot(Sw2_G, rho_G, alpha=self.alpha, beta=0.0, trans='c'))

                            Sigma_kn[i,j] += Sw0
                            dSigma_kn[i,j] += (Sw1 + Sw2)/(2*self.dw)

                        else:
                            Sw1_G = gemmdot(C_wGG[0], rho_G, beta=0.0)
                            Sw1 = np.real(gemmdot(Sw1_G, rho_G, alpha=self.alpha, beta=0.0, trans='c'))
                            Sw2_G = gemmdot(C_wGG[1], rho_G, beta=0.0)
                            Sw2 = np.real(gemmdot(Sw2_G, rho_G, alpha=self.alpha, beta=0.0, trans='c'))

                            Sw0 = (w2-np.abs(w0))/self.dw * Sw1 + (np.abs(w0)-w1)/self.dw * Sw2
                            Sigma_kn[i,j] += np.sign(self.e_kn[ibzkpt1,n] - self.e_kn[ibzkpt2,m]) * Sw0
                            dSigma_kn[i,j] += (Sw2 - Sw1)/self.dw

        return Sigma_kn, dSigma_kn 


    def get_exx(self):

        self.printtxt("calculating Exact exchange and E_XC ")
        v_xc = vxc(self.calc)

        alpha = 5.0
        exx = HybridXC('EXX', alpha=alpha, ecut=self.ecut.max(), bands=self.bands)
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


    def print_gw_finish(self, e_kn, v_kn, e_xx, Sigma_kn, Z_kn, QP_kn):

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
        self.printtxt(QP_kn*Hartree)
