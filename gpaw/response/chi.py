import sys
from time import time, ctime
import numpy as np
from math import sqrt, pi
from ase.units import Hartree, Bohr
from gpaw import extra_parameters
from gpaw.utilities.blas import gemv, scal, axpy
from gpaw.mpi import world, rank, size, serial_comm
from gpaw.fd_operators import Gradient
from gpaw.response.math_func import hilbert_transform
from gpaw.response.parallel import set_communicator, \
     parallel_partition, SliceAlongFrequency, SliceAlongOrbitals
from gpaw.response.kernel import calculate_Kxc, calculate_Kc
from gpaw.utilities.memory import maxrss
from gpaw.response.base import BASECHI

class CHI(BASECHI):
    """This class is a calculator for the linear density response function.

    Parameters:

        nband: int
            Number of bands.
        wmax: float
            Maximum energy for spectrum.
        dw: float
            Frequency interval.
        wlist: tuple
            Frequency points.
        q: ndarray
            Momentum transfer in reduced coordinate.
        Ecut: ndarray
            Planewave cutoff energy.
        eta: float
            Spectrum broadening factor.
        sigma: float
            Width for delta function.
    """

    def __init__(self,
                 calc=None,
                 nbands=None,
                 w=None,
                 q=None,
                 eshift=None,
                 ecut=10.,
                 G_plus_q=False,
                 eta=0.2,
                 rpad=np.array([1,1,1]),
                 vcut=None,
                 ftol=1e-5,
                 txt=None,
                 xc='ALDA',
                 hilbert_trans=True,
                 full_response=False,
                 optical_limit=False,
                 comm=None,
                 kcommsize=None):

        BASECHI.__init__(self, calc=calc, nbands=nbands, w=w, q=q,
                         eshift=eshift, ecut=ecut, G_plus_q=G_plus_q, eta=eta,
                         rpad=rpad, ftol=ftol, txt=txt,
                         optical_limit=optical_limit)
        
        self.xc = xc
        self.hilbert_trans = hilbert_trans
        self.full_hilbert_trans = full_response
        self.vcut = vcut
        self.kcommsize = kcommsize
        self.comm = comm
        if self.comm is None:
            self.comm = world
        self.chi0_wGG = None


    def initialize(self, do_Kxc=False, simple_version=False):

        self.printtxt('')
        self.printtxt('-----------------------------------------')
        self.printtxt('Response function calculation started at:')
        self.starttime = time()
        self.printtxt(ctime())

        BASECHI.initialize(self)

        # Frequency init
        self.dw = None
        if len(self.w_w) == 1:
            self.HilberTrans = False

        if self.hilbert_trans:
            self.dw = self.w_w[1] - self.w_w[0]
            assert ((self.w_w[1:] - self.w_w[:-1] - self.dw) < 1e-10).all() # make sure its linear w grid
            assert self.w_w.max() == self.w_w[-1]
            
            self.dw /= Hartree
            self.w_w  /= Hartree
            self.wmax = self.w_w[-1] 
            self.wcut = self.wmax + 5. / Hartree
            self.Nw  = int(self.wmax / self.dw) + 1
            self.NwS = int(self.wcut / self.dw) + 1
        else:
            self.Nw = len(self.w_w)
            self.NwS = 0
            if len(self.w_w) > 1:
                self.dw = self.w_w[1] - self.w_w[0]
                assert ((self.w_w[1:] - self.w_w[:-1] - self.dw) < 1e-10).all()
                self.dw /= Hartree
                
        if self.hilbert_trans:
            # for band parallelization.
            for n in range(self.nbands):
                if (self.f_kn[:, n] - self.ftol < 0).all():
                    self.nvalbands = n
                    break
        else:
            # if not hilbert transform, all the bands should be used.
            self.nvalbands = self.nbands

        # Parallelization initialize
        self.parallel_init()

        # Printing calculation information
        self.print_chi()

        if extra_parameters.get('df_dry_run'):
            raise SystemExit

        calc = self.calc

        # For LCAO wfs
        if calc.input_parameters['mode'] == 'lcao':
            calc.initialize_positions()        
        self.printtxt('     GS calculator   : %f M / cpu' %(maxrss() / 1024**2))

        if simple_version is True:
            return
        # PAW part init
        # calculate <phi_i | e**(-i(q+G).r) | phi_j>
        # G != 0 part
        self.phi_aGp = self.get_phi_aGp()
        self.printtxt('Finished phi_aGp !')

        # Calculate Coulomb kernel
        self.Kc_GG = calculate_Kc(self.q_c, self.Gvec_Gc, self.acell_cv,
                                  self.bcell_cv, self.calc.atoms.pbc, self.optical_limit, self.vcut)

        # Calculate ALDA kernel (not used in chi0)
        R_av = calc.atoms.positions / Bohr
        if self.xc == 'RPA': #type(self.w_w[0]) is float:
            self.Kxc_GG = np.zeros((self.npw, self.npw))
            self.printtxt('RPA calculation.')
        elif self.xc == 'ALDA':
            nt_sg = calc.density.nt_sG
            if (self.rpad > 1).any() or (self.pbc - True).any():
                nt_sG = np.zeros([self.nspins, self.nG[0], self.nG[1], self.nG[2]])
                for s in range(self.nspins):
                    nt_G = self.pad(nt_sg[s])
                    nt_sG[s] = nt_G
            else:
                nt_sG = nt_sg
            
            self.Kxc_GG = calculate_Kxc(self.gd, # global grid
                                        nt_sG,
                                        self.npw, self.Gvec_Gc,
                                        self.nG, self.vol,
                                        self.bcell_cv, R_av,
                                        calc.wfs.setups,
                                        calc.density.D_asp)

            self.printtxt('Finished ALDA kernel ! ')
        else:
            raise ValueError('%s Not implemented !' %(self.xc))
        
        return


    def calculate(self, spin=0):
        """Calculate the non-interacting density response function. """

        calc = self.calc
        kd = self.kd
        gd = self.gd
        sdisp_cd = gd.sdisp_cd
        ibzk_kc = self.ibzk_kc
        bzk_kc = self.bzk_kc
        kq_k = self.kq_k
        pt = self.pt
        f_kn = self.f_kn
        e_kn = self.e_kn

        # Matrix init
        chi0_wGG = np.zeros((self.Nw_local, self.npw, self.npw), dtype=complex)
        if not (f_kn > self.ftol).any():
            self.chi0_wGG = chi0_wGG
            return

        if self.hilbert_trans:
            specfunc_wGG = np.zeros((self.NwS_local, self.npw, self.npw), dtype = complex)

        # Prepare for the derivative of pseudo-wavefunction
        if self.optical_limit:
            d_c = [Gradient(gd, i, n=4, dtype=complex).apply for i in range(3)]
            dpsit_g = gd.empty(dtype=complex)
            tmp = np.zeros((3), dtype=complex)

        rho_G = np.zeros(self.npw, dtype=complex)
        t0 = time()
        t_get_wfs = 0
        for k in range(self.kstart, self.kend):
            k_pad = False
            if k >= self.nkpt:
                k = 0
                k_pad = True

            # Find corresponding kpoint in IBZ
            ibzkpt1 = kd.bz2ibz_k[k]
            if self.optical_limit:
                ibzkpt2 = ibzkpt1
            else:
                ibzkpt2 = kd.bz2ibz_k[kq_k[k]]
            
            for n in range(self.nstart, self.nend):
#                print >> self.txt, k, n, t_get_wfs, time() - t0
                t1 = time()
                psitold_g = self.get_wavefunction(ibzkpt1, n, True, spin=spin)
                t_get_wfs += time() - t1
                psit1new_g_tmp = kd.transform_wave_function(psitold_g, k)

                if (self.rpad > 1).any() or (self.pbc - True).any():
                    psit1new_g = self.pad(psit1new_g_tmp)
                else:
                    psit1new_g = psit1new_g_tmp

                P1_ai = pt.dict()
                pt.integrate(psit1new_g, P1_ai, k)

                psit1_g = psit1new_g.conj() * self.expqr_g

                for m in range(self.nbands):
                    if k == 0 and n == 0:
                        print >> self.txt, k, n, m, time() - t0

		    if self.hilbert_trans:
			check_focc = (f_kn[ibzkpt1, n] - f_kn[ibzkpt2, m]) > self.ftol
                    else:
                        check_focc = np.abs(f_kn[ibzkpt1, n] - f_kn[ibzkpt2, m]) > self.ftol

                    t1 = time()
                    psitold_g = self.get_wavefunction(ibzkpt2, m, check_focc, spin=spin)
                    t_get_wfs += time() - t1

                    if check_focc:
                        psit2_g_tmp = kd.transform_wave_function(psitold_g, kq_k[k])
                        if (self.rpad > 1).any() or (self.pbc - True).any():
                            psit2_g = self.pad(psit2_g_tmp)
                        else:
                            psit2_g = psit2_g_tmp

                        P2_ai = pt.dict()
                        pt.integrate(psit2_g, P2_ai, kq_k[k])

                        # fft
                        tmp_g = np.fft.fftn(psit2_g*psit1_g) * self.vol / self.nG0

                        for iG in range(self.npw):
                            index = self.Gindex_G[iG]
                            rho_G[iG] = tmp_g[index[0], index[1], index[2]]

                        if self.optical_limit:
                            phase_cd = np.exp(2j * pi * sdisp_cd * bzk_kc[kq_k[k], :, np.newaxis])
                            for ix in range(3):
                                d_c[ix](psit2_g, dpsit_g, phase_cd)
                                tmp[ix] = gd.integrate(psit1_g * dpsit_g)
                            rho_G[0] = -1j * np.dot(self.qq_v, tmp)

                        # PAW correction
                        for a, id in enumerate(calc.wfs.setups.id_a):
                            P_p = np.outer(P1_ai[a].conj(), P2_ai[a]).ravel()
                            gemv(1.0, self.phi_aGp[a], P_p, 1.0, rho_G)

                        if self.optical_limit:
                            rho_G[0] /= self.enoshift_kn[ibzkpt2, m] - self.enoshift_kn[ibzkpt1, n]

                        if k_pad:
                            rho_G[:] = 0.
                        rho_GG = np.outer(rho_G, rho_G.conj())
                        
                        if not self.hilbert_trans:
                            for iw in range(self.Nw_local):
                                w = self.w_w[iw + self.wstart] / Hartree
                                C =  (f_kn[ibzkpt1, n] - f_kn[ibzkpt2, m]) / (
                                     w + e_kn[ibzkpt1, n] - e_kn[ibzkpt2, m] + 1j * self.eta)
                                axpy(C, rho_GG, chi0_wGG[iw])
                        else:
                            focc = f_kn[ibzkpt1,n] - f_kn[ibzkpt2,m]
                            w0 = e_kn[ibzkpt2,m] - e_kn[ibzkpt1,n]
                            scal(focc, rho_GG)

                            # calculate delta function
                            w0_id = int(w0 / self.dw)
                            if w0_id + 1 < self.NwS:
                                # rely on the self.NwS_local is equal in each node!
                                if self.wScomm.rank == w0_id // self.NwS_local:
                                    alpha = (w0_id + 1 - w0/self.dw) / self.dw
                                    axpy(alpha, rho_GG, specfunc_wGG[w0_id % self.NwS_local] )

                                if self.wScomm.rank == (w0_id+1) // self.NwS_local:
                                    alpha =  (w0 / self.dw - w0_id) / self.dw
                                    axpy(alpha, rho_GG, specfunc_wGG[(w0_id+1) % self.NwS_local] )

#                            deltaw = delta_function(w0, self.dw, self.NwS, self.sigma)
#                            for wi in range(self.NwS_local):
#                                if deltaw[wi + self.wS1] > 1e-8:
#                                    specfunc_wGG[wi] += tmp_GG * deltaw[wi + self.wS1]
                if self.nkpt == 1:
                    if n == 0:
                        dt = time() - t0
                        totaltime = dt * self.nband_local
                        self.printtxt('Finished n 0 in %f seconds, estimated %f seconds left.' %(dt, totaltime) )
                    if rank == 0 and self.nband_local // 5 > 0:
                        if n > 0 and n % (self.nband_local // 5) == 0:
                            dt = time() - t0
                            self.printtxt('Finished n %d in %f seconds, estimated %f seconds left.'%(n, dt, totaltime-dt))
            if calc.wfs.world.size != 1:
                self.kcomm.barrier()            
            if k == 0:
                dt = time() - t0
                totaltime = dt * self.nkpt_local
                self.printtxt('Finished k 0 in %f seconds, estimated %f seconds left.' %(dt, totaltime))
                
            if rank == 0 and self.nkpt_local // 5 > 0:            
                if k > 0 and k % (self.nkpt_local // 5) == 0:
                    dt =  time() - t0
                    self.printtxt('Finished k %d in %f seconds, estimated %f seconds left.  '%(k, dt, totaltime - dt) )
        self.printtxt('Finished summation over k')

        self.kcomm.barrier()
        del rho_GG, rho_G
        # Hilbert Transform
        if not self.hilbert_trans:
            self.kcomm.sum(chi0_wGG)
        else:
            self.kcomm.sum(specfunc_wGG)
            if self.wScomm.size == 1:
                chi0_wGG = hilbert_transform(specfunc_wGG, self.Nw, self.dw, self.eta,
                                             self.full_hilbert_trans)[self.wstart:self.wend]
                self.printtxt('Finished hilbert transform !')
                del specfunc_wGG
            else:
                # redistribute specfunc_wGG to all nodes
                assert self.NwS % size == 0
                NwStmp1 = (rank % self.kcomm.size) * self.NwS // size
                NwStmp2 = (rank % self.kcomm.size + 1) * self.NwS // size 
                specfuncnew_wGG = specfunc_wGG[NwStmp1:NwStmp2]
                del specfunc_wGG
                
                coords = np.zeros(self.wcomm.size, dtype=int)
                nG_local = self.npw**2 // self.wcomm.size
                if self.wcomm.rank == self.wcomm.size - 1:
                    nG_local = self.npw**2 - (self.wcomm.size - 1) * nG_local
                self.wcomm.all_gather(np.array([nG_local]), coords)
        
                specfunc_Wg = SliceAlongFrequency(specfuncnew_wGG, coords, self.wcomm)
                self.printtxt('Finished Slice Along Frequency !')
                chi0_Wg = hilbert_transform(specfunc_Wg, self.Nw, self.dw, self.eta,
                                            self.full_hilbert_trans)[:self.Nw]
                self.printtxt('Finished hilbert transform !')
                self.comm.barrier()
                del specfunc_Wg
        
                chi0_wGG = SliceAlongOrbitals(chi0_Wg, coords, self.wcomm)
                self.printtxt('Finished Slice along orbitals !')
                self.comm.barrier()
                del chi0_Wg
        
        self.chi0_wGG = chi0_wGG / self.vol

        self.printtxt('')
        self.printtxt('Finished chi0 !')

        return


    def parallel_init(self):
        """Parallel initialization. By default, only use kcomm and wcomm.

        Parameters:

            kcomm:
                 kpoint communicator
            wScomm:
                 spectral function communicator
            wcomm:
                 frequency communicator
        """

        if extra_parameters.get('df_dry_run'):
            from gpaw.mpi import DryRunCommunicator
            size = extra_parameters['df_dry_run']
            world = DryRunCommunicator(size)
            rank = world.rank
            self.comm = world
        else:
            world = self.comm
            rank = self.comm.rank
            size = self.comm.size

        wcommsize = int(self.NwS * self.npw**2 * 16. / 1024**2) // 1500 # megabyte
        wcommsize += 1
        if size < wcommsize:
            raise ValueError('Number of cpus are not enough ! ')
        if self.kcommsize is None:
            self.kcommsize = world.size
        if wcommsize > size // self.kcommsize: # if matrix too large, overwrite kcommsize and distribute matrix
            self.printtxt('kcommsize is over written ! ')
            while size % wcommsize != 0:
                wcommsize += 1
            self.kcommsize = size // wcommsize
            assert self.kcommsize * wcommsize == size
            if self.kcommsize < 1:
                raise ValueError('Number of cpus are not enough ! ')

        self.kcomm, self.wScomm, self.wcomm = set_communicator(world, rank, size, self.kcommsize)

        if self.nkpt != 1:
            self.nkpt_reshape = self.nkpt
            self.nkpt_reshape, self.nkpt_local, self.kstart, self.kend = parallel_partition(
                               self.nkpt_reshape, self.kcomm.rank, self.kcomm.size, reshape=True, positive=True)
            self.nband_local = self.nbands
            self.nstart = 0
            if self.hilbert_trans:
                self.nend = self.nvalbands
            else:
                self.nend = self.nbands
        else:
            # if number of kpoints == 1, use band parallelization
            self.nkpt_local = 1
            self.kstart = 0
            self.kend = 1

            self.nvalbands, self.nband_local, self.nstart, self.nend = parallel_partition(
                               self.nvalbands, self.kcomm.rank, self.kcomm.size, reshape=False)

        if self.NwS % size != 0:
            self.NwS -= self.NwS % size
            
        self.NwS, self.NwS_local, self.wS1, self.wS2 = parallel_partition(
                               self.NwS, self.wScomm.rank, self.wScomm.size, reshape=False)

        if self.hilbert_trans:
            self.Nw, self.Nw_local, self.wstart, self.wend =  parallel_partition(
                               self.Nw, self.wcomm.rank, self.wcomm.size, reshape=True)
        else:
            if self.Nw > 1:
                assert self.Nw % (size / self.kcomm.size) == 0
                self.wcomm = self.wScomm
                self.Nw, self.Nw_local, self.wstart, self.wend =  parallel_partition(
                               self.Nw, self.wcomm.rank, self.wcomm.size, reshape=False)
            else:
                # if frequency point is too few, then dont parallelize
                self.wcomm = serial_comm
                self.wstart = 0
                self.wend = self.Nw
                self.Nw_local = self.Nw

        return

    def print_chi(self):

        printtxt = self.printtxt
        printtxt('Use Hilbert Transform: %s' %(self.hilbert_trans) )
        printtxt('Calculate full Response Function: %s' %(self.full_hilbert_trans) )
        printtxt('')
        printtxt('Number of frequency points   : %d' %(self.Nw) )
        if self.hilbert_trans:
            printtxt('Number of specfunc points    : %d' % (self.NwS))
        printtxt('')
        printtxt('Parallelization scheme:')
        printtxt('     Total cpus      : %d' %(self.comm.size))
        if self.nkpt == 1:
            printtxt('     nbands parsize  : %d' %(self.kcomm.size))
        else:
            printtxt('     kpoint parsize  : %d' %(self.kcomm.size))
            if self.nkpt_reshape > self.nkpt:
                self.printtxt('        kpoints (%d-%d) are padded with zeros' %(self.nkpt,self.nkpt_reshape))

        if self.hilbert_trans:
            printtxt('     specfunc parsize: %d' %(self.wScomm.size))
        printtxt('     w parsize       : %d' %(self.wcomm.size))
        printtxt('')
        printtxt('Memory usage estimation:')
        printtxt('     chi0_wGG        : %f M / cpu' %(self.Nw_local * self.npw**2 * 16. / 1024**2) )
        if self.hilbert_trans:
            printtxt('     specfunc_wGG    : %f M / cpu' %(self.NwS_local *self.npw**2 * 16. / 1024**2) )
