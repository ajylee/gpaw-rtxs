from time import time, ctime
import numpy as np
import pickle
from math import pi
from ase.units import Hartree
from ase.io import write
from gpaw.io.tar import Writer, Reader
from gpaw.mpi import world, size, rank, serial_comm
from gpaw.utilities.blas import gemmdot, gemm, gemv
from gpaw.utilities import devnull
from gpaw.utilities.memory import maxrss
from gpaw.response.base import BASECHI
from gpaw.response.parallel import parallel_partition
from gpaw.response.df import DF

class BSE(BASECHI):
    """This class defines Belth-Selpether equations."""

    def __init__(self,
                 calc=None,
                 nbands=None,
                 nc=None,
                 nv=None,
                 w=None,
                 q=None,
                 eshift=None,
                 ecut=10.,
                 eta=0.2,
                 rpad=np.array([1,1,1]),
                 vcut=None,
                 ftol=1e-5,
                 txt=None,
                 optical_limit=False,
                 positive_w=False, # True : use Tamm-Dancoff Approx
                 use_W=True, # True: include screened interaction kernel
                 qsymm=True): 

        BASECHI.__init__(self, calc=calc, nbands=nbands, w=w, q=q,
                         eshift=eshift, ecut=ecut, eta=eta, rpad=rpad,
                         ftol=ftol, txt=txt, optical_limit=optical_limit)

        self.epsilon_w = None
        self.positive_w = positive_w
        self.vcut = vcut
        self.nc = nc # conduction band index
        self.nv = nv # valence band index
        self.use_W = use_W
        self.qsymm = qsymm

    def initialize(self):

        self.printtxt('')
        self.printtxt('-----------------------------------------------')
        self.printtxt('Bethe Salpeter Equation calculation started at:')
        self.printtxt(ctime())

        BASECHI.initialize(self)
        
        calc = self.calc
        self.kd = kd = calc.wfs.kd

        # frequency points init
        self.dw = self.w_w[1] - self.w_w[0]
        assert ((self.w_w[1:] - self.w_w[:-1] - self.dw) < 1e-10).all() # make sure its linear w grid
        assert self.w_w.max() == self.w_w[-1]

        self.dw /= Hartree
        self.w_w  /= Hartree
        self.wmax = self.w_w[-1] 
        self.Nw  = int(self.wmax / self.dw) + 1

        # band init
        if self.nc is None and self.positive_w is True: # applied only to semiconductor
            nv = self.nvalence / 2 - 1
            self.nv = np.array([nv, nv+1]) # conduction band start / end
            self.nc = np.array([nv+1, nv+2]) # valence band start / end
            self.printtxt('Number of electrons: %d' %(self.nvalence))
            self.printtxt('Valence band included        : (band %d to band %d)' %(self.nv[0],self.nv[1]-1))
            self.printtxt('Conduction band included     : (band %d to band %d)' %(self.nc[0],self.nc[1]-1))
        elif self.nc == 'all' or self.positive_w is False: # applied to metals
            self.nv = np.array([0, self.nbands])
            self.nc = np.array([0, self.nbands])
            self.printtxt('All the bands are included')
        else:
            self.printtxt('User defined bands for BSE.')
            self.printtxt('Valence band included: (band %d to band %d)' %(self.nv[0],self.nv[1]-1))
            self.printtxt('Conduction band included: (band %d to band %d)' %(self.nc[0],self.nc[1]-1))            

        # find the pair index and initialized pair energy (e_i - e_j) and occupation(f_i-f_j)
        self.e_S = {}
        focc_s = {}
        self.Sindex_S3 = {}
        iS = 0
        kq_k = self.kq_k
        for k1 in range(self.nkpt):
            ibzkpt1 = kd.bz2ibz_k[k1]
            ibzkpt2 = kd.bz2ibz_k[kq_k[k1]]
            for n1 in range(self.nv[0], self.nv[1]): 
                for m1 in range(self.nc[0], self.nc[1]): 
                    focc = self.f_kn[ibzkpt1,n1] - self.f_kn[ibzkpt2,m1]
                    if not self.positive_w: # Dont use Tamm-Dancoff Approx.
                        check_ftol = np.abs(focc) > self.ftol
                    else:
                        check_ftol = focc > self.ftol
                    if check_ftol:           
                        self.e_S[iS] =self.e_kn[ibzkpt2,m1] - self.e_kn[ibzkpt1,n1]
                        focc_s[iS] = focc
                        self.Sindex_S3[iS] = (k1, n1, m1)
                        iS += 1
        self.nS = iS
        self.focc_S = np.zeros(self.nS)
        for iS in range(self.nS):
            self.focc_S[iS] = focc_s[iS]

        if self.use_W:
            # q points init
            self.bzq_qc = kd.get_bz_q_points()
            if not self.qsymm:
                self.ibzq_qc = self.bzq_qc
            else:
                # if use q symmetry, kpoint and qpoint grid should be the same
                (self.ibzq_qc, self.ibzq_q, self.iop_q,
                 self.timerev_q, self.diff_qc) = kd.get_ibz_q_points(self.bzq_qc,
                                                             calc.wfs.symmetry.op_scc)
                if np.abs(self.bzq_qc - self.bzk_kc).sum() < 1e-8:
                    assert np.abs(self.ibzq_qc - kd.ibzk_kc).sum() < 1e-8
            self.nibzq = len(self.ibzq_qc)

        # parallel init
        self.Scomm = world
        # kcomm and wScomm is only to be used when wavefunctions r parallelly distributed.
        self.kcomm = world
        self.wScomm = serial_comm
        
        self.nS, self.nS_local, self.nS_start, self.nS_end = parallel_partition(
                               self.nS, world.rank, world.size, reshape=False)
        self.print_bse()

        if calc.input_parameters['mode'] == 'lcao':
            calc.initialize_positions()        

        # Coulomb kernel init
        self.kc_G = np.zeros(self.npw)
        for iG in range(self.npw):
            index = self.Gindex_G[iG]
            qG = np.dot(self.q_c + self.Gvec_Gc[iG], self.bcell_cv)
            self.kc_G[iG] = 1. / np.inner(qG, qG)
        if self.optical_limit:
            self.kc_G[0] = 0.
            
        self.printtxt('')
        
        return


    def calculate(self):

        calc = self.calc
        f_kn = self.f_kn
        e_kn = self.e_kn
        ibzk_kc = self.ibzk_kc
        bzk_kc = self.bzk_kc
        kq_k = self.kq_k
        focc_S = self.focc_S
        e_S = self.e_S
        op_scc = calc.wfs.symmetry.op_scc

        self.phi_aGp = self.get_phi_aGp()
        if self.use_W:
            bzq_qc=self.bzq_qc
            ibzq_qc = self.ibzq_qc
            if type(self.use_W) is str:
                # read 
                data = pickle.load(open(self.use_W))
                self.dfinvG0_G = data['dfinvG0_G']
                W_qGG = data['W_qGG']
                self.phi_qaGp = data['phi_qaGp']
                self.printtxt('Finished reading screening interaction kernel')
            elif type(self.use_W) is bool:
                # calculate from scratch
                self.printtxt('Calculating screening interaction kernel.')                
                W_qGG = self.screened_interaction_kernel()
            else:
                raise ValueError('use_W can only be string or bool ')

            if not len(self.phi_qaGp) == self.nkpt:                
                import os.path
                if not os.path.isfile('phi_qaGp'):
                    self.printtxt('Calculating phi_qaGp')
                    self.get_phi_qaGp()

                world.barrier()
                self.reader = Reader('phi_qaGp')
                self.printtxt('Finished reading phi_aGp !')
                self.phi_qaGp = None
                self.printtxt('Memory used %f M' %(maxrss() / 1024.**2))
        
       # calculate kernel
        K_SS = np.zeros((self.nS, self.nS), dtype=complex)
        W_SS = np.zeros_like(K_SS)
        self.rhoG0_S = np.zeros((self.nS), dtype=complex)

        t0 = time()
        self.printtxt('Calculating BSE matrix elements.')

        noGmap = 0
        for iS in range(self.nS_start, self.nS_end):
            k1, n1, m1 = self.Sindex_S3[iS]
            rho1_G = self.density_matrix(n1,m1,k1)
            self.rhoG0_S[iS] = rho1_G[0]

            for jS in range(self.nS):
                k2, n2, m2 = self.Sindex_S3[jS]
                rho2_G = self.density_matrix(n2,m2,k2)
                K_SS[iS, jS] = np.sum(rho1_G.conj() * rho2_G * self.kc_G)

                if self.use_W:
                    
                    rho3_G = self.density_matrix(n1,n2,k1,k2)
                    rho4_G = self.density_matrix(m1,m2,self.kq_k[k1],self.kq_k[k2])
                    
                    q_c = bzk_kc[k2] - bzk_kc[k1]
                    q_c[np.where(q_c > 0.501)] -= 1.
                    q_c[np.where(q_c < -0.499)] += 1.

                    if not self.qsymm:
                        ibzq = self.kd.where_is_q(q_c, self.bzq_qc)
                        W_GG = W_qGG[ibzq].copy()
                    else:
                        iq = self.kd.where_is_q(q_c, self.bzq_qc)
                        ibzq = self.ibzq_q[iq]
                        iop = self.iop_q[iq]
                        timerev = self.timerev_q[iq]
                        diff_c = self.diff_qc[iq]
                        invop = np.linalg.inv(op_scc[iop])

                        W_GG_tmp = W_qGG[ibzq]
                        Gindex = np.zeros(self.npw,dtype=int)
    
                        for iG in range(self.npw):
                            G_c = self.Gvec_Gc[iG]
                            if timerev:
                                RotG_c = -np.int8(np.dot(invop, G_c+diff_c).round())
                            else:
                                RotG_c = np.int8(np.dot(invop, G_c+diff_c).round())
                            tmp_G = np.abs(self.Gvec_Gc - RotG_c).sum(axis=1)
                            try:
                                Gindex[iG] = np.where(tmp_G < 1e-5)[0][0]
                            except:
                                noGmap += 1
                                Gindex[iG] = -1
    
                        W_GG = np.zeros_like(W_GG_tmp)
                        for iG in range(self.npw):
                            for jG in range(self.npw):
                                if Gindex[iG] == -1 or Gindex[jG] == -1:
                                    W_GG[iG, jG] = 0
                                else:
                                    W_GG[iG, jG] = W_GG_tmp[Gindex[iG], Gindex[jG]]

                    if k1 == k2:
                        if (n1==n2) or (m1==m2):

                            tmp_G = np.zeros(self.npw)
                            q = np.array([0.0001,0,0])
                            for jG in range(1, self.npw):
                                qG = np.dot(q+self.Gvec_Gc[jG], self.bcell_cv)
                                tmp_G[jG] = self.dfinvG0_G[jG] / np.sqrt(np.inner(qG,qG))

                            const = 1./pi*self.vol*(6*pi**2/self.vol)**(2./3.)
                            tmp_G *= const
                            W_GG[:,0] = tmp_G
                            W_GG[0,:] = tmp_G.conj()
                            W_GG[0,0] = 2./pi*(6*pi**2/self.vol)**(1./3.) \
                                            * self.dfinvG0_G[0] *self.vol

                    tmp_GG = np.outer(rho3_G.conj(), rho4_G) * W_GG
                    W_SS[iS, jS] = np.sum(tmp_GG)
#                    self.printtxt('%d %d %s %s' %(iS, jS, K_SS[iS,jS], W_SS[iS,jS]))
            self.timing(iS, t0, self.nS_local, 'pair orbital') 

        K_SS *= 4 * pi / self.vol
        if self.use_W:
            K_SS -= 0.5 * W_SS / self.vol
        world.sum(K_SS)
        world.sum(self.rhoG0_S)

        self.printtxt('The number of G index outside the Gvec_Gc: %d'%(noGmap))
        # get and solve hamiltonian
        H_SS = np.zeros_like(K_SS)
        for iS in range(self.nS):
            H_SS[iS,iS] = e_S[iS]
            for jS in range(self.nS):
                H_SS[iS,jS] += focc_S[iS] * K_SS[iS,jS]

        if self.positive_w is True: # matrix should be Hermitian
            for iS in range(self.nS):
                for jS in range(self.nS):
                    if np.abs(H_SS[iS,jS]- H_SS[jS,iS].conj()) > 1e-4:
                        print H_SS[iS,jS]- H_SS[jS,iS].conj()
#                    assert np.abs(H_SS[iS,jS]- H_SS[jS,iS].conj()) < 1e-4

#        if not self.positive_w:
        self.w_S, self.v_SS = np.linalg.eig(H_SS)
#        else:
#        from gpaw.utilities.lapack import diagonalize
#        self.w_S = np.zeros(self.nS)
#        diagonalize(H_SS, self.w_S)
#        self.v_SS = H_SS.copy() # eigenvectors in the rows

        data = {
                'w_S': self.w_S,
                'v_SS':self.v_SS,
                'rhoG0_S':self.rhoG0_S
                }
        if rank == 0:
            pickle.dump(data, open('H_SS.pckl', 'w'), -1)

        
        return 


    def screened_interaction_kernel(self):
        """Calcuate W_GG(q)"""

        dfinv_qGG = np.zeros((self.nibzq, self.npw, self.npw),dtype=complex)
        kc_qGG = np.zeros((self.nibzq, self.npw, self.npw))
        dfinvG0_G = np.zeros(self.npw,dtype=complex) # save the wing elements

        t0 = time()
        self.phi_qaGp = {}
        
        for iq in range(self.nibzq):#self.q_start, self.q_end):
            q = self.ibzq_qc[iq]
            optical_limit=False
            if (np.abs(q) < self.ftol).all():
                optical_limit=True
                q = np.array([0.0001, 0, 0])

            df = DF(calc=self.calc, q=q, w=(0.,), nbands=self.nbands,
                    optical_limit=optical_limit,
                    hilbert_trans=False, xc='RPA', rpad=self.rpad, vcut=self.vcut,
                    eta=0.0001, ecut=self.ecut*Hartree, txt='no_output')#, comm=serial_comm)

#            df.e_kn = self.e_kn
            dfinv_qGG[iq] = df.get_inverse_dielectric_matrix(xc='RPA')[0]
            self.phi_qaGp[iq] = df.phi_aGp 
            kc_qGG[iq] = df.Kc_GG

            self.timing(iq, t0, self.nibzq, 'iq')
            assert df.npw == self.npw

            if optical_limit:
                dfinvG0_G = dfinv_qGG[iq,:,0]
                # make sure epsilon_matrix is hermitian.
                assert np.abs(dfinv_qGG[iq,0,:] - dfinv_qGG[iq,:,0].conj()).sum() < 1e-6
                dfinv_qGG[iq,0,0] = np.real(dfinv_qGG[iq,0,0])
            del df

        W_qGG = dfinv_qGG * kc_qGG
#        world.sum(W_qGG)
#        world.broadcast(dfinvG0_G, 0)
        self.dfinvG0_G = dfinvG0_G

        data = {'W_qGG': W_qGG,
                'dfinvG0_G': dfinvG0_G,
                'phi_qaGp':self.phi_qaGp}
        if rank == 0:
            pickle.dump(data, open('W_qGG.pckl', 'w'), -1)

        return W_qGG

                                          
    def print_bse(self):

        printtxt = self.printtxt

        if self.use_W:
            printtxt('Number of q points            : %d' %(self.nibzq))
        printtxt('Number of frequency points   : %d' %(self.Nw) )
        printtxt('Number of pair orbitals      : %d' %(self.nS) )
        printtxt('Parallelization scheme:')
        printtxt('   Total cpus         : %d' %(world.size))
        printtxt('   pair orb parsize   : %d' %(self.Scomm.size))        
        
        return


    def get_phi_qaGp(self):

        phi_aGp = self.phi_aGp.copy()

        N1_max = 0
        N2_max = 0
        natoms = len(phi_aGp)
        for id in range(natoms):
            N1, N2 = phi_aGp[id].shape
            if N1 > N1_max:
                N1_max = N1
            if N2 > N2_max:
                N2_max = N2
        
        del self.phi_qaGp
#        self.phi_qaGp = {}
        nbzq = self.nkpt
        nbzq, nq_local, q_start, q_end = parallel_partition(
                                    nbzq, world.rank, world.size, reshape=False)
        phimax_qaGp = np.zeros((nq_local, natoms, N1_max, N2_max), dtype=complex)
        for iq in range(nq_local):
            self.printtxt('%d' %(iq))
            q_c = self.bzq_qc[iq + q_start]
            tmp_aGp = self.get_phi_aGp(q_c)
            for id in range(natoms):
                N1, N2 = tmp_aGp[id].shape
                phimax_qaGp[iq, id, :N1, :N2] = tmp_aGp[id]
        world.barrier()

        # write to disk
        filename = 'phi_qaGp'
        if world.rank == 0:
            w = Writer(filename)
            w.dimension('nbzq', nbzq)
            w.dimension('natoms', natoms)
            w.dimension('nG', N1_max)
            w.dimension('nii', N2_max)
            w.add('phi_qaGp', ('nbzq', 'natoms', 'nG', 'nii',), dtype=complex)

        for q in range(nbzq):
            if nbzq % size != 0:
                qrank = q // (nbzq // size + 1)
            else:
                qrank = q // (nbzq // size)

            if qrank == 0:
                if world.rank == 0:
                    phi_aGp = phimax_qaGp[q - q_start]
            else:
                if world.rank == qrank:
                    phi_aGp = phimax_qaGp[q - q_start]
                    world.send(phi_aGp, 0, q)
                elif world.rank == 0:
                    world.receive(phi_aGp, qrank, q)
            if world.rank == 0:
                w.fill(phi_aGp)
        world.barrier()
        if world.rank == 0:
            w.close()
        
        return

    def load_phi_aGp(self, reader, iq):

        phimax_aGp = np.array(reader.get('phi_qaGp', iq), complex)

        phi_aGp = {}
        natoms = len(phimax_aGp)
        for a in range(natoms):
            N1, N2 = self.phi_aGp[a].shape
            phi_aGp[a] = phimax_aGp[a, :N1, :N2]

        return phi_aGp


    def get_dielectric_function(self, filename='df.dat', readfile=None, overlap=True):

        if self.epsilon_w is None:
            self.initialize()

            if readfile is None:
                self.calculate()
                self.printtxt('Calculating dielectric function.')
            else:
                data = pickle.load(open(readfile))
                self.w_S  = data['w_S']
                self.v_SS = data['v_SS']
                self.rhoG0_S = data['rhoG0_S']
                self.printtxt('Finished reading H_SS.pckl')

            w_S = self.w_S
            v_SS = self.v_SS # v_SS[:,lamda]
            rhoG0_S = self.rhoG0_S
            focc_S = self.focc_S

            # get overlap matrix
            if not self.positive_w:
                tmp = np.dot(v_SS.conj().T, v_SS )
                overlap_SS = np.linalg.inv(tmp)
    
            # get chi
            epsilon_w = np.zeros(self.Nw, dtype=complex)
            t0 = time()

            A_S = np.dot(rhoG0_S, v_SS)
            B_S = np.dot(rhoG0_S*focc_S, v_SS)
            if not self.positive_w:
                C_S = np.dot(B_S.conj(), overlap_SS.T) * A_S
            else:
                C_S = B_S.conj() * A_S

            for iw in range(self.Nw):
                tmp_S = 1. / (iw*self.dw - w_S + 1j*self.eta)
                epsilon_w[iw] += np.dot(tmp_S, C_S)
    
            epsilon_w *=  - 4 * pi / np.inner(self.qq_v, self.qq_v) / self.vol
            epsilon_w += 1        

            self.epsilon_w = epsilon_w
    
        if rank == 0:
            f = open(filename,'w')
            for iw in range(self.Nw):
                energy = iw * self.dw * Hartree
                print >> f, energy, np.real(epsilon_w[iw]), np.imag(epsilon_w[iw])
            f.close()
        # Wait for I/O to finish
        world.barrier()

        """Check f-sum rule."""
        N1 = 0
        for iw in range(self.Nw):
            w = iw * self.dw
            N1 += np.imag(epsilon_w[iw]) * w
        N1 *= self.dw * self.vol / (2 * pi**2)

        self.printtxt('')
        self.printtxt('Sum rule:')
        nv = self.nvalence
        self.printtxt('N1 = %f, %f  %% error' %(N1, (N1 - nv) / nv * 100) )

        return epsilon_w


    def timing(self, i, t0, n_local, txt):

        if i == 0:
            dt = time() - t0
            self.totaltime = dt * n_local
            self.printtxt('  Finished %s 0 in %f seconds, estimated %f seconds left.' %(txt, dt, self.totaltime))
            
        if rank == 0 and n_local // 5 > 0:            
            if i > 0 and i % (n_local // 5) == 0:
                dt =  time() - t0
                self.printtxt('  Finished %s %d in %f seconds, estimated %f seconds left.  '%(txt, i, dt, self.totaltime - dt) )

        return    

    def get_e_h_density(self, lamda=None, filename=None):

        if filename is not None:
            self.load(filename)
            self.initialize()
            
        gd = self.gd
        w_S = self.w_S
        v_SS = self.v_SS
        A_S = v_SS[:, lamda]
        kq_k = self.kq_k
        kd = self.kd

        # Electron density
        nte_R = gd.zeros()
        
        for iS in range(self.nS_start, self.nS_end):
            print 'electron density:', iS
            k1, n1, m1 = self.Sindex_S3[iS]
            ibzkpt1 = kd.bz2ibz_k[k1]
            psitold_g = self.get_wavefunction(ibzkpt1, n1)
            psit1_g = kd.transform_wave_function(psitold_g, k1)

            for jS in range(self.nS):
                k2, n2, m2 = self.Sindex_S3[jS]
                if m1 == m2 and k1 == k2:
                    psitold_g = self.get_wavefunction(ibzkpt1, n2)
                    psit2_g = kd.transform_wave_function(psitold_g, k1)

                    nte_R += A_S[iS] * A_S[jS].conj() * psit1_g.conj() * psit2_g

        # Hole density
        nth_R = gd.zeros()
        
        for iS in range(self.nS_start, self.nS_end):
            print 'hole density:', iS
            k1, n1, m1 = self.Sindex_S3[iS]
            ibzkpt1 = kd.bz2ibz_k[kq_k[k1]]
            psitold_g = self.get_wavefunction(ibzkpt1, m1)
            psit1_g = kd.transform_wave_function(psitold_g, kq_k[k1])

            for jS in range(self.nS):
                k2, n2, m2 = self.Sindex_S3[jS]
                if n1 == n2 and k1 == k2:
                    psitold_g = self.get_wavefunction(ibzkpt1, m2)
                    psit2_g = kd.transform_wave_function(psitold_g, kq_k[k1])

                    nth_R += A_S[iS] * A_S[jS].conj() * psit1_g * psit2_g.conj()
                    
        self.Scomm.sum(nte_R)
        self.Scomm.sum(nth_R)


        if rank == 0:
            write('rho_e.cube',self.calc.atoms, format='cube', data=nte_R)
            write('rho_h.cube',self.calc.atoms, format='cube', data=nth_R)
            
        world.barrier()
        
        return 

    def get_excitation_wavefunction(self, lamda=None,filename=None, re_c=None, rh_c=None):
        """ garbage at the moment. come back later"""
        if filename is not None:
            self.load(filename)
            self.initialize()
            
        gd = self.gd
        w_S = self.w_S
        v_SS = self.v_SS
        A_S = v_SS[:, lamda]
        kq_k = self.kq_k
        kd = self.kd

        nx, ny, nz = self.nG[0], self.nG[1], self.nG[2]
        nR = 9
        nR2 = (nR - 1 ) // 2
        if re_c is not None:
            psith_R = gd.zeros(dtype=complex)
            psith2_R = np.zeros((nR*nx, nR*ny, nz), dtype=complex)
            
        elif rh_c is not None:
            psite_R = gd.zeros(dtype=complex)
            psite2_R = np.zeros((nR*nx, ny, nR*nz), dtype=complex)
        else:
            self.printtxt('No wavefunction output !')
            return
            
        for iS in range(self.nS_start, self.nS_end):

            k, n, m = self.Sindex_S3[iS]
            ibzkpt1 = kd.bz2ibz_k[k]
            ibzkpt2 = kd.bz2ibz_k[kq_k[k]]
            print 'hole wavefunction', iS, (k,n,m),A_S[iS]
            
            psitold_g = self.get_wavefunction(ibzkpt1, n)
            psit1_g = kd.transform_wave_function(psitold_g, k)

            psitold_g = self.get_wavefunction(ibzkpt2, m)
            psit2_g = kd.transform_wave_function(psitold_g, kq_k[k])

            if re_c is not None:
                # given electron position, plot hole wavefunction
                tmp = A_S[iS] * psit1_g[re_c].conj() * psit2_g
                psith_R += tmp

                k_c = self.bzk_kc[k] + self.q_c
                for i in range(nR):
                    for j in range(nR):
                        R_c = np.array([i-nR2, j-nR2, 0])
                        psith2_R[i*nx:(i+1)*nx, j*ny:(j+1)*ny, 0:nz] += \
                                                tmp * np.exp(1j*2*pi*np.dot(k_c,R_c))
                
            elif rh_c is not None:
                # given hole position, plot electron wavefunction
                tmp = A_S[iS] * psit1_g.conj() * psit2_g[rh_c] * self.expqr_g
                psite_R += tmp

                k_c = self.bzk_kc[k]
                k_v = np.dot(k_c, self.bcell_cv)
                for i in range(nR):
                    for j in range(nR):
                        R_c = np.array([i-nR2, 0, j-nR2])
                        R_v = np.dot(R_c, self.acell_cv)
                        assert np.abs(np.dot(k_v, R_v) - np.dot(k_c, R_c) * 2*pi).sum() < 1e-5
                        psite2_R[i*nx:(i+1)*nx, 0:ny, j*nz:(j+1)*nz] += \
                                                tmp * np.exp(-1j*np.dot(k_v,R_v))
                
            else:
                pass

        if re_c is not None:
            self.Scomm.sum(psith_R)
            self.Scomm.sum(psith2_R)
            if rank == 0:
                write('psit_h.cube',self.calc.atoms, format='cube', data=psith_R)

                atoms = self.calc.atoms
                shift = atoms.cell[0:2].copy()
                positions = atoms.positions
                atoms.cell[0:2] *= nR2
                atoms.positions += shift * (nR2 - 1)
                
                write('psit_bigcell_h.cube',atoms, format='cube', data=psith2_R)
        elif rh_c is not None:
            self.Scomm.sum(psite_R)
            self.Scomm.sum(psite2_R)
            if rank == 0:
                write('psit_e.cube',self.calc.atoms, format='cube', data=psite_R)

                atoms = self.calc.atoms
#                shift = atoms.cell[0:2].copy()
                positions = atoms.positions
                atoms.cell[0:2] *= nR2
#                atoms.positions += shift * (nR2 - 1)
                
                write('psit_bigcell_e.cube',atoms, format='cube', data=psite2_R)
                
        else:
            pass

        world.barrier()
            
        return
    

    def load(self, filename):

        data = pickle.load(open(filename))
        self.w_S  = data['w_S']
        self.v_SS = data['v_SS']

        self.printtxt('Read succesfully !')
        

    def save(self, filename):
        """Dump essential data"""

        data = {'w_S'  : self.w_S,
                'v_SS' : self.v_SS}
        
        if rank == 0:
            pickle.dump(data, open(filename, 'w'), -1)

        world.barrier()

