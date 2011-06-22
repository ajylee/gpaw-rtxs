# Copyright (C) 2010  CAMd
# Please see the accompanying LICENSE file for further information.

"""This module provides all the classes and functions associated with the
evaluation of exact exchange with k-point sampling."""

from math import pi, sqrt

import numpy as np
from ase import Atoms

from gpaw.xc import XC
from gpaw.xc.kernel import XCNull
from gpaw.xc.functional import XCFunctional
from gpaw.utilities import hartree, pack, unpack2, packed_index
from gpaw.lfc import LFC
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.kpoint import KPoint as KPoint0
from gpaw.mpi import world


class KPoint:
    def __init__(self, kd, kpt=None):
        """Helper class for parallelizing over k-points.

        Placeholder for wave functions, occupation numbers,
        projections, and global k-point index."""
        
        self.kd = kd
        
        if kpt is not None:
            self.psit_nG = kpt.psit_nG
            self.f_n = kpt.f_n / kpt.weight / kd.nbzkpts * 2 / kd.nspins

            self.P_ani = kpt.P_ani
            self.k = kpt.k
            self.s = kpt.s
            
        self.requests = []
        
    def next(self):
        """Create empty object.

        Data will be received from other processor."""
        
        kpt = KPoint(self.kd)

        # intialize array for receiving:
        kpt.psit_nG = np.empty_like(self.psit_nG)
        kpt.f_n = np.empty_like(self.f_n)

        # Total number of projector functions:
        I = sum([P_ni.shape[1] for P_ni in self.P_ani.values()])
        
        kpt.P_In = np.empty((I, len(kpt.f_n)), complex)

        kpt.P_ani = {}
        I1 = 0
        for a, P_ni in self.P_ani.items():
            I2 = I1 + P_ni.shape[1]
            kpt.P_ani[a] = kpt.P_In[I1:I2].T
            I1 = I2

        kpt.k = (self.k + 1) % self.kd.nibzkpts
        kpt.s = self.s
        
        return kpt
        
    def start_sending(self, rank):
        P_In = np.concatenate([P_ni.T for P_ni in self.P_ani.values()])
        self.requests += [
            self.kd.comm.send(self.psit_nG, rank, block=False, tag=1),
            self.kd.comm.send(self.f_n, rank, block=False, tag=2),
            self.kd.comm.send(P_In, rank, block=False, tag=3)]
        
    def start_receiving(self, rank):
        self.requests += [
            self.kd.comm.receive(self.psit_nG, rank, block=False, tag=1),
            self.kd.comm.receive(self.f_n, rank, block=False, tag=2),
            self.kd.comm.receive(self.P_In, rank, block=False, tag=3)]
        
    def wait(self):
        self.kd.comm.waitall(self.requests)
        self.requests = []
        

class HybridXC(XCFunctional):
    orbital_dependent = True
    def __init__(self, name, hybrid=None, xc=None, finegrid=False,
                 alpha=None, skip_gamma=False):
        """Mix standard functionals with exact exchange.

        name: str
            Name of hybrid functional.
        hybrid: float
            Fraction of exact exchange.
        xc: str or XCFunctional object
            Standard DFT functional with scaled down exchange.
        finegrid: boolean
            Use fine grid for energy functional evaluations?
        """

        if name == 'EXX':
            assert hybrid is None and xc is None
            hybrid = 1.0
            xc = XC(XCNull())
        elif name == 'PBE0':
            assert hybrid is None and xc is None
            hybrid = 0.25
            xc = XC('HYB_GGA_XC_PBEH')
        elif name == 'B3LYP':
            assert hybrid is None and xc is None
            hybrid = 0.2
            xc = XC('HYB_GGA_XC_B3LYP')
            
        if isinstance(xc, str):
            xc = XC(xc)

        self.hybrid = hybrid
        self.xc = xc
        self.type = xc.type
        self.alpha = alpha
        self.skip_gamma = skip_gamma
        self.exx = None
        
        XCFunctional.__init__(self, name)

    def get_setup_name(self):
        return 'PBE'

    def calculate_radial(self, rgd, n_sLg, Y_L, v_sg,
                         dndr_sLg=None, rnablaY_Lv=None,
                         tau_sg=None, dedtau_sg=None):
        return self.xc.calculate_radial(rgd, n_sLg, Y_L, v_sg,
                                        dndr_sLg, rnablaY_Lv)
    
    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None,
                                 addcoredensity=True, a=None):
        return self.xc.calculate_paw_correction(setup, D_sp, dEdD_sp,
                                 addcoredensity, a)
    
    def initialize(self, density, hamiltonian, wfs, occupations):
        self.xc.initialize(density, hamiltonian, wfs, occupations)
        self.nspins = wfs.nspins
        self.setups = wfs.setups
        self.density = density
        self.kpt_u = wfs.kpt_u
        
        self.gd = density.gd
        self.kd = wfs.kd
        self.bd = wfs.bd

        N_c = self.gd.N_c
        N = self.gd.N_c.prod()
        vol = self.gd.dv * N
        
        if self.alpha is None:
            # XXX ?
            self.alpha = 6 * vol**(2 / 3.0) / pi**2
            
        self.gamma = (vol / (2 * pi)**2 * sqrt(pi / self.alpha) *
                      self.kd.nbzkpts)
        ecut = 0.5 * pi**2 / (self.gd.h_cv**2).sum(1).max()
        print('alpha=%f' % self.alpha)
        print('ecut=%f Hartree' % ecut)

        if self.kd.N_c is None:
            self.bzk_kc = np.zeros((1, 3))
            dfghdfgh
        else:
            n = self.kd.N_c * 2 - 1
            bzk_kc = np.indices(n).transpose((1, 2, 3, 0))
            bzk_kc.shape = (-1, 3)
            bzk_kc -= self.kd.N_c - 1
            self.bzk_kc = bzk_kc.astype(float) / self.kd.N_c
        
        self.pwd = PWDescriptor(ecut, self.gd, self.bzk_kc)

        n = 0
        for k_c, Gpk2_G in zip(self.bzk_kc[:], self.pwd.G2_qG):
            if (k_c > -0.5).all() and (k_c <= 0.5).all(): #XXX???
                if k_c.any():
                    self.gamma -= np.dot(np.exp(-self.alpha * Gpk2_G),
                                         Gpk2_G**-1)
                else:
                    self.gamma -= np.dot(np.exp(-self.alpha * Gpk2_G[1:]),
                                         Gpk2_G[1:]**-1)
                n += 1

        assert n == self.kd.N_c.prod()
        
        self.ghat = LFC(self.gd,
                        [setup.ghat_l for setup in density.setups],
                        dtype=complex)

        self.ghat.set_k_points(self.bzk_kc)
        
        self.interpolator = density.interpolator

    def set_positions(self, spos_ac):
        self.ghat.set_positions(spos_ac)
        self.spos_ac = spos_ac

    def calculate(self, gd, n_sg, v_sg=None, e_g=None):
        # Normal XC contribution:
        exc = self.xc.calculate(gd, n_sg, v_sg, e_g)

        # Add EXX contribution:
        return exc + self.exx

    def calculate_exx(self):
        """Non-selfconsistent calculation."""

        kd = self.kd
        K = kd.nibzkpts
        W = world.size // self.nspins
        parallel = (W > 1)
        
        self.exx_skn = np.zeros((self.nspins, K, self.bd.nbands))
        self.debug_skn = np.zeros((self.nspins, K, self.bd.nbands))

        #self.x_kK = np.zeros((K, kd.nbzkpts))
        #for k1 in range(K):
        #    for K in range(kd.nbzkpts):
                
        for s in range(self.nspins):
            kpt1_q = [KPoint(kd, kpt)
                      for kpt in self.kpt_u if kpt.s == s]
            kpt2_q = kpt1_q[:]

            if len(kpt1_q) == 0:
                # No s-spins on this CPU:
                continue

            # Send rank:
            srank = kd.get_rank_and_index(s, (kpt1_q[0].k - 1) % K)[0]
            # Receive rank:
            rrank = kd.get_rank_and_index(s, (kpt1_q[-1].k + 1) % K)[0]

            # Shift k-points K - 1 times:
            for i in range(K):
                if i < K - 1:
                    if parallel:
                        kpt = kpt2_q[-1].next()
                        kpt.start_receiving(rrank)
                        kpt2_q[0].start_sending(srank)
                    else:
                        kpt = kpt2_q[0]

                for kpt1, kpt2 in zip(kpt1_q, kpt2_q):
                    for k, ik in enumerate(kd.bz2ibz_k):
                        if ik == kpt2.k:
                            self.apply(kpt1, kpt2, k)

                if i < K - 1:
                    if parallel:
                        kpt.wait()
                        kpt2_q[0].wait()
                    kpt2_q.pop(0)
                    kpt2_q.append(kpt)
            
        self.exx = 0.0
        world.sum(self.exx_skn)
        for kpt in self.kpt_u:
            self.exx += 0.5 * np.dot(kpt.f_n, self.exx_skn[kpt.s, kpt.k])
        self.exx = world.sum(self.exx)
        world.sum(self.debug_skn)
        assert (self.debug_skn == self.kd.nbzkpts * self.bd.nbands).all()
        self.exx += self.calculate_exx_paw_correction()
        
    def apply(self, kpt1, kpt2, k):
        k1_c = self.kd.ibzk_kc[kpt1.k]
        k2_c = self.kd.bzk_kc[k]
        k12_c = k1_c - k2_c
        N_c = self.gd.N_c
        eikr_R = np.exp(2j * pi * np.dot(np.indices(N_c).T, k12_c / N_c).T)

        for q, k_c in enumerate(self.bzk_kc):
            if abs(k_c + k12_c).max() < 1e-9:
                q0 = q
                break

        Gpk2_G = self.pwd.G2_qG[q0]
        if Gpk2_G[0] == 0:
            Gpk2_G = Gpk2_G.copy()
            Gpk2_G[0] = 1.0 / self.gamma

        N = N_c.prod()
        vol = self.gd.dv * N
        nspins = self.nspins

        same = abs(k1_c - k2_c).max() < 1e-9
        fcut = 1e-10
        is_ibz2 = abs(k2_c - self.kd.ibzk_kc[kpt2.k]).max() < 1e-9
        
        for n1, psit1_R in enumerate(kpt1.psit_nG):
            f1 = kpt1.f_n[n1]
            for n2, psit2_R in enumerate(kpt2.psit_nG):
                if same:
                    assert is_ibz2
                    if n2 > n1:
                        continue
                elif is_ibz2:
                    if kpt1.k > kpt2.k:
                        if n2 > n1:
                            continue
                    else:
                        if n2 >= n1:
                            continue
                        
                f2 = kpt2.f_n[n2]

                x = 1.0
                if same and n1 == n2:
                    x = 0.5
                    
                self.debug_skn[kpt1.s, kpt1.k, n1] += x
                if is_ibz2:
                    self.debug_skn[kpt2.s, kpt2.k, n2] += x

                if abs(f1) < fcut and abs(f2) < fcut:
                    continue

                if self.skip_gamma and same:
                    continue
                
                nt_R = self.calculate_pair_density(n1, n2, kpt1, kpt2, q0, k)
                nt_G = self.pwd.fft(nt_R * eikr_R) / N
                vt_G = nt_G.copy()
                vt_G *= -pi * vol / Gpk2_G
                e = np.vdot(nt_G, vt_G).real * nspins * self.hybrid * x

                self.exx_skn[kpt1.s, kpt1.k, n1] += 2 * f2 * e
                if is_ibz2:
                    self.exx_skn[kpt2.s, kpt2.k, n2] += 2 * f1 * e

    def calculate_exx_paw_correction(self):
        exx = 0
        deg = 2 // self.nspins  # spin degeneracy
        for a, D_sp in self.density.D_asp.items():
            setup = self.setups[a]
            for D_p in D_sp:
                D_ii = unpack2(D_p)
                ni = len(D_ii)

                for i1 in range(ni):
                    for i2 in range(ni):
                        A = 0.0
                        for i3 in range(ni):
                            p13 = packed_index(i1, i3, ni)
                            for i4 in range(ni):
                                p24 = packed_index(i2, i4, ni)
                                A += setup.M_pp[p13, p24] * D_ii[i3, i4]
                        p12 = packed_index(i1, i2, ni)
                        exx -= self.hybrid / deg * D_ii[i1, i2] * A

                if setup.X_p is not None:
                    exx -= self.hybrid * np.dot(D_p, setup.X_p)
            exx += self.hybrid * setup.ExxC
        return exx
    
    def calculate_pair_density(self, n1, n2, kpt1, kpt2, q, k):
        psit2_G = self.kd.transform_wave_function(kpt2.psit_nG[n2], k)
        nt_G = kpt1.psit_nG[n1].conj() * psit2_G

        s = self.kd.sym_k[k]
        time_reversal = self.kd.time_reversal_k[k]
        k2_c = self.kd.ibzk_kc[kpt2.k]

        Q_aL = {}
        for a, P1_ni in kpt1.P_ani.items():
            P1_i = P1_ni[n1]

            b = self.kd.symmetry.a_sa[s, a]
            S_c = (np.dot(self.spos_ac[a], self.kd.symmetry.op_scc[s]) -
                   self.spos_ac[b])
            assert abs(S_c.round() - S_c).max() < 1e-13
            x = np.exp(2j * pi * np.dot(k2_c, S_c))
            P2_i = np.dot(self.setups[a].R_sii[s], kpt2.P_ani[b][n2]) * x
            if time_reversal:
                P2_i = P2_i.conj()

            D_ii = np.outer(P1_i.conj(), P2_i)
            D_p = pack(D_ii)
            Q_aL[a] = np.dot(D_p, self.setups[a].Delta_pL)

        self.ghat.add(nt_G, Q_aL, q)
        return nt_G
