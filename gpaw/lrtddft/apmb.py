"""Omega matrix for functionals with Hartree-Fock exchange.

"""
from math import sqrt

import numpy as np
from numpy.linalg import inv

from ase.units import Hartree

import gpaw.mpi as mpi
from gpaw.lrtddft.omega_matrix import OmegaMatrix
from gpaw.pair_density import PairDensity
from gpaw.utilities import pack
from gpaw.utilities.lapack import diagonalize, gemm, sqrt_matrix
from gpaw.utilities.timing import Timer

class ApmB(OmegaMatrix):
    """Omega matrix for functionals with Hartree-Fock exchange.
 
    """

    def get_full(self):

        self.paw.timer.start('ApmB RPA')
        self.ApB = self.Om
        self.AmB = self.get_rpa()
        self.paw.timer.stop()

        if self.xc is not None:
            self.paw.timer.start('ApmB XC')
            self.get_xc() # inherited from OmegaMatrix
            self.paw.timer.stop()
        
    def get_rpa(self):
        """Calculate RPA and Hartree-fock part of the A+-B matrices."""

        # shorthands
        kss = self.fullkss
        finegrid = self.finegrid

        # calculate omega matrix
        nij = len(kss)
        print >> self.txt, 'RPAhyb', nij, 'transitions'
        
        AmB = np.zeros((nij, nij))
        ApB = self.ApB

        # storage place for Coulomb integrals
        integrals = {}
        
        for ij in range(nij):
            print >> self.txt,'RPAhyb kss['+'%d'%ij+']=', kss[ij]

            timer = Timer()
            timer.start('init')
            timer2 = Timer()
                      
            # smooth density including compensation charges
            timer2.start('with_compensation_charges 0')
            rhot_p = kss[ij].with_compensation_charges(
                finegrid is not 0)
            timer2.stop()
            
            # integrate with 1/|r_1-r_2|
            timer2.start('poisson')
            phit_p = np.zeros(rhot_p.shape, rhot_p.dtype)
            self.poisson.solve(phit_p,rhot_p, charge=None)
            timer2.stop()

            timer.stop()
            t0 = timer.get_time('init')
            timer.start(ij)

            if finegrid == 1:
                rhot = kss[ij].with_compensation_charges()
                phit = self.gd.zeros()
                self.restrict(phit_p, phit)
            else:
                phit = phit_p
                rhot = rhot_p

            for kq in range(ij, nij):
                if kq != ij:
                    # smooth density including compensation charges
                    timer2.start('kq with_compensation_charges')
                    rhot = kss[kq].with_compensation_charges(
                        finegrid is 2)
                    timer2.stop()
                pre = self.weight_Kijkq(ij, kq)

                timer2.start('integrate')
                I = self.Coulomb_integral_kss(kss[ij], kss[kq], phit, rhot)
                if kss[ij].spin == kss[kq].spin:
                    name = self.Coulomb_integral_name(kss[ij].i, kss[ij].j,
                                                      kss[kq].i, kss[kq].j,
                                                      kss[ij].spin         )
                    integrals[name] = I
                ApB[ij,kq]= pre * I
                timer2.stop()
                
                if ij == kq:
                    epsij =  kss[ij].get_energy() / kss[ij].get_weight()
                    AmB[ij,kq] += epsij
                    ApB[ij,kq] += epsij

            timer.stop()
##            timer2.write()
            if ij < (nij - 1):
                t = timer.get_time(ij) # time for nij-ij calculations
                t = .5*t*(nij-ij)  # estimated time for n*(n+1)/2, n=nij-(ij+1)
                print >> self.txt,'RPAhyb estimated time left',\
                      self.timestring(t0*(nij-ij-1)+t)

        # add HF parts and apply symmetry
        timer.start('RPA hyb HF part')
        if hasattr(self.xc, 'hybrid'):
            weight = self.xc.hybrid
        else:
            weight = 0.0
        for ij in range(nij):
            i = kss[ij].i
            j = kss[ij].j
            s = kss[ij].spin
            for kq in range(ij,nij):
                if kss[ij].pspin == kss[kq].pspin:
                    k = kss[kq].i
                    q = kss[kq].j
                    ikjq = self.Coulomb_integral_ijkq(i, k, j, q, s, integrals)
                    iqkj = self.Coulomb_integral_ijkq(i, q, k, j, s, integrals)
                    ApB[ij,kq] -= weight * ( ikjq + iqkj )
                    AmB[ij,kq] -= weight * ( ikjq - iqkj )
                
                ApB[kq,ij] = ApB[ij,kq]
                AmB[kq,ij] = AmB[ij,kq]
        timer.stop()
        
        return AmB
    
    def Coulomb_integral_name(self, i, j, k, l, spin):
        """return a unique name considering the Coulomb integral
        symmetry"""
        def ij_name(i, j):
            return str(max(i, j)) + ' ' + str(min(i, j))
        
        # maximal gives the first
        if max(i, j) >= max(k, l):
            base = ij_name(i, j) + ' ' + ij_name(k, l) 
        else:
            base = ij_name(k, l) + ' ' + ij_name(i, j)
        return base + ' ' + str(spin)

    def Coulomb_integral_ijkq(self, i, j, k, q, spin, integrals):
        name = self.Coulomb_integral_name(i, j, k, q, spin)
        if name in integrals:
            return integrals[name]
        # create the Kohn-Sham singles
        kss_ij = PairDensity(self.paw)
        kss_ij.initialize(self.paw.wfs.kpt_u[spin], i, j)
        kss_kq = PairDensity(self.paw)
        kss_kq.initialize(self.paw.wfs.kpt_u[spin], k, q)
##         kss_ij = KSSingle(i, j, spin, spin, self.paw)
##         kss_kq = KSSingle(k, q, spin, spin, self.paw)

        rhot_p = kss_ij.with_compensation_charges(
            self.finegrid is not 0)
        phit_p = np.zeros(rhot_p.shape, rhot_p.dtype)
        self.poisson.solve(phit_p, rhot_p, charge=None)

        if self.finegrid == 1:
            phit = self.gd.zeros()
            self.restrict(phit_p, phit)
        else:
            phit = phit_p
            
        rhot = kss_kq.with_compensation_charges(
            self.finegrid is 2)

        integrals[name] = self.Coulomb_integral_kss(kss_ij, kss_kq,
                                                    phit, rhot)
        return integrals[name]
    
    def Coulomb_integral_kss(self, kss_ij, kss_kq, phit, rhot):
        # smooth part
        I = self.gd.integrate(rhot * phit)
        
        wfs = self.paw.wfs
        Pij_ani = wfs.kpt_u[kss_ij.spin].P_ani
        Pkq_ani = wfs.kpt_u[kss_kq.spin].P_ani

        # Add atomic corrections
        Ia = 0.0
        for a, Pij_ni in Pij_ani.items():
            Pi_i = Pij_ni[kss_ij.i]
            Pj_i = Pij_ni[kss_ij.j]
            Dij_ii = np.outer(Pi_i, Pj_i)
            Dij_p = pack(Dij_ii)
            Pk_i = Pkq_ani[a][kss_kq.i]
            Pq_i = Pkq_ani[a][kss_kq.j]
            Dkq_ii = np.outer(Pk_i, Pq_i)
            Dkq_p = pack(Dkq_ii)
            C_pp = wfs.setups[a].M_pp
            #   ----
            # 2 >      P   P  C    P  P
            #   ----    ip  jr prst ks qt
            #   prst
            Ia += 2.0*np.dot(Dkq_p, np.dot(C_pp, Dij_p))
        I += self.gd.comm.sum(Ia)

        return I

    def timestring(self,t):
        ti = int(t+.5)
        td = int(ti//86400)
        st=''
        if td>0:
            st+='%d'%td+'d'
            ti-=td*86400
        th = int(ti//3600)
        if th>0:
            st+='%d'%th+'h'
            ti-=th*3600
        tm = int(ti//60)
        if tm>0:
            st+='%d'%tm+'m'
            ti-=tm*60
        st+='%d'%ti+'s'
        return st

    def diagonalize(self, istart=None, jend=None, energy_range=None):
        """Evaluate Eigenvectors and Eigenvalues:"""

        map, kss = self.get_map(istart, jend, energy_range)
        nij = len(kss)
        if map is None:
            ApB = self.ApB.copy()
            AmB = self.AmB.copy()
            nij = len(kss)
        else:
            ApB = np.empty((nij, nij))
            AmB = np.empty((nij, nij))
            for ij in range(nij):
                for kq in range(nij):
                    ApB[ij,kq] = self.ApB[map[ij],map[kq]]
                    AmB[ij,kq] = self.AmB[map[ij],map[kq]]

        # the occupation matrix
        C = np.empty((nij,))
        for ij in range(nij):
            C[ij] = 1. / kss[ij].fij

        S = C * inv(AmB) * C
        S = sqrt_matrix(inv(S).copy())

        # get Omega matrix
        M = np.zeros(ApB.shape)
        gemm(1.0, ApB, S, 0.0, M)
        self.eigenvectors = np.zeros(ApB.shape)
        gemm(1.0, S, M, 0.0, self.eigenvectors)
        
        self.eigenvalues = np.zeros((len(kss)))
        self.kss = kss
        diagonalize(self.eigenvectors, self.eigenvalues)
        
    def read(self, filename=None, fh=None):
        """Read myself from a file"""
        if mpi.rank == mpi.MASTER:
            if fh is None:
                f = open(filename, 'r')
            else:
                f = fh

            f.readline()
            nij = int(f.readline())
            ApB = np.zeros((nij, nij))
            for ij in range(nij):
                l = f.readline().split()
                for kq in range(ij, nij):
                    ApB[ij, kq] = float(l[kq - ij])
                    ApB[kq, ij] = ApB[ij, kq]
            self.ApB = ApB

            f.readline()
            nij = int(f.readline())
            AmB = np.zeros((nij, nij))
            for ij in range(nij):
                l = f.readline().split()
                for kq in range(ij, nij):
                    AmB[ij, kq] = float(l[kq - ij])
                    AmB[kq, ij] = AmB[ij, kq]
            self.AmB = AmB

            if fh is None:
                f.close()

    def weight_Kijkq(self, ij, kq):
        """weight for the coupling matrix terms"""
        return 2.
    
    def write(self, filename=None, fh=None):
        """Write current state to a file."""
        if mpi.rank == mpi.MASTER:
            if fh is None:
                f = open(filename, 'w')
            else:
                f = fh

            f.write('# A+B\n')
            nij = len(self.fullkss)
            f.write('%d\n' % nij)
            for ij in range(nij):
                for kq in range(ij, nij):
                    f.write(' %g' % self.ApB[ij, kq])
                f.write('\n')
            
            f.write('# A-B\n')
            nij = len(self.fullkss)
            f.write('%d\n' % nij)
            for ij in range(nij):
                for kq in range(ij, nij):
                    f.write(' %g' % self.AmB[ij, kq])
                f.write('\n')
            
            if fh is None:
                f.close()

    def __str__(self):
        string = '<ApmB> '
        if hasattr(self,'eigenvalues'):
            string += 'dimension '+ ('%d'%len(self.eigenvalues))
            string += "\neigenvalues: "
            for ev in self.eigenvalues:
                string += ' ' + ('%f'%(sqrt(ev) * Hartree))
        return string
    


