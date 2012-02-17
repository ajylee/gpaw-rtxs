import numpy as np

from gpaw.lfc import BasisFunctions
from gpaw.utilities import unpack
from gpaw.utilities.tools import tri2full
from gpaw import debug
from gpaw.lcao.overlap import NewTwoCenterIntegrals as NewTCI
from gpaw.utilities.blas import gemm, gemmdot
from gpaw.wavefunctions.base import WaveFunctions


def get_r_and_offsets(nl, spos_ac, cell_cv):
    r_and_offset_aao = {}

    def add(a1, a2, R_c, offset):
        if not (a1, a2) in r_and_offset_aao:
            r_and_offset_aao[(a1, a2)] = []
        r_and_offset_aao[(a1, a2)].append((R_c, offset))
    
    for a1, spos1_c in enumerate(spos_ac):
        a2_a, offsets = nl.get_neighbors(a1)
        for a2, offset in zip(a2_a, offsets):
            spos2_c = spos_ac[a2] + offset

            R_c = np.dot(spos2_c - spos1_c, cell_cv)
            add(a1, a2, R_c, offset)
            if a1 != a2 or offset.any():
                add(a2, a1, -R_c, -offset)
    
    return r_and_offset_aao


def add_paw_correction_to_overlap(setups, P_aqMi, S_qMM, Mstart=0,
                                  Mstop=None):
    if Mstop is None:
        Mstop = setups.nao
    for a, P_qMi in P_aqMi.items():
        dO_ii = np.asarray(setups[a].dO_ii, S_qMM.dtype)
        for S_MM, P_Mi in zip(S_qMM, P_qMi):
            dOP_iM = np.zeros((dO_ii.shape[1], setups.nao),
                              P_Mi.dtype)
            # (ATLAS can't handle uninitialized output array)
            gemm(1.0, P_Mi, dO_ii, 0.0, dOP_iM, 'c')
            gemm(1.0, dOP_iM, P_Mi[Mstart:Mstop],
                 1.0, S_MM, 'n')



class LCAOWaveFunctions(WaveFunctions):
    def __init__(self, ksl, gd, nvalence, setups, bd,
                 dtype, world, kd, timer=None):
        WaveFunctions.__init__(self, gd, nvalence, setups, bd,
                               dtype, world, kd, timer)
        self.ksl = ksl
        self.S_qMM = None
        self.T_qMM = None
        self.P_aqMi = None
        
        self.timer.start('TCI: Evaluate splines')
        self.tci = NewTCI(gd.cell_cv, gd.pbc_c, setups, kd.ibzk_qc, kd.gamma)
        self.timer.stop('TCI: Evaluate splines')
        
        self.basis_functions = BasisFunctions(gd,
                                              [setup.phit_j
                                               for setup in setups],
                                              kd,
                                              cut=True)

    def empty(self, n=(), dtype=float, global_array=False, realspace=False):
        if realspace:
            return self.gd.empty(n, dtype, global_array)
        else:
            if isinstance(n, int):
                n = (n,)
            nao = self.setups.nao
            return np.empty(n + (nao,), dtype)

    def summary(self, fd):
        fd.write('Wave functions: LCAO\n')
        
    def set_eigensolver(self, eigensolver):
        WaveFunctions.set_eigensolver(self, eigensolver)
        eigensolver.initialize(self.gd, self.dtype, self.setups.nao, self.ksl)

    def set_positions(self, spos_ac):
        self.timer.start('Basic WFS set positions')
        WaveFunctions.set_positions(self, spos_ac)
        self.timer.stop('Basic WFS set positions')
        self.timer.start('Basis functions set positions')
        self.basis_functions.set_positions(spos_ac)
        self.timer.stop('Basis functions set positions')
        if self.ksl is not None:
            self.basis_functions.set_matrix_distribution(self.ksl.Mstart,
                                                         self.ksl.Mstop)

        nq = len(self.kd.ibzk_qc)
        nao = self.setups.nao
        mynbands = self.bd.mynbands
        
        Mstop = self.ksl.Mstop
        Mstart = self.ksl.Mstart
        mynao = Mstop - Mstart

        if self.ksl.using_blacs: # XXX
            # S and T have been distributed to a layout with blacs, so
            # discard them to force reallocation from scratch.
            #
            # TODO: evaluate S and T when they *are* distributed, thus saving
            # memory and avoiding this problem
            self.S_qMM = None
            self.T_qMM = None
        
        S_qMM = self.S_qMM
        T_qMM = self.T_qMM
        
        if S_qMM is None: # XXX
            # First time:
            assert T_qMM is None
            if self.ksl.using_blacs: # XXX
                self.tci.set_matrix_distribution(Mstart, mynao)
                
            S_qMM = np.empty((nq, mynao, nao), self.dtype)
            T_qMM = np.empty((nq, mynao, nao), self.dtype)
        
        for kpt in self.kpt_u:
            if kpt.C_nM is None:
                kpt.C_nM = np.empty((mynbands, nao), self.dtype)

        self.allocate_arrays_for_projections(
            self.basis_functions.my_atom_indices)
            
        self.P_aqMi = {}
        for a in self.basis_functions.my_atom_indices:
            ni = self.setups[a].ni
            self.P_aqMi[a] = np.empty((nq, nao, ni), self.dtype)

        for kpt in self.kpt_u:
            q = kpt.q
            kpt.P_aMi = dict([(a, P_qMi[q])
                              for a, P_qMi in self.P_aqMi.items()])

        self.timer.start('TCI: Calculate S, T, P')
        # Calculate lower triangle of S and T matrices:
        self.tci.calculate(spos_ac, S_qMM, T_qMM, self.P_aqMi)
        add_paw_correction_to_overlap(self.setups, self.P_aqMi, S_qMM,
                                      self.ksl.Mstart, self.ksl.Mstop)
        self.timer.stop('TCI: Calculate S, T, P')

        S_MM = None # allow garbage collection of old S_qMM after redist
        S_qMM = self.ksl.distribute_overlap_matrix(S_qMM)
        T_qMM = self.ksl.distribute_overlap_matrix(T_qMM)

        for kpt in self.kpt_u:
            q = kpt.q
            kpt.S_MM = S_qMM[q]
            kpt.T_MM = T_qMM[q]


        if (debug and self.band_comm.size == 1 and self.gd.comm.rank == 0 and
            nao > 0 and not self.ksl.using_blacs):
            # S and T are summed only on comm master, so check only there
            from numpy.linalg import eigvalsh
            self.timer.start('Check positive definiteness')
            for S_MM in S_qMM:
                tri2full(S_MM, UL='L')
                smin = eigvalsh(S_MM).real.min()
                if smin < 0:
                    raise RuntimeError('Overlap matrix has negative '
                                       'eigenvalue: %e' % smin)
            self.timer.stop('Check positive definiteness')
        self.positions_set = True
        self.S_qMM = S_qMM
        self.T_qMM = T_qMM

    def initialize(self, density, hamiltonian, spos_ac):
        if density.nt_sG is None:
            if self.kpt_u[0].f_n is None or self.kpt_u[0].C_nM is None:
                density.initialize_from_atomic_densities(self.basis_functions)
            else:
                # We have the info we need for a density matrix, so initialize
                # from that instead of from scratch.  This will be the case
                # after set_positions() during a relaxation
                density.initialize_from_wavefunctions(self)
        else:
            # After a restart, nt_sg doesn't exist yet, so we'll have to
            # make sure it does.  Of course, this should have been taken care
            # of already by this time, so we should improve the code elsewhere
            density.calculate_normalized_charges_and_mix()
        hamiltonian.update(density)
           
    def calculate_density_matrix(self, f_n, C_nM, rho_MM=None):
        # ATLAS can't handle uninitialized output array:
        #rho_MM.fill(42)

        self.timer.start('Calculate density matrix')
        rho_MM = self.ksl.calculate_density_matrix(f_n, C_nM, rho_MM)
        self.timer.stop('Calculate density matrix')
        return rho_MM

        # ----------------------------
        if 1:
            # XXX Should not conjugate, but call gemm(..., 'c')
            # Although that requires knowing C_Mn and not C_nM.
            # that also conforms better to the usual conventions in literature
            Cf_Mn = C_nM.T.conj() * f_n
            gemm(1.0, C_nM, Cf_Mn, 0.0, rho_MM, 'n')
            self.bd.comm.sum(rho_MM)
        else:
            # Alternative suggestion. Might be faster. Someone should test this
            C_Mn = C_nM.T.copy()
            r2k(0.5, C_Mn, f_n * C_Mn, 0.0, rho_MM)
            tri2full(rho_MM)

    def calculate_density_matrix_delta(self, d_nn, C_nM, rho_MM=None):
        # ATLAS can't handle uninitialized output array:
        #rho_MM.fill(42)

        self.timer.start('Calculate density matrix')
        rho_MM = self.ksl.calculate_density_matrix_delta(d_nn, C_nM, rho_MM)
        self.timer.stop('Calculate density matrix')
        return rho_MM

    def add_to_density_from_k_point_with_occupation(self, nt_sG, kpt, f_n):
        """Add contribution to pseudo electron-density. Do not use the standard
        occupation numbers, but ones given with argument f_n."""
        # Custom occupations are used in calculation of response potential
        # with GLLB-potential
        Mstart = self.basis_functions.Mstart
        Mstop = self.basis_functions.Mstop
        if kpt.rho_MM is None:
            rho_MM = self.calculate_density_matrix(f_n, kpt.C_nM)
            if hasattr(kpt, 'c_on'):
                assert self.bd.comm.size == 1
                d_nn = np.zeros((self.bd.mynbands, self.bd.mynbands), dtype=kpt.C_nM.dtype)
                for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                    assert abs(c_n.imag).max() < 1e-14
                    d_nn += ne * np.outer(c_n.conj(), c_n).real
                rho_MM += self.calculate_density_matrix_delta(d_nn, kpt.C_nM)
        else:
            rho_MM = kpt.rho_MM
        self.timer.start('Construct density')
        self.basis_functions.construct_density(rho_MM,
                                               nt_sG[kpt.s], kpt.q)
        self.timer.stop('Construct density')

    def add_to_kinetic_density_from_k_point(self, taut_G, kpt):
        raise NotImplementedError('Kinetic density calculation for LCAO '
                                  'wavefunctions is not implemented.')

    def calculate_forces(self, hamiltonian, F_av):
        self.timer.start('LCAO forces')
        from gpaw.kohnsham_layouts import BlacsOrbitalLayouts
        from gpaw.blacs import BlacsGrid, Redistributor
        isblacs = isinstance(self.ksl, BlacsOrbitalLayouts) # XXX
        
        spos_ac = self.tci.atoms.get_scaled_positions() % 1.0
        nao = self.ksl.nao
        mynao = self.ksl.mynao
        nq = len(self.kd.ibzk_qc)
        dtype = self.dtype
        ksl = self.ksl

        self.timer.start('LCAO forces: tci derivative')
        if not isblacs:
            dThetadR_qvMM = np.empty((nq, 3, mynao, nao), dtype)
            dTdR_qvMM = np.empty((nq, 3, mynao, nao), dtype)
            dPdR_aqvMi = {}
            for a in self.basis_functions.my_atom_indices:
                ni = self.setups[a].ni
                dPdR_aqvMi[a] = np.empty((nq, 3, nao, ni), dtype)
            self.tci.calculate_derivative(spos_ac, dThetadR_qvMM, dTdR_qvMM,
                                          dPdR_aqvMi)
            comm = self.gd.comm
            comm.sum(dThetadR_qvMM)
            comm.sum(dTdR_qvMM)
        self.timer.stop('LCAO forces: tci derivative')
        
        tci = self.tci
        
        P_aqMi = self.P_aqMi
        
        gd = self.gd
        bd = self.bd

        Mstart = ksl.Mstart
        Mstop = ksl.Mstop
        
        bfs = self.basis_functions
        my_atom_indices = bfs.my_atom_indices
        atom_indices = bfs.atom_indices
        
        def _slices(indices):
            for a in indices:
                M1 = bfs.M_a[a] - Mstart
                M2 = M1 + self.setups[a].niAO
                if M2 > 0:
                    yield a, max(0, M1), M2
        
        def slices():
            return _slices(atom_indices)
        
        def my_slices():
            return _slices(my_atom_indices)
        
        #
        #         -----                    -----
        #          \    -1                  \    *
        # E      =  )  S     H    rho     =  )  c     eps  f  c
        #  mu nu   /    mu x  x z    z nu   /    n mu    n  n  n nu
        #         -----                    -----
        #          x z                       n
        #
        # We use the transpose of that matrix.  The first form is used
        # if rho is given, otherwise the coefficients are used.
        self.timer.start('LCAO forces: initial')

        def get_density_matrix(f_n, C_nM, redistributor):
            rho1_mm = ksl.calculate_blocked_density_matrix(f_n, C_nM).conj()
            rho_mm = redistributor.redistribute(rho1_mm)
            return rho_mm
        

        rhoT_uMM = []
        ET_uMM = []

        if not isblacs:
            if self.kpt_u[0].rho_MM is None:
                self.timer.start('get density matrix')
                for kpt in self.kpt_u:
                    rhoT_MM = self.ksl.get_transposed_density_matrix(kpt.f_n,
                                                                     kpt.C_nM)
                    rhoT_uMM.append(rhoT_MM)
                    ET_MM = self.ksl.get_transposed_density_matrix(kpt.f_n
                                                                   * kpt.eps_n,
                                                                   kpt.C_nM)
                    ET_uMM.append(ET_MM)

                    if hasattr(kpt, 'c_on'):
                        # XXX does this work with BLACS/non-BLACS/etc.?
                        assert self.bd.comm.size == 1
                        d_nn = np.zeros((self.bd.mynbands, self.bd.mynbands), dtype=kpt.C_nM.dtype)
                        for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                                d_nn += ne * np.outer(c_n.conj(), c_n)
                        rhoT_MM += self.ksl.get_transposed_density_matrix_delta(d_nn, kpt.C_nM)
                        ET_MM += self.ksl.get_transposed_density_matrix_delta(d_nn * kpt.eps_n, kpt.C_nM)
                self.timer.stop('get density matrix')
            else:
                # XXX wont work now
                H_MM = self.eigensolver.calculate_hamiltonian_matrix(hamiltonian,
                                                                     self,
                                                                     kpt)
                tri2full(H_MM)
                S_MM = self.S_qMM[q].copy()
                tri2full(S_MM)
                ET_MM = np.linalg.solve(S_MM, gemmdot(H_MM, kpt.rho_MM)).T.copy()
                del S_MM, H_MM
                rhoT_MM = kpt.rho_MM.T.copy()
        self.timer.stop('LCAO forces: initial')

        if isblacs: # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            pcutoff_a = [max([pt.get_cutoff() for pt in setup.pt_j])
                         for setup in self.setups]
            phicutoff_a = [max([phit.get_cutoff() for phit in setup.phit_j])
                           for setup in self.setups]

            #if self.world.rank == 0:
            #    print 'p cut'
            #    print pcutoff_a
            #    print 'phi cut'
            #    print phicutoff_a

            # XXX should probably use bdsize x gdsize instead
            # That would be consistent with some existing grids
            slgrid = ksl.blockgrid
            grid = BlacsGrid(ksl.block_comm, self.gd.comm.size,
                             self.bd.comm.size)
            
            blocksize1 = -(-nao // grid.nprow)
            blocksize2 = -(-nao // grid.npcol)
            # XXX what are rows and columns actually?
            desc = grid.new_descriptor(nao, nao, blocksize1, blocksize2)
            
            rhoT_umm = []
            ET_umm = []
            redistributor = Redistributor(grid.comm, ksl.mmdescriptor, desc)
            Fpot_av = np.zeros_like(F_av)
            for u, kpt in enumerate(self.kpt_u):
                self.timer.start('get density matrix')
                rhoT_mm = get_density_matrix(kpt.f_n, kpt.C_nM, redistributor)
                rhoT_umm.append(rhoT_mm)
                self.timer.stop('get density matrix')
                
                self.timer.start('LCAO forces: potential')
                rhoT_mM = ksl.distribute_to_columns(rhoT_mm, desc)
                
                vt_G = hamiltonian.vt_sG[kpt.s]
                Fpot_av += bfs.calculate_force_contribution(vt_G, rhoT_mM,
                                                            kpt.q)
                del rhoT_mM
                self.timer.stop('LCAO forces: potential')
            
            self.timer.start('get density matrix')
            for kpt in self.kpt_u:
                ET_mm = get_density_matrix(kpt.f_n * kpt.eps_n, kpt.C_nM,
                                           redistributor)
                ET_umm.append(ET_mm)
            self.timer.stop('get density matrix')
            
            M1start = blocksize1 * grid.myrow
            M2start = blocksize2 * grid.mycol
            
            M1stop = min(M1start + blocksize1, nao)
            M2stop = min(M2start + blocksize2, nao)
            
            m1max = M1stop - M1start
            m2max = M2stop - M2start


        
        if not isblacs:
            # Kinetic energy contribution
            #
            #           ----- d T
            #  a         \       mu nu
            # F += 2 Re   )   -------- rho
            #            /    d R         nu mu
            #           -----    mu nu
            #        mu in a; nu
            #
            Fkin_av = np.zeros_like(F_av)
            for u, kpt in enumerate(self.kpt_u):
                dEdTrhoT_vMM = (dTdR_qvMM[kpt.q]
                                * rhoT_uMM[u][np.newaxis]).real
                for a, M1, M2 in my_slices():
                    Fkin_av[a, :] += 2.0 * dEdTrhoT_vMM[:, M1:M2].sum(-1).sum(-1)
            del dEdTrhoT_vMM


            # Density matrix contribution due to basis overlap
            #
            #            ----- d Theta
            #  a          \           mu nu
            # F  += -2 Re  )   ------------  E
            #             /        d R        nu mu
            #            -----        mu nu
            #         mu in a; nu
            #
            Ftheta_av = np.zeros_like(F_av)
            for u, kpt in enumerate(self.kpt_u):
                dThetadRE_vMM = (dThetadR_qvMM[kpt.q]
                                 * ET_uMM[u][np.newaxis]).real
                for a, M1, M2 in my_slices():
                    Ftheta_av[a, :] += -2.0 * dThetadRE_vMM[:, M1:M2].sum(-1).sum(-1)
            del dThetadRE_vMM

        from gpaw.lcao.overlap import OppositeDirection,\
             TwoCenterIntegralCalculator, AtomicDisplacement, PairFilter

        if isblacs:
            self.timer.start('prepare TCI loop')
            M_a = bfs.M_a
            
            Fkin2_av = np.zeros_like(F_av)
            Ftheta2_av = np.zeros_like(F_av)
            Frho2_av = np.zeros_like(F_av)

            cell_cv = tci.atoms.cell
            spos_ac = tci.atoms.get_scaled_positions() % 1.0

            derivativecalc = TwoCenterIntegralCalculator(self.kd.ibzk_qc,
                                                         derivative=True)
            overlapcalc = TwoCenterIntegralCalculator(self.kd.ibzk_qc,
                                                      derivative=False)

            def get_phases(offset):
                return overlapcalc.phaseclass(overlapcalc.ibzk_qc, offset)

            # XXX this is not parallel *AT ALL*.  Parallelize somehow.
            nl = tci.atompairs.pairs.neighbors
            self.timer.start('get neighbors')

            r_and_offset_aao = get_r_and_offsets(nl, spos_ac, cell_cv)
            atompairs = r_and_offset_aao.keys()
            atompairs.sort()

            self.timer.stop('get neighbors')

            T_expansions = tci.T_expansions
            Theta_expansions = tci.Theta_expansions
            P_expansions = tci.P_expansions
            nq = len(self.ibzk_qc)
            
            sendreqs = []
            recvreqs = []
            dH_asp = hamiltonian.dH_asp
            comm = grid.comm

            ranks_bg = np.arange(self.gd.comm.size *
                                 self.bd.comm.size).reshape(self.bd.comm.size,
                                                            self.gd.comm.size)
            
            def rank2bgrank(rank):
                b_rank, g_rank = divmod(rank, self.gd.comm.size)
                assert ranks_bg[b_rank, g_rank] == rank, (ranks_bg[b_rank,
                                                                   g_rank],
                                                          rank)
                return b_rank, g_rank

##             for a2 in range(len(self.setups)):
##             #for a2 in bfs.my_atom_indices:
##                 I1 = P_expansions.M2_a[a2]
##                 I2 = I1 + P_expansions.shape[1]

##                 continue
            
##                 Pcpu_x1, Pcpu_y1 = Pdesc.index2grid(0, I1)
##                 Pcpu_x2, Pcpu_y2 = Pdesc.index2grid(nao - 1, I2 - 1)

##                 if a2 in dH_asp:# and self.bd.comm.rank == 0:
##                     print comm.rank, a2, (Pcpu_y1, Pcpu_y2, Pcpu_x1, Pcpu_x2)
##                     for Pcpu_y in range(Pcpu_y1, Pcpu_y2 + 1):
##                         for Pcpu_x in range(Pcpu_x1, Pcpu_x2 + 1):
##                             dstrank = grid.coords2rank(Pcpu_x, Pcpu_y)
##                             b_rank, g_rank = rank2bgrank(dstrank)
##                             if g_rank == self.gd.comm.rank:
##                                 continue
##                             if b_rank != self.bd.comm.rank:
##                                 continue
##                             print comm.rank, 'send', a2, 'to', dstrank
##                             #sendreqs.append(comm.send(dH_asp[a2], dstrank,
##                             #                          #tag=a2,
##                             #                          block=False))

##                 # XXXXXX this will break when Pcpu_y1 < mycol < Pcpu_y2
##                 # (which should not really happen in practical cases though)
##                 if Pcpu_y1 == grid.mycol or Pcpu_y2 == grid.mycol:
##                     g_rank = bfs.sphere_a[a2].rank
##                     b_rank = self.bd.rank # aha
                    
##                     srcrank = ranks_bg[b_rank, g_rank]
                    
##                     #b_rank, g_rank = rank2bgrank(srcrank)
##                     if g_rank == self.gd.rank:
##                         dH_sp = dH_asp[a2]
##                     else:
##                     #elif self.bd.rank == b_rank:
##                         ni = self.setups[a2].ni
##                         dH_sp = np.empty((self.nspins, ni * (ni + 1) // 2))
##                         print comm.rank, 'recv', a2, 'from', srcrank
##                         #recvreqs.append(comm.receive(dH_sp, srcrank, #tag=a2,
##                         #                             block=False))

            #mydH_asp = {}

            self.timer.start('broadcast dH')
            alldH_asp = {}
            for a in range(len(self.setups)):
                gdrank = bfs.sphere_a[a].rank
                if gdrank == gd.rank:
                    dH_sp = dH_asp[a]
                else:
                    ni = self.setups[a].ni
                    dH_sp = np.empty((self.nspins, ni * (ni + 1) // 2))
                gd.comm.broadcast(dH_sp, gdrank)
                # okay, now everyone gets copies of dH_sp
                alldH_asp[a] = dH_sp
            self.timer.stop('broadcast dH')
            #sendto = set()
            #recvfrom = set()

            #P_a = P_expansions.M2_a

            #disps = []
            #for disp in derivativecalc.iter(ProjectorPairFilter(tci.atompairs)):
            #    disps.append(disp)
                
            
            #for disp in overlapcalc.iter(ProjectorPairFilter(tci.atompairs)):
            #for disp in disps:
            #    a1 = disp.a1
            #    a2 = disp.a2
            if 0: # no sending if dH_asp is broadcast
                if a2 in self.P_aqMi and a2 >= a1:
                    # figure out who needs dH, and send
                    M1 = M_a[a1]
                    P1 = P_a[a2]

                    P_expansion = P_expansions.get(a1, a2)
                    nM, nP = P_expansion.shape

                    M2 = M1 + nM
                    P2 = P1 + nP
                    xrank1, yrank1 = Pdesc.index2grid(M1, P1)
                    xrank2, yrank2 = Pdesc.index2grid(M2 - 1, P2 - 1)

                    assert xrank2 >= xrank1, ((xrank1, xrank2), (M1, M2))
                    assert yrank2 >= yrank1, ((yrank1, yrank2), (Pstart, Pend))

                    for xrank in range(xrank1, xrank2 + 1):
                        for yrank in range(yrank1, yrank2 + 1):
                            rank = grid.coords2rank(xrank, yrank)
                            b_rank, g_rank = rank2bgrank(rank)
                            if g_rank == self.gd.comm.rank:
                                assert a2 in self.P_aqMi
                                continue
                            if b_rank == self.bd.comm.rank:
                                #assert False, (comm.rank, rank, b_rank, g_rank)
                                if not (rank, a2) in sendto:
                                    print self.world.rank, 'SEND'
                                    sendto.add((rank, a2))

                ### XXX this is repeated below; work something smarter out
                m1start = M_a[a1] - M1start

                if m1start >= blocksize1:
                    continue

                P_expansion = P_expansions.get(a1, a2)
                m1stop = min(m1start + P_expansion.shape[0], m1max)

                if m1stop <= 0:
                    continue

                I1 = P_expansions.M2_a[a1]
                M1 = P_expansions.M1_a[a1]

                M2 = M1 + P_expansions.shape[0]
                I2 = I1 + P_expansions.shape[1]

                P_qmi = P_expansion.zeros((nq,), dtype=dtype)
                disp.evaluate_overlap(P_expansion, P_qmi)

                if 0: # no recving if dH_asp is broadcast
                    for a in [a2]:#[a1, a2]:
                        if a in self.P_aqMi:
                            mydH_asp[a] = dH_asp[a]
                        else:
                            src_gdrank = bfs.sphere_a[a].rank
                            srcrank = ranks_bg[self.bd.comm.rank, src_gdrank]
                            if not (srcrank, a) in recvfrom:
                                print self.world.rank, 'RECV'
                                recvfrom.add((srcrank, a))


            if 0: # no sendrecv if dH_asp is broadcast
                for dstrank, a2 in sendto:
                    sendreqs.append(comm.send(dH_asp[a2], dstrank,
                                          tag=a2,
                                          block=False))
                for srcrank, a2 in recvfrom:
                    ni = self.setups[a2].ni
                    dH_sp = np.empty((self.nspins, ni * (ni + 1) // 2))
                    recvreqs.append(comm.receive(dH_sp, srcrank, tag=a2,
                                                 block=False))
                    mydH_asp[a2] = dH_sp

                self.world.barrier()

                for (rank, a2) in recvfrom:
                    assert not a2 in self.P_aqMi

                a2values = [a2 for (rank, a2) in recvfrom]
                for a2 in self.P_aqMi:
                    assert not a2 in a2values


                print comm.rank, 'sendto', sendto, 'recvfrom', recvfrom
                self.world.barrier()
                assert comm.sum(len(sendreqs)) == comm.sum(len(recvreqs))
                comm.waitall(sendreqs + recvreqs)

            # This will get sort of hairy.  We need to account for some
            # three-center overlaps, such as:
            #
            #         a1
            #      Phi   ~a3    a3  ~a3     a2     a2,a1
            #   < ----  |p  > dH   <p   |Phi  > rho
            #      dR
            #
            # To this end we will loop over all pairs of atoms (a1, a3),
            # and then a sub-loop over (a3, a2).



            from gpaw.lcao.overlap import DerivativeAtomicDisplacement
            class Displacement(DerivativeAtomicDisplacement):
                def __init__(self, a1, a2, R_c, offset):
                    phases = overlapcalc.phaseclass(overlapcalc.ibzk_qc,
                                                    offset)
                    DerivativeAtomicDisplacement.__init__(self, None, a1, a2,
                                                          R_c, offset, phases)

            # Cache of Displacement objects with spherical harmonics with
            # evaluated spherical harmonics.
            disp_aao = {}

            def get_displacements(a1, a2, maxdistance):
                # XXX the way maxdistance is handled it can lead to
                # bad caching when different maxdistances are passed
                # to subsequent calls with same pair of atoms
                disp_o = disp_aao.get((a1, a2))
                if disp_o is None:
                    self.timer.start('displacements')
                    disp_o = []
                    for r, offset in r_and_offset_aao[(a1, a2)]:
                        if np.linalg.norm(r) > maxdistance:
                            continue
                        disp = Displacement(a1, a2, r, offset)
                        disp_o.append(disp)
                    disp_aao[(a1, a2)] = disp_o
                    self.timer.stop('displacements')
                return [disp for disp in disp_o if disp.r < maxdistance]
                
            # THIS IS WHERE THE REAL THING STARTS
            self.timer.stop('prepare TCI loop')
            self.timer.start('not so complicated loop')


            for (a1, a2) in atompairs:
                #for (a1, a2), disp_o in zip(atompairs, disp_aao):
                if a1 >= a2:
                    # Actually this leads to bad load balance.
                    # We should take a1 > a2 or a1 < a2 equally many times.
                    # Maybe decide which of these choices
                    # depending on whether a2 % 1 == 0
                    continue
                
                m1start = M_a[a1] - M1start
                m2start = M_a[a2] - M2start
                if m1start >= blocksize1 or m2start >= blocksize2:
                    continue

                T_expansion = T_expansions.get(a1, a2)
                Theta_expansion = Theta_expansions.get(a1, a2)
                P_expansion = P_expansions.get(a1, a2)
                nm1, nm2 = T_expansion.shape

                m1stop = min(m1start + nm1, m1max)
                m2stop = min(m2start + nm2, m2max)

                if m1stop <= 0 or m2stop <= 0:
                    continue

                m1start = max(m1start, 0)
                m2start = max(m2start, 0)
                J1start = max(0, M1start - M_a[a1])
                J2start = max(0, M2start - M_a[a2])
                M1stop = J1start + m1stop - m1start
                J2stop = J2start + m2stop - m2start

                dTdR_qvmm = T_expansion.zeros((nq, 3), dtype=dtype)
                dThetadR_qvmm = Theta_expansion.zeros((nq, 3), dtype=dtype)

                disp_o = get_displacements(a1, a2,
                                           phicutoff_a[a1] + phicutoff_a[a2])
                for disp in disp_o:
                    disp.evaluate_overlap(T_expansion, dTdR_qvmm)
                    disp.evaluate_overlap(Theta_expansion, dThetadR_qvmm)

                for u, kpt in enumerate(self.kpt_u):
                    rhoT_mm = rhoT_umm[u][m1start:m1stop, m2start:m2stop]
                    ET_mm = ET_umm[u][m1start:m1stop, m2start:m2stop]
                    Fkin_v = 2.0 * (dTdR_qvmm[kpt.q][:, J1start:M1stop,
                                                     J2start:J2stop]
                                    * rhoT_mm[np.newaxis]).real.sum(-1).sum(-1)
                    Ftheta_v = 2.0 * (dThetadR_qvmm[kpt.q][:, J1start:M1stop,
                                                           J2start:J2stop]
                                      * ET_mm[np.newaxis]).real.sum(-1).sum(-1)
                    Fkin2_av[a1] += Fkin_v
                    Fkin2_av[a2] -= Fkin_v
                    Ftheta2_av[a1] -= Ftheta_v
                    Ftheta2_av[a2] += Ftheta_v

            Fkin_av = Fkin2_av
            Ftheta_av = Ftheta2_av
            self.timer.stop('not so complicated loop')

            dHP_and_dSP_aauim = {}

            a2values = {}
            for (a2, a3) in atompairs:
                if not a3 in a2values:
                    a2values[a3] = []
                a2values[a3].append(a2)

            #if self.world.rank == 0:
            #    print 'neighbour counts'
            #    X = [len(a2values[x]) for x in a2values]
            #    print sum(X) / float(len(X))
            #    print X

            Fatom_av = np.zeros_like(F_av)
            Frho_av = np.zeros_like(F_av)
            self.timer.start('complicated loop')
            for a1, a3 in atompairs:
                if a1 == a3:
                    continue
                
                m1start = M_a[a1] - M1start
                if m1start >= blocksize1:
                    continue
                
                P_expansion = P_expansions.get(a1, a3)
                nm1 = P_expansion.shape[0]
                m1stop = min(m1start + nm1, m1max)
                if m1stop <= 0:
                    continue

                m1start = max(m1start, 0)
                J1start = max(0, M1start - M_a[a1])
                J1stop = J1start + m1stop - m1start

                disp_o = get_displacements(a1, a3,
                                           phicutoff_a[a1] + pcutoff_a[a3])
                if len(disp_o) == 0:
                    continue
                
                dPdR_qvmi = P_expansion.zeros((nq, 3), dtype=dtype)
                for disp in disp_o:
                    disp.evaluate_overlap(P_expansion, dPdR_qvmi)

                dPdR_qvmi = dPdR_qvmi[:, :, J1start:J1stop, :].copy()
                for a2 in a2values[a3]:
                    m2start = M_a[a2] - M2start
                    if m2start >= blocksize2:
                        continue

                    P_expansion2 = P_expansions.get(a2, a3)
                    nm2 = P_expansion2.shape[0]
                    m2stop = min(m2start + nm2, m2max)
                    if m2stop <= 0:
                        continue
                    
                    disp_o = get_displacements(a2, a3,
                                               phicutoff_a[a2] + pcutoff_a[a3])
                    if len(disp_o) == 0:
                        continue

                    m2start = max(m2start, 0)
                    J2start = max(0, M2start - M_a[a2])
                    J2stop = J2start + m2stop - m2start

                    if (a2, a3) in dHP_and_dSP_aauim:
                        dHP_uim, dSP_uim = dHP_and_dSP_aauim[(a2, a3)]
                    else:
                        P_qmi = P_expansion2.zeros((nq,), dtype=dtype)
                        for disp in disp_o:
                            # XXX We only use evaluate_direct, but disp
                            # already contain spherical harmonics derivatives.
                            #
                            # Maybe use lazy evaluation of derivatives?
                            disp.evaluate_direct(P_expansion2, P_qmi)
                        P_qmi = P_qmi[:, J2start:J2stop].copy()
                        dH_sp = alldH_asp[a3]
                        dS_ii = self.setups[a3].dO_ii
                        
                        dHP_uim = []
                        dSP_uim = []
                        for u, kpt in enumerate(self.kpt_u):
                            dH_ii = unpack(dH_sp[kpt.s])
                            dHP_im = np.dot(P_qmi[kpt.q],
                                            dH_ii).T.conj()
                            # XXX only need nq of these
                            dSP_im = np.dot(P_qmi[kpt.q], dS_ii).T.conj()
                            dHP_uim.append(dHP_im)
                            dSP_uim.append(dSP_im)
                            dHP_and_dSP_aauim[(a2, a3)] = dHP_uim, dSP_uim
                    
                    for u, kpt in enumerate(self.kpt_u):
                        rhoT_mm = rhoT_umm[u][m1start:m1stop, m2start:m2stop]
                        ET_mm = ET_umm[u][m1start:m1stop, m2start:m2stop]
                        dPdRdHP_vmm = np.dot(dPdR_qvmi[kpt.q], dHP_uim[u])
                        dPdRdSP_vmm = np.dot(dPdR_qvmi[kpt.q], dSP_uim[u])
                        
                        Fatom_c = 2.0 * (dPdRdHP_vmm
                                         * rhoT_mm).real.sum(-1).sum(-1)
                        Frho_c = 2.0 * (dPdRdSP_vmm
                                        * ET_mm).real.sum(-1).sum(-1)
                        Fatom_av[a1] += Fatom_c
                        Fatom_av[a3] -= Fatom_c

                        Frho_av[a1] -= Frho_c
                        Frho_av[a3] += Frho_c
                        
            self.timer.stop('complicated loop')
        
        if not isblacs:
            # Potential contribution
            #
            #           -----      /  d Phi  (r)
            #  a         \        |        mu    ~
            # F += -2 Re  )       |   ---------- v (r)  Phi  (r) dr rho
            #            /        |     d R                nu          nu mu
            #           -----    /         a
            #        mu in a; nu
            #
            self.timer.start('LCAO forces: potential')
            Fpot_av = np.zeros_like(F_av)
            for u, kpt in enumerate(self.kpt_u):
                vt_G = hamiltonian.vt_sG[kpt.s]
                Fpot_av += bfs.calculate_force_contribution(vt_G, rhoT_uMM[u],
                                                            kpt.q)
            self.timer.stop('LCAO forces: potential')

            # Density matrix contribution from PAW correction
            #
            #           -----                        -----
            #  a         \      a                     \     b
            # F +=  2 Re  )    Z      E        - 2 Re  )   Z      E
            #            /      mu nu  nu mu          /     mu nu  nu mu
            #           -----                        -----
            #           mu nu                    b; mu in a; nu
            #
            # with
            #                  b*
            #         -----  dP
            #   b      \       i mu    b   b
            #  Z     =  )   -------- dS   P
            #   mu nu  /     dR        ij  j nu
            #         -----    b mu
            #           ij
            #
            self.timer.start('LCAO forces: paw correction')
            Frho_av = np.zeros_like(F_av)
            for u, kpt in enumerate(self.kpt_u):
                work_MM = np.zeros((mynao, nao), dtype)
                ZE_MM = None
                for b in my_atom_indices:
                    setup = self.setups[b]
                    dO_ii = np.asarray(setup.dO_ii, dtype)
                    dOP_iM = np.zeros((setup.ni, nao), dtype)
                    gemm(1.0, self.P_aqMi[b][kpt.q], dO_ii, 0.0, dOP_iM, 'c')
                    for v in range(3):
                        gemm(1.0, dOP_iM, dPdR_aqvMi[b][kpt.q][v][Mstart:Mstop],
                             0.0, work_MM, 'n')
                        ZE_MM = (work_MM * ET_uMM[u]).real
                        for a, M1, M2 in slices():
                            dE = 2 * ZE_MM[M1:M2].sum()
                            Frho_av[a, v] -= dE # the "b; mu in a; nu" term
                            Frho_av[b, v] += dE # the "mu nu" term
            del work_MM, ZE_MM
            self.timer.stop('LCAO forces: paw correction')

            # Atomic density contribution
            #            -----                         -----
            #  a          \     a                       \     b
            # F  += -2 Re  )   A      rho       + 2 Re   )   A      rho
            #             /     mu nu    nu mu          /     mu nu    nu mu
            #            -----                         -----
            #            mu nu                     b; mu in a; nu
            #
            #                  b*
            #         ----- d P
            #  b       \       i mu   b   b
            # A     =   )   ------- dH   P
            #  mu nu   /    d R       ij  j nu
            #         -----    b mu
            #           ij
            #
            self.timer.start('LCAO forces: atomic density')
            Fatom_av = np.zeros_like(F_av)
            for u, kpt in enumerate(self.kpt_u):
                for b in my_atom_indices:
                    H_ii = np.asarray(unpack(hamiltonian.dH_asp[b][kpt.s]), dtype)
                    HP_iM = gemmdot(H_ii,
                                    np.ascontiguousarray(self.P_aqMi[b][kpt.q].T.conj()))
                    for v in range(3):
                        dPdR_Mi = dPdR_aqvMi[b][kpt.q][v][Mstart:Mstop]
                        ArhoT_MM = (gemmdot(dPdR_Mi, HP_iM) * rhoT_uMM[u]).real
                        for a, M1, M2 in slices():
                            dE = 2 * ArhoT_MM[M1:M2].sum()
                            Fatom_av[a, v] += dE # the "b; mu in a; nu" term
                            Fatom_av[b, v] -= dE # the "mu nu" term
            self.timer.stop('LCAO forces: atomic density')

        def printforce(F, title=None):
            F = F.copy()
            if title is not None:
                if self.world.rank == 0:
                    print title
            self.gd.comm.sum(F)
            self.bd.comm.sum(F)
            if self.world.rank == 0:
                print F*100

        F_av += Fkin_av + Fpot_av + Ftheta_av + Frho_av + Fatom_av
        self.timer.start('Wait for sum')
        self.ksl.orbital_comm.sum(F_av)
        if self.bd.comm.rank == 0:
            self.kpt_comm.sum(F_av, 0)
        self.timer.stop('Wait for sum')
        self.timer.stop('LCAO forces')

    def _get_wave_function_array(self, u, n, realspace=True):
        kpt = self.kpt_u[u]
        if kpt.C_nM is None:
            # Hack to make sure things are available after restart
            self.lazyloader.load(self)
        
        C_M = kpt.C_nM[n]

        if realspace:
            psit_G = self.gd.zeros(dtype=self.dtype)
            self.basis_functions.lcao_to_grid(C_M, psit_G, kpt.q)
            return psit_G
        else:
            return C_M

    def load_lazily(self, hamiltonian, spos_ac):
        """Horrible hack to recalculate lcao coefficients after restart."""
        class LazyLoader:
            def __init__(self, hamiltonian, spos_ac):
                self.hamiltonian = hamiltonian
                self.spos_ac = spos_ac
            
            def load(self, wfs):
                wfs.set_positions(self.spos_ac)
                wfs.eigensolver.iterate(hamiltonian, wfs)
                del wfs.lazyloader
        
        self.lazyloader = LazyLoader(hamiltonian, spos_ac)
        
    def write(self, writer, write_wave_functions=False):
        writer['Mode'] = 'lcao'
        
        if not write_wave_functions:
            return
   
        writer.dimension('nbasis', self.setups.nao)
        writer.add('WaveFunctionCoefficients',
                   ('nspins', 'nibzkpts', 'nbands', 'nbasis'),
                   dtype=self.dtype)

        for s in range(self.nspins):
            for k in range(self.nibzkpts):
                C_nM = self.collect_array('C_nM', k, s)
                writer.fill(C_nM, s, k)

    def read_coefficients(self, reader):
        for kpt in self.kpt_u:
            kpt.C_nM = self.bd.empty(self.setups.nao, dtype=self.dtype)
            for n in self.bd.get_band_indices():
                kpt.C_nM[n] = reader.get('WaveFunctionCoefficients',
                                         kpt.s, kpt.k, n)

    def estimate_memory(self, mem):
        nq = len(self.kd.ibzk_qc)
        nao = self.setups.nao
        ni_total = sum([setup.ni for setup in self.setups])
        itemsize = mem.itemsize[self.dtype]
        mem.subnode('C [qnM]', nq * self.bd.mynbands * nao * itemsize)
        nM1, nM2 = self.ksl.get_overlap_matrix_shape()
        mem.subnode('S, T [2 x qmm]', 2 * nq * nM1 * nM2 * itemsize)
        mem.subnode('P [aqMi]', nq * nao * ni_total // self.gd.comm.size)
        self.tci.estimate_memory(mem.subnode('TCI'))
        self.basis_functions.estimate_memory(mem.subnode('BasisFunctions'))
        self.eigensolver.estimate_memory(mem.subnode('Eigensolver'),
                                         self.dtype)
