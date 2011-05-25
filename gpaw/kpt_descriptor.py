# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""K-point/spin combination-descriptors

This module contains classes for defining combinations of two indices:

* Index k for irreducible kpoints in the 1st Brillouin zone.
* Index s for spin up/down if spin-polarized (otherwise ignored).

"""

import numpy as np
from ase.units import Bohr
from ase.dft import monkhorst_pack

from gpaw.symmetry import Symmetry
from gpaw.kpoint import KPoint
import _gpaw


class KPointDescriptor:
    """Descriptor-class for k-points."""

    def __init__(self, kpts, nspins=1):
        """Construct descriptor object for kpoint/spin combinations (ks-pair).

        Parameters
        ----------
        kpts: None, list of ints, or ndarray
            Specification of the k-point grid. None=Gamma, list of
            ints=Monkhorst-Pack, ndarray=user specified.
        nspins: int
            Number of spins.

        Attributes
        ============  ======================================================
        ``N_c``       Number of k-points in the different directions.
        ``nspins``    Number of spins.
        ``nibzkpts``  Number of irreducible kpoints in 1st Brillouin zone.
        ``nks``       Number of k-point/spin combinations in total.
        ``mynks``     Number of k-point/spin combinations on this CPU.
        ``gamma``     Boolean indicator for gamma point calculation.
        ``comm``      MPI-communicator for kpoint distribution.
        ============  ======================================================
        
        """

        if kpts is None:
            self.bzk_kc = np.zeros((1, 3))
            self.N_c = np.array((1, 1, 1), dtype=int)
        elif isinstance(kpts[0], int):
            self.bzk_kc = monkhorst_pack(kpts)
            self.N_c = np.array(kpts, dtype=int)
        else:
            self.bzk_kc = np.array(kpts, float)
            self.N_c = None

        self.nspins = nspins
        self.nbzkpts = len(self.bzk_kc)
        
        # Gamma-point calculation
        self.gamma = self.nbzkpts == 1 and not self.bzk_kc[0].any()

        self.symmetry = None
        self.comm = None
        self.ibzk_kc = None
        self.weight_k = None
        self.nibzkpts = None

        self.rank0 = None
        self.mynks = None
        self.ks0 = None
        self.ibzk_qc = None

    def __len__(self):
        """Return number of k-point/spin combinations of local CPU."""
        
        return self.mynks

    def set_symmetry(self, atoms, setups, usesymm, N_c=None):
        """Create symmetry object and construct irreducible Brillouin zone.

        Parameters
        ----------
        atoms: Atoms object
            Defines atom positions and types and also unit cell and
            boundary conditions.
        setups: instance of class Setups
            PAW setups for the atoms.
        usesymm: bool
            Symmetry flag.
        N_c: three int's or None
            If not None:  Check also symmetry of grid.
        """

        if (~atoms.pbc & self.bzk_kc.any(0)).any():
            raise ValueError('K-points can only be used with PBCs!')

        # Construct a Symmetry instance containing the identity operation
        # only
        # Round off
        magmom_a = atoms.get_initial_magnetic_moments().round(decimals=3)
        id_a = zip(magmom_a, setups.id_a)
        self.symmetry = Symmetry(id_a, atoms.cell / Bohr, atoms.pbc)
        
        if self.gamma or usesymm is None:
            # Point group and time-reversal symmetry neglected
            nkpts = len(self.bzk_kc)
            self.weight_k = np.ones(nkpts) / nkpts
            self.ibzk_kc = self.bzk_kc.copy()
            self.sym_k = np.zeros(nkpts)
            self.time_reversal_k = np.zeros(nkpts, bool)
            self.kibz_k = np.arange(nkpts)
        else:
            if usesymm:
                # Find symmetry operations of atoms
                self.symmetry.analyze(atoms.get_scaled_positions())
                
                if N_c is not None:
                    self.symmetry.prune_symmetries_grid(N_c)

            (self.ibzk_kc, self.weight_k,
             self.sym_k,
             self.time_reversal_k,
             self.kibz_k) = self.symmetry.reduce(self.bzk_kc)

        setups.set_symmetry(self.symmetry)

        # Number of irreducible k-points and k-point/spin combinations.
        self.nibzkpts = len(self.ibzk_kc)
        self.nks = self.nibzkpts * self.nspins
        
    def set_communicator(self, comm):
        """Set k-point communicator."""

        # Ranks < self.rank0 have mynks0 k-point/spin combinations and
        # ranks >= self.rank0 have mynks0+1 k-point/spin combinations.
        mynks0, x = divmod(self.nks, comm.size)
        self.rank0 = comm.size - x
        self.comm = comm

        # My number and offset of k-point/spin combinations
        self.mynks, self.ks0 = self.get_count(), self.get_offset()

        if self.nspins == 2 and comm.size == 1:
            # Avoid duplicating k-points in local list of k-points.
            self.ibzk_qc = self.ibzk_kc.copy()
        else:
            self.ibzk_qc = np.vstack((self.ibzk_kc,
                                      self.ibzk_kc))[self.get_slice()]

    def create_k_points(self, gd):
        """Return a list of KPoints."""
 
        sdisp_cd = gd.sdisp_cd

        kpt_u = []

        for ks in range(self.ks0, self.ks0 + self.mynks):
            s, k = divmod(ks, self.nibzkpts)
            q = (ks - self.ks0) % self.nibzkpts
            weight = self.weight_k[k] * 2 / self.nspins
            if self.gamma:
                phase_cd = np.ones((3, 2), complex)
            else:
                phase_cd = np.exp(2j * np.pi *
                                  sdisp_cd * self.ibzk_kc[k, :, np.newaxis])
            kpt_u.append(KPoint(weight, s, k, q, phase_cd))

        return kpt_u

    def collect(self, a_ux, broadcast=True):
        """Collect distributed data to all."""

        if self.comm.rank == 0 or broadcast:
            xshape = a_ux.shape[1:]
            a_skx = np.empty((self.nspins, self.nibzkpts) + xshape, a_ux.dtype)
            a_Ux = a_skx.reshape((-1,) + xshape)
        else:
            a_skx = None

        if self.comm.rank > 0:
            self.comm.send(a_ux, 0)
        else:
            u1 = self.get_count(0)
            a_Ux[0:u1] = a_ux
            requests = []
            for rank in range(1, self.comm.size):
                u2 = u1 + self.get_count(rank)
                requests.append(self.comm.receive(a_Ux[u1:u2], rank,
                                                  block=False))
                u1 = u2
            assert u1 == len(a_Ux)
            self.comm.waitall(requests)
        
        if broadcast:
            self.comm.broadcast(a_Ux, 0)

        return a_skx

    def transform_wave_function(self, psit_G, k):
        """Transform wave function from IBZ to BZ.

        k is the index of the desired k-point in the full BZ.
        """
        
        s = self.sym_k[k]
        time_reversal = self.time_reversal_k[k]
        op_cc = np.linalg.inv(self.symmetry.op_scc[s]).round().astype(int)

        # Identity
        if (np.abs(op_cc - np.eye(3, dtype=int)) < 1e-10).all():
            if time_reversal:
                return psit_G.conj()
            else:
                return psit_G
        # General point group symmetry
        else:
            ik = self.kibz_k[k]
            kibz_c = self.ibzk_kc[ik]
            b_g = np.zeros_like(psit_G)
            if time_reversal:
                kbz_c = -np.dot(self.symmetry.op_scc[s], kibz_c)
                _gpaw.symmetrize_wavefunction(psit_G, b_g, op_cc.copy(),
                                              kibz_c, -kbz_c)
                return b_g.conj()
            else:
                kbz_c = np.dot(self.symmetry.op_scc[s], kibz_c)
                _gpaw.symmetrize_wavefunction(psit_G, b_g, op_cc.copy(),
                                              kibz_c, kbz_c)
                return b_g

    def find_k_plus_q(self, q_c, kpts_k=None):
        """Find the indices of k+q for all kpoints in the Brillouin zone.

        In case that k+q is outside the BZ, the k-point inside the BZ
        corresponding to k+q is given.
        
        Parameters
        ----------
        q_c: ndarray
            Coordinates for the q-vector in units of the reciprocal
            lattice vectors.
        kpts_k: list of ints
            Restrict search to specified k-points.

        """

        # Monkhorst-pack grid
        if self.N_c is not None:
            N_c = self.N_c
            dk_c = 1. / N_c
            kmax_c = (N_c - 1) * dk_c / 2.

        if kpts_k is None:
            kpts_kc = self.bzk_kc
        else:
            kpts_kc = self.bzk_kc[kpts_k]
            
        # k+q vectors
        kplusq_kc = kpts_kc + q_c

        # Translate back into the first BZ
        if self.N_c is not None:
            kplusq_kc[np.where(kplusq_kc > 0.5)] -= 1.
            kplusq_kc[np.where(kplusq_kc <= -0.5)] += 1.

        # List of k+q indices
        kplusq_k = []

        # N = np.zeros(3, dtype=int)
        
        # Find index of k+q vector
        for kplusq, kplusq_c in enumerate(kplusq_kc):

            # Calculate index for Monkhorst-Pack grids
            if self.N_c is not None:
                N = np.asarray(np.round((kplusq_c + kmax_c) / dk_c),
                               dtype=int)
                kplusq_k.append(N[2] + N[1] * N_c[2] +
                                N[0] * N_c[2] * N_c[1])
            else:
                k = np.argmin(np.sum(np.abs(self.bzk_kc - kplusq_c), axis=1))
                kplusq_k.append(k)

            # Check the k+q vector index
            k_c = self.bzk_kc[kplusq_k[kplusq]]
            assert abs(kplusq_c - k_c).sum() < 1e-8, "Could not find k+q!"
        
        return kplusq_k

    def get_bz_q_points(self, folding=True):
        """Return the q=k1-k2."""

        bzk_kc = self.bzk_kc
        # Get all q-points
        all_qs = []
        for k1 in bzk_kc:
            for k2 in bzk_kc:
                all_qs.append(k1-k2)
        all_qs = np.array(all_qs)

        # Fold q-points into Brillouin zone
        if folding:
            all_qs[np.where(all_qs > 0.501)] -= 1.
            all_qs[np.where(all_qs < -0.499)] += 1.

        # Make list of non-identical q-points in full BZ
        bz_qs = [all_qs[0]]
        for q_a in all_qs:
            q_in_list = False
            for q_b in bz_qs:
                if (abs(q_a[0]-q_b[0]) < 0.01 and
                    abs(q_a[1]-q_b[1]) < 0.01 and
                    abs(q_a[2]-q_b[2]) < 0.01):
                    q_in_list = True
                    break
            if q_in_list == False:
                bz_qs.append(q_a)
        self.bzq_kc = bz_qs

        return

    def where_is_q(self, q_c, folding=True):
        """Find the index of q points."""

        if folding:
            q_c[np.where(q_c>0.501)] -= 1
            q_c[np.where(q_c<-0.499)] += 1

        found = False
        for ik in range(len(self.bzq_kc)):
            if (np.abs(self.bzq_kc[ik] - q_c) < 1e-8).all():
                found = True
                return ik
                break
            
        if found is False:
            print self.bzq_kc, q_c
            raise ValueError('q-points can not be found!')
        

    def get_count(self, rank=None):
        """Return the number of ks-pairs which belong to a given rank."""

        if rank is None:
            rank = self.comm.rank
        assert rank in xrange(self.comm.size)
        mynks0 = self.nks // self.comm.size
        mynks = mynks0
        if rank >= self.rank0:
            mynks += 1
        return mynks

    def get_offset(self, rank=None):
        """Return the offset of the first ks-pair on a given rank."""

        if rank is None:
            rank = self.comm.rank
        assert rank in xrange(self.comm.size)
        mynks0 = self.nks // self.comm.size
        ks0 = rank * mynks0
        if rank >= self.rank0:
            ks0 += rank - self.rank0
        return ks0

    def get_rank_and_index(self, s, k):
        """Find rank and local index of k-point/spin combination."""
        
        u = self.where_is(s, k)
        rank, myu = self.who_has(u)
        return rank, myu
   
    def get_slice(self, rank=None):
        """Return the slice of global ks-pairs which belong to a given rank."""
        
        if rank is None:
            rank = self.comm.rank
        assert rank in xrange(self.comm.size)
        mynks, ks0 = self.get_count(rank), self.get_offset(rank)
        uslice = slice(ks0, ks0 + mynks)
        return uslice

    def get_indices(self, rank=None):
        """Return the global ks-pair indices which belong to a given rank."""
        
        uslice = self.get_slice(rank)
        return np.arange(*uslice.indices(self.nks))

    def get_ranks(self):
        """Return array of ranks as a function of global ks-pair indices."""
        
        ranks = np.empty(self.nks, dtype=int)
        for rank in range(self.comm.size):
            uslice = self.get_slice(rank)
            ranks[uslice] = rank
        assert (ranks >= 0).all() and (ranks < self.comm.size).all()
        return ranks

    def who_has(self, u):
        """Convert global index to rank information and local index."""

        mynks0 = self.nks // self.comm.size
        if u < mynks0 * self.rank0:
            rank, myu = divmod(u, mynks0)
        else:
            rank, myu = divmod(u - mynks0 * self.rank0, mynks0 + 1)
            rank += self.rank0
        return rank, myu

    def global_index(self, myu, rank=None):
        """Convert rank information and local index to global index."""
        
        if rank is None:
            rank = self.comm.rank
        assert rank in xrange(self.comm.size)
        ks0 = self.get_offset(rank)
        u = ks0 + myu
        return u

    def what_is(self, u):
        """Convert global index to corresponding kpoint/spin combination."""
        
        s, k = divmod(u, self.nibzkpts)
        return s, k

    def where_is(self, s, k):
        """Convert kpoint/spin combination to the global index thereof."""
        
        u = k + self.nibzkpts * s
        return u

    #def get_size_of_global_array(self):
    #    return (self.nspins*self.nibzkpts,)
    #
    #def ...



class KPointDescriptorOld:
    """Descriptor-class for ordered lists of kpoint/spin combinations

    TODO
    """ #XXX

    def __init__(self, nspins, nibzkpts, comm=None, gamma=True, dtype=float):
        """Construct descriptor object for kpoint/spin combinations (ks-pair).

        Parameters:

        nspins: int
            Number of spins.
        nibzkpts: int
            Number of irreducible kpoints in 1st Brillouin zone.
        comm: MPI-communicator
            Communicator for kpoint-groups.
        gamma: bool
            More to follow.
        dtype: NumPy dtype
            More to follow.

        Note that if comm.size is greater than the number of spins, then
        the kpoints cannot all be located at the gamma point and therefor
        the gamma boolean loses its significance.

        Attributes:

        ============  ======================================================
        ``nspins``    Number of spins.
        ``nibzkpts``  Number of irreducible kpoints in 1st Brillouin zone.
        ``nks``       Number of k-point/spin combinations in total.
        ``mynks``     Number of k-point/spin combinations on this CPU.
        ``gamma``     Boolean indicator for gamma point calculation.
        ``dtype``     Data type appropriate for wave functions.
        ``beg``       Beginning of ks-pair indices in group (inclusive).
        ``end``       End of ks-pair indices in group (exclusive).
        ``step``      Stride for ks-pair indices between ``beg`` and ``end``.
        ``comm``      MPI-communicator for kpoint distribution.
        ============  ======================================================
        """
        
        if comm is None:
            comm = mpi.serial_comm
        self.comm = comm
        self.rank = self.comm.rank

        self.nspins = nspins
        self.nibzkpts = nibzkpts
        self.nks = self.nibzkpts * self.nspins

        # XXX Check from distribute_cpus in mpi/__init__.py line 239 rev. 4187
        if self.nks % self.comm.size != 0:
            raise RuntimeError('Cannot distribute %d k-point/spin ' \
                               'combinations to %d processors' % \
                               (self.nks, self.comm.size))

        self.mynks = self.nks // self.comm.size

        # TODO Move code from PAW.initialize in paw.py lines 319-328 rev. 4187
        self.gamma = gamma
        self.dtype = dtype

        uslice = self.get_slice()
        self.beg, self.end, self.step = uslice.indices(self.nks)

    #XXX u is global kpoint index

    def __len__(self):
        return self.mynks

    def get_rank_and_index(self, s, k):
        """Find rank and local index of k-point/spin combination."""
        u = self.where_is(s, k)
        rank, myu = self.who_has(u)
        return rank, myu

    def get_slice(self, rank=None):
        """Return the slice of global ks-pairs which belong to a given rank."""
        if rank is None:
            rank = self.comm.rank
        assert rank in xrange(self.comm.size)
        ks0 = rank * self.mynks
        uslice = slice(ks0, ks0 + self.mynks)
        return uslice

    def get_indices(self, rank=None):
        """Return the global ks-pair indices which belong to a given rank."""
        uslice = self.get_slice(rank)
        return np.arange(*uslice.indices(self.nks))

    def get_ranks(self):
        """Return array of ranks as a function of global ks-pair indices."""
        ranks = np.empty(self.nks, dtype=int)
        for rank in range(self.comm.size):
            uslice = self.get_slice(rank)
            ranks[uslice] = rank
        assert (ranks >= 0).all() and (ranks < self.comm.size).all()
        return ranks

    def who_has(self, u):
        """Convert global index to rank information and local index."""
        rank, myu = divmod(u, self.mynks)
        return rank, myu

    def global_index(self, myu, rank=None):
        """Convert rank information and local index to global index."""
        if rank is None:
            rank = self.comm.rank
        u = rank * self.mynks + myu
        return u

    def what_is(self, u):
        """Convert global index to corresponding kpoint/spin combination."""
        s, k = divmod(u, self.nibzkpts)
        return s, k

    def where_is(self, s, k):
        """Convert kpoint/spin combination to the global index thereof."""
        u = k + self.nibzkpts * s
        return u

    #def get_size_of_global_array(self):
    #    return (self.nspins*self.nibzkpts,)
    #
    #def ...


