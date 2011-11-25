"""Module for calculating the electron-phonon coupling in an LCAO basis.

Electron-phonon coupling::
   
                  __              
                  \     l   +         +
        H      =   )   g   c   c   ( a   + a  ),
         el-ph    /_    ij  i   j     l     l
                 l,ij            
    
where the electron phonon coupling matrix is given by::

                      ______
             l       / hbar         ___
            g   =   /-------  < i | \ /  V   * e  | j > .
             ij   \/ 2 M w           'u   eff   l
                          l

Here, l denotes the vibrational normal mode, w_l and e_l is the frequency and
mass-scaled polarization vector, respectively, M is the effective mass and i, j
atomic orbitals. The coupling can be calculated for both finite and periodic
systems.

In PAW the matrix elements of the derivative of the potential is given by the
sum of the following contributions:: 
        
                  d                  d  ~
            < i | -- v | j > = < i | -- v | j>
                  dP                 dP

                               _
                              \        ~a     d   .       ~a
                            +  ) < i | p  >   -- /_\H   < p | j >
                              /_        i     dP     ij    j
                              a,ij

                               _
                              \        d  ~a     .        ~a
                            +  ) < i | -- p  >  /_\H    < p | j >
                              /_       dP  i        ij     j
                              a,ij

                               _
                              \        ~a     .        d  ~a
                            +  ) < i | p  >  /_\H    < -- p  | j >
                              /_        i        ij    dP  j
                              a,ij
                              
"""

import sys
import cPickle as pickle
from math import pi
from os.path import isfile

import numpy as np

import ase.units as units
from ase.phonons import Displacement
from ase.parallel import rank, barrier

from gpaw.utilities import unpack2
from gpaw.utilities.tools import tri2full
from gpaw.utilities.timing import StepTimer, nulltimer, Timer
from gpaw.lcao.overlap import ManySiteDictionaryWrapper, \
     TwoCenterIntegralCalculator
from gpaw.lcao.tightbinding import TightBinding
from gpaw.kpt_descriptor import KPointDescriptor


class ElectronPhononCoupling: # (Displacement):
    """Class for calculating the electron-phonon coupling in an LCAO basis.

    The derivative of the effective potential wrt atomic displacements is
    obtained from a finite difference approximation to the derivative by doing
    a self-consistent calculation for atomic displacements in the +/-
    directions. These calculations are carried out in the ``run`` member
    function.

    The subsequent calculation of the coupling matrix in the basis of atomic
    orbitals (or Bloch-sums hereof for periodic systems) is handled by the
    ``calculate_matrix`` member function. 
   
    """

    def __init__(self, atoms, calc=None, supercell=(1, 1, 1), name='elph',
                 delta=0.01):
        """Init with an instance of class ``Atoms`` and a calculator.

        Parameters
        ----------
        atoms: Atoms object
            The atoms to work on.
        calc: Calculator
            Calculator for the supercell calculation.
        supercell: tuple
            Size of supercell given by the number of repetitions (l, m, n) of
            the small unit cell in each direction.
        name: str
            Name to use for files.
        delta: float
            Magnitude of displacements.

        """

        # Store atoms and calculator
        self.atoms = atoms
        self.calc = calc
        
        # Displace all atoms in the unit cell by default
        self.indices = range(len(atoms))
        self.name = name
        self.delta = delta
        self.N_c = supercell
        # Center cell offset
        N_c = self.N_c
        self.offset = N_c[0] // 2 * (N_c[1] * N_c[2]) + N_c[1] // 2 * N_c[2] \
                      + N_c[2] // 2
        # Log
        self.set_log()

        # LCAO calculator
        self.calc_lcao = None
        # Supercell matrix
        self.g_xNNMM = None
        
    def __call__(self, atoms_N):
        """Extract effective potential and projector coefficients."""

        # Do calculation
        atoms_N.get_potential_energy()
        # Get calculator
        calc = atoms_N.get_calculator()

        # Effective potential (in Hartree) and projector coefficients
        Vt_G = calc.hamiltonian.vt_sG[0]
        Vt_G = calc.wfs.gd.collect(Vt_G, broadcast=True)
        dH_asp = calc.hamiltonian.dH_asp

        setups = calc.wfs.setups
        nspins = calc.wfs.nspins
        gd_comm = calc.wfs.gd.comm
                
        dH_all_asp = {}
        for a, setup in enumerate(setups):
            ni = setup.ni
            nii = ni * (ni + 1) // 2
            dH_tmp_sp = np.zeros((nspins, nii))
            if a in dH_asp:
                dH_tmp_sp[:] = dH_asp[a]
            gd_comm.sum(dH_tmp_sp)
            dH_all_asp[a] = dH_tmp_sp

        return Vt_G, dH_all_asp

    def run(self):
        """Run the calculations for the required displacements.

        This will do a calculation for 6 displacements per atom, +-x, +-y, and
        +-z. Only those calculations that are not already done will be
        started. Be aware that an interrupted calculation may produce an empty
        file (ending with .pckl), which must be deleted before restarting the
        job. Otherwise the calculation for that displacement will not be done.

        """

        # Atoms in the supercell -- repeated in the lattice vector directions
        # beginning with the last
        atoms_N = self.atoms * self.N_c
        
        # Set calculator if provided
        assert self.calc is not None, "Provide calculator in __init__ method"
        atoms_N.set_calculator(self.calc)
        
        # Do calculation on equilibrium structure
        filename = self.name + '.eq.pckl'
        
        if not isfile(filename):
            # Wait for all ranks to enter
            barrier()
            # Create file
            if rank == 0:
                fd = open(filename, 'w')
                fd.close()

            # Call __call__
            output = self.__call__(atoms_N)
            # Write output to file
            if rank == 0:
                fd = open(filename, 'w')
                pickle.dump(output, fd)
                sys.stdout.write('Writing %s\n' % filename)
                fd.close()
            sys.stdout.flush()

        # Positions of atoms to be displaced in the center unit cell
        N_atoms = len(self.atoms)
        offset = N_atoms * self.offset
        pos = atoms_N.positions[offset: offset + N_atoms].copy()
        
        # Loop over all displacements
        for a in self.indices:
            for i in range(3):
                for sign in [-1, 1]:
                    # Filename for atomic displacement
                    filename = '%s.%d%s%s.pckl' % \
                               (self.name, a, 'xyz'[i], ' +-'[sign])
                    # Wait for ranks before checking for file
                    # barrier()                    
                    if isfile(filename):
                        # Skip if already done
                        continue
                    # Wait for ranks
                    barrier()
                    if rank == 0:
                        fd = open(filename, 'w')
                        fd.close()

                    # Update atomic positions
                    atoms_N.positions[offset + a, i] = \
                        pos[a, i] + sign * self.delta

                    # Call __call__
                    output = self.__call__(atoms_N)
                    # Write output to file    
                    if rank == 0:
                        fd = open(filename, 'w')
                        pickle.dump(output, fd)
                        sys.stdout.write('Writing %s\n' % filename)
                        fd.close()
                    sys.stdout.flush()
                    # Return to initial positions
                    atoms_N.positions[offset + a, i] = pos[a, i]
                    
    def set_lcao_calculator(self, calc):
        """Set LCAO calculator to be used in the matrix element evaluation."""

        # Add parameter checks here
        # - check that gamma
        # - check that no symmetries are used
        # - ...
        parameters = calc.input_parameters
        assert parameters['mode'] == 'lcao', "LCAO mode required."
        assert parameters['usesymm'] != True, "Symmetries not supported."
            
        self.calc_lcao = calc

    def set_log(self, log=None):
        """Set output log."""

        if log is None:
            self.timer = nulltimer
        elif log == '-':
            self.timer = StepTimer(name='EPCM')
        else:
            self.timer = StepTimer(name='EPCM', out=open(log, 'w'))

    def lattice_vectors(self):
        """Return lattice vectors for cell in the supercell."""

        # Lattice vectors
        R_cN = np.indices(self.N_c).reshape(3, -1)
        N_c = np.array(self.N_c)[:, np.newaxis]
        # R_cN += N_c // 2
        # R_cN %= N_c
        R_cN -= N_c // 2

        return R_cN
    
    def calculate_supercell_matrix(self, dump=0, name=None, cutoff=None, atoms=None):
        """Calculate matrix elements of the el-ph coupling in the LCAO basis.

        This function calculates the matrix elements between LCAOs and local
        atomic gradients of the effective potential. The matrix elements are
        calculated for the supercell used to obtain finite-difference
        approximations to the derivatives of the effective potential wrt to
        atomic displacements.

        Parameters
        ----------
        dump: int
            Dump supercell matrix to pickle file (default: 0).
            0: Supercell matrix not saved
            1: Supercell matrix saved in a single pickle file.
            2: Dump matrix for different gradients in separate files. Useful
               for large systems where the total array gets too large.
        name: string
            User specified name of the generated pickle file(s). If not
            provided, the string in the ``name`` attribute is used.
        cutoff: float
            Apply specified cutoff (default: None).
        atoms: Atoms object
            Calculate supercell for atoms different from the ones provided in
            the ```__init__`` method.
            
        """

        assert self.calc_lcao is not None, "Set LCAO calculator"
            
        # Supercell atoms
        if atoms is None:
            atoms_N = self.atoms * self.N_c
        else:
            atoms_N = atoms
        
        # Initialize calculator if required and extract useful quantities
        calc = self.calc_lcao
        if not hasattr(calc.wfs, 'S_qMM'):
            calc.initialize(atoms_N)
            calc.initialize_positions(atoms_N)

        # Extract useful objects from the calculator
        wfs = calc.wfs
        gd = calc.wfs.gd
        kd = calc.wfs.kd
        kpt_u = wfs.kpt_u
        setups = wfs.setups
        nao = setups.nao
        bfs = wfs.basis_functions
        dtype = wfs.dtype
        spin = 0 # XXX

        # Basis info for atoms in reference cell
        basis = self.calc_lcao.input_parameters['basis']        
        niAO_a = [setups[a].niAO for a in range(len(self.atoms))]
        M_a = [bfs.M_a[a] for a in range(len(self.atoms))]
        
        # If gamma calculation, overlap with neighboring cell cannot be removed
        if kd.gamma:
            print "WARNING: Gamma-point calculation."
        else:
            # Bloch to real-space converter
            tb = TightBinding(atoms_N, calc)

        self.timer.write_now("Calculating supercell matrix")

        self.timer.write_now("Calculating real-space gradients")        
        # Calculate finite-difference gradients (in Hartree / Bohr)
        V1t_xG, dH1_xasp = self.calculate_gradient()
        self.timer.write_now("Finished real-space gradients")
        
        # For the contribution from the derivative of the projectors
        dP_aqvMi = self.calculate_dP_aqvMi(wfs)
        # Equilibrium atomic Hamiltonian matrix (projector coefficients)
        dH_asp = pickle.load(open(self.name + '.eq.pckl'))[1]
        
        # Check that the grid is the same as in the calculator
        assert np.all(V1t_xG.shape[-3:] == (gd.N_c + gd.pbc_c - 1)), \
               "Mismatch in grids."

        # Calculate < i k | grad H | j k >, i.e. matrix elements in Bloch basis
        # List for supercell matrices;
        g_xNNMM = []
        self.timer.write_now("Calculating gradient of PAW Hamiltonian")

        # Do each cartesian component separately
        for x, V1t_G in enumerate(V1t_xG):

            # Corresponding atomic and cartesian indices
            a = x // 3
            v = x % 3
            self.timer.write_now("%s-gradient of atom %u" % 
                                 (['x','y','z'][v], a))

            # Array for different k-point components
            g_qMM = np.zeros((len(kpt_u), nao, nao), dtype)
            
            # 1) Gradient of effective potential
            self.timer.write_now("Starting gradient of pseudo part")
            for kpt in kpt_u:
                # Matrix elements
                geff_MM = np.zeros((nao, nao), dtype)
                bfs.calculate_potential_matrix(V1t_G, geff_MM, q=kpt.q)
                tri2full(geff_MM, 'L')
                # Insert in array
                g_qMM[kpt.q] += geff_MM

            self.timer.write_now("Finished gradient of pseudo part")
    
            # 2) Gradient of non-local part (projectors)
            self.timer.write_now("Starting gradient of dH^a part")
            P_aqMi = calc.wfs.P_aqMi
            # 2a) dH^a part has contributions from all other atoms
            for kpt in kpt_u:
                # Matrix elements
                gp_MM = np.zeros((nao, nao), dtype)
                dH1_asp = dH1_xasp[x]
                for a_, dH1_sp in dH1_asp.items():
                    dH1_ii = unpack2(dH1_sp[spin])
                    gp_MM += np.dot(P_aqMi[a_][kpt.q], np.dot(dH1_ii,
                                    P_aqMi[a_][kpt.q].T.conjugate()))
                g_qMM[kpt.q] += gp_MM
            self.timer.write_now("Finished gradient of dH^a part")
            
            self.timer.write_now("Starting gradient of projectors part")
            # 2b) dP^a part has only contributions from the same atoms
            dP_qvMi = dP_aqvMi[a]
            dH_ii = unpack2(dH_asp[a][spin])
            for kpt in kpt_u:
                #XXX Sort out the sign here; conclusion -> sign = +1 !
                P1HP_MM = +1 * np.dot(dP_qvMi[kpt.q][v], np.dot(dH_ii,
                                      P_aqMi[a][kpt.q].T.conjugate()))
                # Matrix elements
                gp_MM = P1HP_MM + P1HP_MM.T.conjugate()
                g_qMM[kpt.q] += gp_MM
            self.timer.write_now("Finished gradient of projectors part")
            
            # Extract R_c=(0, 0, 0) block by Fourier transforming
            if kd.gamma or kd.N_c is None:
                g_MM = g_qMM[0]
            else:
                # Convert to array
                g_MM = tb.bloch_to_real_space(g_qMM, R_c=(0, 0, 0))[0]

            # Reshape to global unit cell indices
            N = np.prod(self.N_c)
            # Number of basis function in the primitive cell
            assert (nao % N) == 0, "Alarm ...!"
            nao_cell = nao / N
            g_NMNM = g_MM.reshape((N, nao_cell, N, nao_cell))
            g_NNMM = g_NMNM.swapaxes(1, 2).copy()
            self.timer.write_now("Finished supercell matrix")

            if dump != 2:
                g_xNNMM.append(g_NNMM)
            else:
                if name is not None:
                    fname = '%s.supercell_matrix_x_%2.2u.%s.pckl' % (name, x, basis)
                else:
                    fname = self.name + \
                            '.supercell_matrix_x_%2.2u.%s.pckl' % (x, basis)
                if kd.comm.rank == 0:
                    fd = open(fname, 'w')
                    pickle.dump((g_NNMM, M_a, niAO_a), fd, 2)
                    fd.close()
                    
        self.timer.write_now("Finished gradient of PAW Hamiltonian")
        
        if dump != 2:
            # Collect gradients in one array
            self.g_xNNMM = np.array(g_xNNMM)
            # Apply cutoff
            if cutoff is not None:
                cutoff = float(cutoff)
                self.apply_cutoff(self.g_xNNMM, M_a, niAO_a, cutoff)
            
            # Dump to pickle file using binary mode together with basis info
            if dump and kd.comm.rank == 0:
                if name is not None:
                    fname = '%s.pckl' % name
                else:
                    fname = self.name + '.supercell_matrix.%s.pckl' % basis
                fd = open(fname, 'w')                
                pickle.dump((self.g_xNNMM, M_a, niAO_a), fd, 2)
                fd.close()

    def load_supercell_matrix(self, basis=None, name=None, multiple=False,
                              cutoff=None):
        """Load supercell matrix from pickle file.

        Parameters
        ----------
        basis: string
            String specifying the LCAO basis used to calculate the supercell
            matrix, e.g. dz(dzp).
        name: string
            User specified name of the pickle file.
        multiple: bool
            Load each derivative from individual files.
        cutoff: float
            Zero matrix elements from basis functions located further away from
            the atomic gradient than the cutoff.

        """

        assert (basis is not None) or (name is not None), \
               "Provide basis or name."
        
        if not multiple:
            # File name
            if name is not None:
                fname = name
            else:
                fname = self.name + '.supercell_matrix.%s.pckl' % basis
            fd = open(fname)
            self.g_xNNMM, M_a, niAO_a = pickle.load(fd)
            fd.close()
        else:
            g_xNNMM = []
            for x in range(len(self.indices)*3):
                if name is not None:
                    fname = name
                else:
                    fname = self.name + \
                            '.supercell_matrix_x_%2.2u.%s.pckl' % (x, basis)
                fd = open(fname, 'r')
                g_NNMM, M_a, niAO_a = pickle.load(fd)
                fd.close()
                g_xNNMM.append(g_NNMM)
            self.g_xNNMM = np.array(g_xNNMM)
        
        if cutoff is not None:
            cutoff = float(cutoff)
            self.apply_cutoff(self.g_xNNMM, M_a, niAO_a, cutoff)
                                  
    def apply_cutoff(self, g_xNNMM, M_a, niAO_a, r_c):
        """Zero matrix element with basis functions beyond the cutoff."""

        # Number of atoms and primitive cells
        N_atoms = len(self.indices)
        N = np.prod(self.N_c)
        nao = g_xNNMM.shape[-1]
        
        # Reshape array
        g_avNNMM = g_xNNMM.reshape(N_atoms, 3, N, N, nao, nao)
        
        # Make slices for orbitals on atoms
        slice_a = []
        for a in range(len(self.atoms)):
            start = M_a[a] ;
            stop = start + niAO_a[a]
            s = slice(start, stop)
            slice_a.append(s)
            
        # Lattice vectors
        R_cN = self.lattice_vectors()

        # Unit cell vectors
        cell_vc = self.atoms.cell.transpose()
        # Atomic positions in reference cell
        pos_av = self.atoms.get_positions()
        
        # Zero elements with a distance to atoms in the reference cell
        # larger than the cutoff
        for n in range(N):
            # Lattice vector to cell
            R_v = np.dot(cell_vc, R_cN[:, n])
            # Atomic positions in cell
            posn_av = pos_av + R_v
            for i, a in enumerate(self.indices):
                dist_a = np.sqrt(np.sum((pos_av[a] - posn_av)**2, axis=-1))
                # Atoms indices where the distance is larger than the cufoff
                j_a = np.where(dist_a > r_c)[0]
                # Zero elements
                for j in j_a:
                    g_avNNMM[a, :, n, :, slice_a[j], :] = 0.0
                    g_avNNMM[a, :, :, n, :, slice_a[j]] = 0.0
        
    def coupling(self, kpts, qpts, c_kn, u_ql, omega_ql=None, kpts_from=None):
        """Calculate el-ph couplings in Bloch basis for the electrons.

        This function calculates the electron-phonon coupling between the
        specified Bloch states, i.e.::

                      ______ 
            mnl      / hbar               ^
           g    =   /-------  < m k + q | e  . grad V  | n k >
            kq    \/ 2 M w                 ql        q
                          ql

        In case the ``omega_ql`` keyword argument is not given, the bare matrix
        element (in units of eV / Ang) without the sqrt prefactor is returned. 
        
        Parameters
        ----------
        kpts: ndarray or tuple.
            k-vectors of the Bloch states. When a tuple of integers is given, a
            Monkhorst-Pack grid with the specified number of k-points along the
            directions of the reciprocal lattice vectors is generated.
        qpts: ndarray or tuple.
            q-vectors of the phonons.
        c_kn: ndarray
            Expansion coefficients for the Bloch states. The ordering must be
            the same as in the ``kpts`` argument.
        u_ql: ndarray
            Mass-scaled polarization vectors (in units of 1 / sqrt(amu)) of the
            phonons. Again, the ordering must be the same as in the
            corresponding ``qpts`` argument.
        omega_ql: ndarray
            Vibrational frequencies in eV. 
        kpts_from: list of ints or int
            Calculate only the matrix element for the k-vectors specified by
            their index in the ``kpts`` argument (default: all).

        In short, phonon frequencies and mode vectors must be given in ase units.

        """

        assert self.g_xNNMM is not None, "Load supercell matrix."
        assert len(c_kn.shape) == 3
        assert len(u_ql.shape) == 4
        if omega_ql is not None:
            assert np.all(u_ql.shape[:2] == omega_ql.shape[:2])
            
        # Use the KPointDescriptor to keep track of the k and q-vectors
        kd_kpts = KPointDescriptor(kpts)
        kd_qpts = KPointDescriptor(qpts)
        # Check that number of k- and q-points agree with the number of Bloch
        # functions and polarization vectors
        assert kd_kpts.nbzkpts == len(c_kn)
        assert kd_qpts.nbzkpts == len(u_ql)
        
        # Include all k-point per default
        if kpts_from is None:
            kpts_kc = kd_kpts.bzk_kc
            kpts_k = range(kd_kpts.nbzkpts)
        else:
            kpts_kc = kd_kpts.bzk_kc[kpts_from]
            if isinstance(kpts_from, int):
                kpts_k = list([kpts_from])
            else:
                kpts_k = list(kpts_from)

        # Supercell matrix (real matrix in Hartree / Bohr)
        g_xNNMM = self.g_xNNMM
        
        # Number of phonon modes and electronic bands
        L = len(u_ql[0])
        M = len(c_kn[0])
        # Lattice vectors
        R_cN = self.lattice_vectors()
        # Number of unit cell in supercell
        N = np.prod(self.N_c)
        
        # Allocate array for couplings
        g_qklnn = np.zeros((kd_qpts.nbzkpts, len(kpts_kc), L, M, M),
                           dtype=complex)
       
        self.timer.write_now("Calculating coupling matrix elements")
        for q, q_c in enumerate(kd_qpts.bzk_kc):

            # Find indices of k+q for the k-points
            kplusq_k = kd_kpts.find_k_plus_q(q_c, kpts_k=kpts_k)

            # Here, ``i`` is counting from 0 and ``k`` is the global index of
            # the k-point 
            for i, (k, k_c) in enumerate(zip(kpts_k, kpts_kc)):

                assert np.allclose(k_c + q_c, kd_kpts.bzk_kc[kplusq_k[i]] ), \
                       (i, k, k_c, q_c, kd_kpts.bzk_kc[kplusq_k[i]])
                # assert np.all(k_c + q_c == kd_kpts.bzk_kc[kplusq_k[i]] )

                # LCAO coefficient for Bloch states
                ck_nM = c_kn[k]
                ckplusq_nM = c_kn[kplusq_k[i]]
                g_nnNNx = np.dot(ckplusq_nM.conj(),
                                 np.dot(g_xNNMM, ck_nM.T)).swapaxes(1, 4) # .copy()

                # Mass scaled polarization vector
                u_lx = u_ql[q].reshape(L, 3 * len(self.atoms))
                g_lnnNN = np.dot(g_nnNNx, u_lx.T).transpose(4, 0, 1, 2, 3)
                
                # Generate phase factors
                phase_NN = np.zeros((N, N), dtype=complex)
                for m in range(N):
                    for n in range(N):
                        Rm_c = R_cN[:, m]
                        Rn_c = R_cN[:, n]
                        phase_NN[m, n] = np.exp(2.j * pi * (
                            np.dot(k_c, Rm_c - Rn_c) + np.dot(q_c, Rm_c)))

                # Multiply phases and sum over unit cells
                g_lnn = np.sum(np.sum(g_lnnNN * phase_NN, axis=-1), axis=-1)
                g_qklnn[q, i] = g_lnn

                # XXX Temp
                if np.all(q_c == 0.0):
                    # These should be real
                    print g_qklnn[q].imag.min(), g_qklnn[q].imag.max()
                    
        self.timer.write_now("Finished calculation of coupling matrix elements")
                                                
        # Return the bare matrix element if frequencies are not given
        if omega_ql is None:
            # Convert to eV / Ang
            g_qklnn *= units.Hartree / units.Bohr
        else:
            # Multiply prefactor sqrt(hbar / 2 * M * omega) in units of Bohr
            amu = units._amu # atomic mass unit
            me = units._me   # electron mass
            g_qklnn /= np.sqrt(2 * amu / me / units.Hartree * \
                           omega_ql[:, np.newaxis, :, np.newaxis, np.newaxis])
            # Convert to eV
            g_qklnn *= units.Hartree
            
        # Return couplings in eV (or eV / Ang)
        return g_qklnn
        
    def calculate_gradient(self):
        """Calculate gradient of effective potential and projector coefs.

        This function loads the generated pickle files and calculates
        finite-difference derivatives.

        """

        # Array and dict for finite difference derivatives
        V1t_xG = []
        dH1_xasp = []

        x = 0
        for a in self.indices:
            for v in range(3):
                name = '%s.%d%s' % (self.name, a, 'xyz'[v])
                # Potential and atomic density matrix for atomic displacement
                try:
                    Vtm_G, dHm_asp = pickle.load(open(name + '-.pckl'))
                    Vtp_G, dHp_asp = pickle.load(open(name + '+.pckl'))
                except (IOError, EOFError):
                    raise IOError, "%s(-/+).pckl" % name
                
                # FD derivatives in Hartree / Bohr
                V1t_G = (Vtp_G - Vtm_G) / (2 * self.delta / units.Bohr)
                V1t_xG.append(V1t_G)

                dH1_asp = {}
                for atom in dHm_asp.keys():
                    dH1_asp[atom] = (dHp_asp[atom] - dHm_asp[atom]) / \
                                    (2 * self.delta / units.Bohr)
                dH1_xasp.append(dH1_asp)
                x += 1
                
        return np.array(V1t_xG), dH1_xasp

    def calculate_dP_aqvMi(self, wfs):
        """Overlap between LCAO basis functions and gradient of projectors.

        Only the gradient wrt the atomic positions in the reference cell is
        computed.

        """
        
        nao = wfs.setups.nao
        nq = len(wfs.ibzk_qc)

        # Derivatives in reference cell
        dP_aqvMi = {}
        for atom, setup in zip(self.atoms, wfs.setups): 
            a = atom.index
            dP_aqvMi[a] = np.zeros((nq, 3, nao, setup.ni), wfs.dtype)

        # Calculate overlap between basis function and gradient of projectors
        # NOTE: the derivative is calculated wrt the atomic position and not
        # the real-space coordinate
        calc = TwoCenterIntegralCalculator(wfs.ibzk_qc, derivative=True)
        expansions = ManySiteDictionaryWrapper(wfs.tci.P_expansions, dP_aqvMi)
        calc.calculate(wfs.tci.atompairs, [expansions], [dP_aqvMi])

        # Extract derivatives in the reference unit cell
        # dP_aqvMi = {}
        # for atom in self.atoms:
        #     dP_aqvMi[atom.index] = dPall_aqvMi[atom.index]
            
        return dP_aqvMi
