# -*- coding: utf-8 -*-
# Copyright (C) 2003-2007  CAMP
# Copyright (C) 2007-2008  CAMd
# Please see the accompanying LICENSE file for further information.

"""This module defines a PAW-class.

The central object that glues everything together!"""

import numpy as np
from ase.units import Bohr, Hartree
from ase.dft import monkhorst_pack

import gpaw.io
import gpaw.mpi as mpi
import gpaw.occupations as occupations
from gpaw import dry_run, memory_estimate_depth, KohnShamConvergenceError
from gpaw.hooks import hooks
from gpaw.density import Density
from gpaw.eigensolvers import get_eigensolver
from gpaw.band_descriptor import BandDescriptor
from gpaw.grid_descriptor import GridDescriptor
from gpaw.kohnsham_layouts import get_KohnSham_layouts
from gpaw.hamiltonian import Hamiltonian
from gpaw.utilities.timing import Timer
from gpaw.xc import XC
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.wavefunctions.base import EmptyWaveFunctions
from gpaw.wavefunctions.fd import FDWaveFunctions
from gpaw.wavefunctions.lcao import LCAOWaveFunctions
from gpaw.wavefunctions.pw import PW
from gpaw.utilities.memory import MemNode, maxrss
from gpaw.parameters import InputParameters
from gpaw.setup import Setups
from gpaw.output import PAWTextOutput
from gpaw.scf import SCFLoop
from gpaw.forces import ForceCalculator
from gpaw.utilities import h2gpts


class PAW(PAWTextOutput):
    """This is the main calculation object for doing a PAW calculation."""

    timer_class = Timer

    def __init__(self, filename=None, **kwargs):
        """ASE-calculator interface.

        The following parameters can be used: `nbands`, `xc`, `kpts`,
        `spinpol`, `gpts`, `h`, `charge`, `usesymm`, `width`, `mixer`,
        `hund`, `lmax`, `fixdensity`, `convergence`, `txt`, `parallel`,
        `softgauss` and `stencils`.

        If you don't specify any parameters, you will get:

        Defaults: neutrally charged, LDA, gamma-point calculation, a
        reasonable grid-spacing, zero Kelvin electronic temperature,
        and the number of bands will be equal to the number of atomic
        orbitals present in the setups. Only occupied bands are used
        in the convergence decision. The calculation will be
        spin-polarized if and only if one or more of the atoms have
        non-zero magnetic moments. Text output will be written to
        standard output.

        For a non-gamma point calculation, the electronic temperature
        will be 0.1 eV (energies are extrapolated to zero Kelvin) and
        all symmetries will be used to reduce the number of
        **k**-points."""

        PAWTextOutput.__init__(self)
        self.grid_descriptor_class = GridDescriptor
        self.input_parameters = InputParameters()
        self.timer = self.timer_class()

        self.scf = None
        self.forces = ForceCalculator(self.timer)
        self.wfs = EmptyWaveFunctions()
        self.occupations = None
        self.density = None
        self.hamiltonian = None
        self.atoms = None

        self.initialized = False

        # Possibly read GPAW keyword arguments from file:
        if filename is not None and filename.endswith('.gkw'):
            from gpaw.utilities.kwargs import load
            parameters = load(filename)
            parameters.update(kwargs)
            kwargs = parameters
            filename = None  # XXX

        if filename is not None:
            comm = kwargs.get('communicator', mpi.world)
            reader = gpaw.io.open(filename, 'r', comm)
            self.atoms = gpaw.io.read_atoms(reader)
            par = self.input_parameters
            par.read(reader)

        self.set(**kwargs)

        if filename is not None:
            # Setups are not saved in the file if the setups were not loaded
            # *from* files in the first place
            if par.setups is None:
                if par.idiotproof:
                    raise RuntimeError('Setups not specified in file. Use '
                                       'idiotproof=False to proceed anyway.')
                else:
                    par.setups = {None: 'paw'}
            if par.basis is None:
                if par.idiotproof:
                    raise RuntimeError('Basis not specified in file. Use '
                                       'idiotproof=False to proceed anyway.')
                else:
                    par.basis = {}

            self.initialize()
            self.read(reader)

            self.print_cell_and_parameters()

        self.observers = []

    def read(self, reader):
        gpaw.io.read(self, reader)

    def set(self, **kwargs):
        p = self.input_parameters

        # Prune input for things that didn't change
        for key, value in kwargs.items():
            if key == 'kpts':
                oldbzk_kc = kpts2ndarray(p.kpts)
                newbzk_kc = kpts2ndarray(value)
                if (len(oldbzk_kc) == len(newbzk_kc) and
                    (oldbzk_kc == newbzk_kc).all()):
                    kwargs.pop('kpts')
            elif np.all(p[key] == value):
                kwargs.pop(key)

        if (kwargs.get('h') is not None) and (kwargs.get('gpts') is not None):
            raise TypeError("""You can't use both "gpts" and "h"!""")
        if 'h' in kwargs:
            p['gpts'] = None
        if 'gpts' in kwargs:
            p['h'] = None

        # Special treatment for dictionary parameters:
        for name in ['convergence', 'parallel']:
            if kwargs.get(name) is not None:
                tmp = p[name]
                for key in kwargs[name]:
                    if not key in tmp:
                        raise KeyError('Unknown subparameter "%s" in '
                                       'dictionary parameter "%s"' % (key,
                                                                      name))
                tmp.update(kwargs[name])
                kwargs[name] = tmp

        self.initialized = False

        for key in kwargs:
            if key == 'basis' and  p['mode'] == 'fd':
                continue

            if key == 'eigensolver':
                self.wfs.set_eigensolver(None)
            
            if key in ['fixmom', 'mixer',
                       'verbose', 'txt', 'hund', 'random',
                       'eigensolver', 'poissonsolver', 'idiotproof', 'notify']:
                continue

            if key in ['convergence', 'fixdensity', 'maxiter']:
                self.scf = None
                continue

            # More drastic changes:
            self.scf = None
            self.wfs.set_orthonormalized(False)
            if key in ['lmax', 'width', 'stencils', 'external', 'xc',
                       'occupations']:
                self.hamiltonian = None
                self.occupations = None
            elif key in ['charge']:
                self.hamiltonian = None
                self.density = None
                self.wfs = EmptyWaveFunctions()
                self.occupations = None
            elif key in ['kpts', 'nbands']:
                self.wfs = EmptyWaveFunctions()
                self.occupations = None
            elif key in ['h', 'gpts', 'setups', 'spinpol', 'usesymm',
                         'parallel', 'communicator', 'dtype']:
                self.density = None
                self.occupations = None
                self.hamiltonian = None
                self.wfs = EmptyWaveFunctions()
            elif key in ['mode', 'basis']:
                self.wfs = EmptyWaveFunctions()
            elif key in ['parsize', 'parsize_bands', 'parstride_bands']:
                name = {'parsize': 'domain',
                        'parsize_bands': 'band',
                        'parstride_bands': 'stridebands'}[key]
                raise DeprecationWarning(
                    'Keyword argument has been moved ' +
                    "to the 'parallel' dictionary keyword under '%s'." % name)
            else:
                raise TypeError("Unknown keyword argument: '%s'" % key)

        p.update(kwargs)

    def calculate(self, atoms=None, converge=False,
                  force_call_to_set_positions=False):
        """Update PAW calculaton if needed."""

        self.timer.start('Initialization')
        if atoms is None:
            atoms = self.atoms

        if self.atoms is None:
            # First time:
            self.initialize(atoms)
            self.set_positions(atoms)
        elif (len(atoms) != len(self.atoms) or
              (atoms.get_atomic_numbers() !=
               self.atoms.get_atomic_numbers()).any() or
              (atoms.get_initial_magnetic_moments() !=
               self.atoms.get_initial_magnetic_moments()).any() or
              (atoms.get_cell() != self.atoms.get_cell()).any() or
              (atoms.get_pbc() != self.atoms.get_pbc()).any()):
            # Drastic changes:
            self.wfs = EmptyWaveFunctions()
            self.occupations = None
            self.density = None
            self.hamiltonian = None
            self.scf = None
            self.initialize(atoms)
            self.set_positions(atoms)
        elif not self.initialized:
            self.initialize(atoms)
            self.set_positions(atoms)
        elif (atoms.get_positions() != self.atoms.get_positions()).any():
            self.density.reset()
            self.set_positions(atoms)
        elif not self.scf.converged:
            # Do not call scf.check_convergence() here as it overwrites
            # scf.converged, and setting scf.converged is the only
            # 'practical' way for a user to force the calculation to proceed
            self.set_positions(atoms)
        elif force_call_to_set_positions:
            self.set_positions(atoms)

        self.timer.stop('Initialization')

        if self.scf.converged:
            return
        else:
            self.print_cell_and_parameters()

        self.timer.start('SCF-cycle')
        for iter in self.scf.run(self.wfs, self.hamiltonian, self.density,
                                 self.occupations):
            self.call_observers(iter)
            self.print_iteration(iter)
            self.iter = iter
        self.timer.stop('SCF-cycle')

        if self.scf.converged:
            self.call_observers(iter, final=True)
            self.print_converged(iter)
            if 'converged' in hooks:
                hooks['converged'](self)
        elif converge:
            if 'not_converged' in hooks:
                hooks['not_converged'](self)
            raise KohnShamConvergenceError('Did not converge!')

    def initialize_positions(self, atoms=None):
        """Update the positions of the atoms."""
        if atoms is None:
            atoms = self.atoms
        else:
            # Save the state of the atoms:
            self.atoms = atoms.copy()

        self.check_atoms()

        spos_ac = atoms.get_scaled_positions() % 1.0

        self.wfs.set_positions(spos_ac)
        self.density.set_positions(spos_ac, self.wfs.rank_a)
        self.hamiltonian.set_positions(spos_ac, self.wfs.rank_a)

        return spos_ac

    def set_positions(self, atoms=None):
        """Update the positions of the atoms and initialize wave functions."""
        spos_ac = self.initialize_positions(atoms)
        self.wfs.initialize(self.density, self.hamiltonian, spos_ac)
        self.scf.reset()
        self.forces.reset()
        self.print_positions()

    def initialize(self, atoms=None):
        """Inexpensive initialization."""

        if atoms is None:
            atoms = self.atoms
        else:
            # Save the state of the atoms:
            self.atoms = atoms.copy()

        par = self.input_parameters

        world = par.communicator
        if world is None:
            world = mpi.world
        elif hasattr(world, 'new_communicator'):
            # Check for whether object has correct type already
            #
            # Using isinstance() is complicated because of all the
            # combinations, serial/parallel/debug...
            pass
        else:
            # world should be a list of ranks:
            world = mpi.world.new_communicator(np.asarray(world))
        self.wfs.world = world

        self.set_text(par.txt, par.verbose)

        natoms = len(atoms)

        pos_av = atoms.get_positions() / Bohr
        cell_cv = atoms.get_cell()
        pbc_c = atoms.get_pbc()
        Z_a = atoms.get_atomic_numbers()
        magmom_av = atoms.get_initial_magnetic_moments()

        if isinstance(par.xc, str):
            xc = XC(par.xc)
        else:
            xc = par.xc

        setups = Setups(Z_a, par.setups, par.basis, par.lmax, xc, world)

        if magmom_av.ndim == 1:
            collinear = True
            magmom_av, magmom_a = np.zeros((natoms, 3)), magmom_av
            magmom_av[:, 2] = magmom_a
        else:
            collinear = False
            
        magnetic = magmom_av.any()

        spinpol = par.spinpol
        if par.hund:
            if natoms != 1:
                raise ValueError('hund=True arg only valid for single atoms!')
            spinpol = True
            magmom_av[0] = (0, 0, setups[0].get_hunds_rule_moment(par.charge))
            
        if spinpol is None:
            spinpol = magnetic
        elif magnetic and not spinpol:
            raise ValueError('Non-zero initial magnetic moment for a ' +
                             'spin-paired calculation!')

        if collinear:
            nspins = 1 + int(spinpol)
            ncomp = 1
        else:
            nspins = 1
            ncomp = 2

        # K-point descriptor
        kd = KPointDescriptor(par.kpts, nspins, collinear)

        width = par.width
        if width is None:
            if kd.gamma:
                width = 0.0
            else:
                width = 0.1  # eV
        else:
            assert par.occupations is None
      
        if par.gpts is not None and par.h is None:
            N_c = np.array(par.gpts)
        else:
            if par.h is None:
                self.text('Using default value for grid spacing.')
                h = 0.2
            else:
                h = par.h
            N_c = h2gpts(h, cell_cv)

        cell_cv /= Bohr

        if hasattr(self, 'time') or par.dtype == complex:
            dtype = complex
        else:
            if kd.gamma:
                dtype = float
            else:
                dtype = complex

        kd.set_symmetry(atoms, setups, magmom_av, par.usesymm, N_c)

        nao = setups.nao
        nvalence = setups.nvalence - par.charge
        M_v = magmom_av.sum(0)
        M = np.dot(M_v, M_v)**0.5
        
        nbands = par.nbands
        if nbands is None:
            nbands = nao
        elif nbands > nao and par.mode == 'lcao':
            raise ValueError('Too many bands for LCAO calculation: ' +
                             '%d bands and only %d atomic orbitals!' %
                             (nbands, nao))

        if nvalence < 0:
            raise ValueError(
                'Charge %f is not possible - not enough valence electrons' %
                par.charge)

        if nbands <= 0:
            nbands = int(nvalence + M + 0.5) // 2 + (-nbands)

        if nvalence > 2 * nbands:
            raise ValueError('Too few bands!  Electrons: %f, bands: %d'
                             % (nvalence, nbands))

        nbands *= ncomp

        if par.width is not None:
            self.text('**NOTE**: please start using '
                      'occupations=FermiDirac(width).')
        if par.fixmom:
            self.text('**NOTE**: please start using '
                      'occupations=FermiDirac(width, fixmagmom=True).')

        if self.occupations is None:
            if par.occupations is None:
                # Create object for occupation numbers:
                self.occupations = occupations.FermiDirac(width, par.fixmom)
            else:
                self.occupations = par.occupations

        self.occupations.magmom = M_v[2]

        cc = par.convergence

        if par.mode == 'lcao':
            niter_fixdensity = 0
        else:
            niter_fixdensity = None

        if self.scf is None:
            self.scf = SCFLoop(
                cc['eigenstates'] * nvalence,
                cc['energy'] / Hartree * max(nvalence, 1),
                cc['density'] * nvalence,
                par.maxiter, par.fixdensity,
                niter_fixdensity)

        parsize, parsize_bands = par.parallel['domain'], par.parallel['band']

        if parsize_bands is None:
            parsize_bands = 1

        # TODO delete/restructure so all checks are in BandDescriptor
        if nbands % parsize_bands != 0:
            raise RuntimeError('Cannot distribute %d bands to %d processors' %
                               (nbands, parsize_bands))

        if not self.wfs:
            if parsize == 'domain only':  # XXX this was silly!
                parsize = world.size

            domain_comm, kpt_comm, band_comm = mpi.distribute_cpus(parsize,
                parsize_bands, nspins, kd.nibzkpts, world, par.idiotproof)

            kd.set_communicator(kpt_comm)

            parstride_bands = par.parallel['stridebands']
            bd = BandDescriptor(nbands, band_comm, parstride_bands)

            if (self.density is not None and
                self.density.gd.comm.size != domain_comm.size):
                # Domain decomposition has changed, so we need to
                # reinitialize density and hamiltonian:
                if par.fixdensity:
                    raise RuntimeError("I'm confused - please specify parsize."
                                       )
                self.density = None
                self.hamiltonian = None

            # Construct grid descriptor for coarse grids for wave functions:
            gd = self.grid_descriptor_class(N_c, cell_cv, pbc_c,
                                            domain_comm, parsize)

            # do k-point analysis here? XXX
            args = (gd, nvalence, setups, bd, dtype, world, kd, self.timer)

            if par.mode == 'lcao':
                # Layouts used for general diagonalizer
                sl_lcao = par.parallel['sl_lcao']
                if sl_lcao is None:
                    sl_lcao = par.parallel['sl_default']
                lcaoksl = get_KohnSham_layouts(sl_lcao, 'lcao',
                                               gd, bd, dtype,
                                               nao=nao, timer=self.timer)

                if collinear:
                    self.wfs = LCAOWaveFunctions(lcaoksl, *args)
                else:
                    from gpaw.xc.noncollinear import \
                         NonCollinearLCAOWaveFunctions
                    self.wfs = NonCollinearLCAOWaveFunctions(lcaoksl, *args)

            elif par.mode == 'fd' or isinstance(par.mode, PW):
                # buffer_size keyword only relevant for fdpw
                buffer_size = par.parallel['buffer_size']
                # Layouts used for diagonalizer
                sl_diagonalize = par.parallel['sl_diagonalize']
                if sl_diagonalize is None:
                    sl_diagonalize = par.parallel['sl_default']
                diagksl = get_KohnSham_layouts(sl_diagonalize, 'fd',
                                               gd, bd, dtype,
                                               buffer_size=buffer_size,
                                               timer=self.timer)

                # Layouts used for orthonormalizer
                sl_inverse_cholesky = par.parallel['sl_inverse_cholesky']
                if sl_inverse_cholesky is None:
                    sl_inverse_cholesky = par.parallel['sl_default']
                if sl_inverse_cholesky != sl_diagonalize:
                    message = 'sl_inverse_cholesky != sl_diagonalize ' \
                        'is not implemented.'
                    raise NotImplementedError(message)
                orthoksl = get_KohnSham_layouts(sl_inverse_cholesky, 'fd',
                                                gd, bd, dtype,
                                                buffer_size=buffer_size,
                                                timer=self.timer)

                # Use (at most) all available LCAO for initialization
                lcaonbands = min(nbands, nao)
                lcaobd = BandDescriptor(lcaonbands, band_comm, parstride_bands)
                assert nbands <= nao or bd.comm.size == 1
                assert lcaobd.mynbands == min(bd.mynbands, nao)  # XXX

                # Layouts used for general diagonalizer (LCAO initialization)
                sl_lcao = par.parallel['sl_lcao']
                if sl_lcao is None:
                    sl_lcao = par.parallel['sl_default']
                initksl = get_KohnSham_layouts(sl_lcao, 'lcao',
                                               gd, lcaobd, dtype,
                                               nao=nao,
                                               timer=self.timer)

                if par.mode == 'fd':
                    self.wfs = FDWaveFunctions(par.stencils[0], diagksl,
                                               orthoksl, initksl, *args)
                else:
                    # Planewave basis:
                    self.wfs = par.mode(diagksl, orthoksl, initksl,
                                        gd, nvalence, setups, bd,
                                        world, kd, self.timer)
            else:
                self.wfs = par.mode(self, *args)
        else:
            self.wfs.set_setups(setups)

        if not self.wfs.eigensolver:
            # Number of bands to converge:
            nbands_converge = cc['bands']
            if nbands_converge == 'all':
                nbands_converge = nbands
            elif nbands_converge != 'occupied':
                assert isinstance(nbands_converge, int)
                if nbands_converge < 0:
                    nbands_converge += nbands
            eigensolver = get_eigensolver(par.eigensolver, par.mode,
                                          par.convergence)
            eigensolver.nbands_converge = nbands_converge
            # XXX Eigensolver class doesn't define an nbands_converge property
            self.wfs.set_eigensolver(eigensolver)

        if self.density is None:
            gd = self.wfs.gd
            if par.stencils[1] != 9:
                # Construct grid descriptor for fine grids for densities
                # and potentials:
                finegd = gd.refine()
            else:
                # Special case (use only coarse grid):
                finegd = gd

            self.density = Density(gd, finegd, nspins,
                                   par.charge + setups.core_charge, collinear)

        self.density.initialize(setups, par.stencils[1], self.timer,
                                magmom_av, par.hund)
        self.density.set_mixer(par.mixer)

        if self.hamiltonian is None:
            gd, finegd = self.density.gd, self.density.finegd
            self.hamiltonian = Hamiltonian(gd, finegd, nspins,
                                           setups, par.stencils[1], self.timer,
                                           xc, par.poissonsolver,
                                           par.external, collinear)

        xc.initialize(self.density, self.hamiltonian, self.wfs,
                      self.occupations)

        self.text()
        self.print_memory_estimate(self.txt, maxdepth=memory_estimate_depth)
        self.txt.flush()

        if dry_run:
            self.dry_run()

        self.initialized = True

    def dry_run(self):
        # Can be overridden like in gpaw.atom.atompaw
        self.print_cell_and_parameters()
        self.print_positions()
        self.txt.flush()
        raise SystemExit

    def restore_state(self):
        """After restart, calculate fine density and poisson solution.

        These are not initialized by default.
        TODO: Is this really the most efficient way?
        """
        spos_ac = self.atoms.get_scaled_positions() % 1.0
        self.density.nct.set_positions(spos_ac)
        self.density.ghat.set_positions(spos_ac)
        self.density.nct_G = self.density.gd.zeros()
        self.density.nct.add(self.density.nct_G, 1.0 / self.density.nspins)
        self.density.interpolate()
        self.density.calculate_pseudo_charge(0)
        self.hamiltonian.set_positions(spos_ac, self.wfs.rank_a)
        self.hamiltonian.update(self.density)

    def attach(self, function, n=1, *args, **kwargs):
        """Register observer function.

        Call *function* every *n* iterations using *args* and
        *kwargs* as arguments."""

        try:
            slf = function.im_self
        except AttributeError:
            pass
        else:
            if slf is self:
                # function is a bound method of self.  Store the name
                # of the method and avoid circular reference:
                function = function.im_func.func_name

        self.observers.append((function, n, args, kwargs))

    def call_observers(self, iter, final=False):
        """Call all registered callback functions."""
        for function, n, args, kwargs in self.observers:
            if ((iter % n) == 0) != final:
                if isinstance(function, str):
                    function = getattr(self, function)
                function(*args, **kwargs)

    def get_reference_energy(self):
        return self.wfs.setups.Eref * Hartree

    def write(self, filename, mode='', cmr_params={}, **kwargs):
        """Write state to file.

        use mode='all' to write the wave functions.  cmr_params is a
        dictionary that allows you to specify parameters for CMR
        (Computational Materials Repository).
        """

        self.timer.start('IO')
        gpaw.io.write(self, filename, mode, cmr_params=cmr_params, **kwargs)
        self.timer.stop('IO')

    def get_myu(self, k, s):
        """Return my u corresponding to a certain kpoint and spin - or None"""
        # very slow, but we are sure that we have it
        for u in range(len(self.wfs.kpt_u)):
            if self.wfs.kpt_u[u].k == k and self.wfs.kpt_u[u].s == s:
                return u
        return None

    def get_homo_lumo(self):
        """Return HOMO and LUMO eigenvalues."""
        return self.occupations.get_homo_lumo(self.wfs) * Hartree

    def estimate_memory(self, mem):
        """Estimate memory use of this object."""
        for name, obj in [('Density', self.density),
                          ('Hamiltonian', self.hamiltonian),
                          ('Wavefunctions', self.wfs),
                          ]:
            obj.estimate_memory(mem.subnode(name))

    def print_memory_estimate(self, txt=None, maxdepth=-1):
        """Print estimated memory usage for PAW object and components.

        maxdepth is the maximum nesting level of displayed components.

        The PAW object must be initialize()'d, but needs not have large
        arrays allocated."""
        # NOTE.  This should work with --dry-run=N
        #
        # However, the initial overhead estimate is wrong if this method
        # is called within a real mpirun/gpaw-python context.
        if txt is None:
            txt = self.txt
        txt.write('Memory estimate\n')
        txt.write('---------------\n')

        mem_init = maxrss()  # initial overhead includes part of Hamiltonian!
        txt.write('Process memory now: %.2f MiB\n' % (mem_init / 1024.0**2))

        mem = MemNode('Calculator', 0)
        try:
            self.estimate_memory(mem)
        except AttributeError, m:
            txt.write('Attribute error: %r' % m)
            txt.write('Some object probably lacks estimate_memory() method')
            txt.write('Memory breakdown may be incomplete')
        totalsize = mem.calculate_size()
        mem.write(txt, maxdepth=maxdepth)

    def converge_wave_functions(self):
        """Converge the wave-functions if not present."""

        if not self.wfs or not self.scf:
            self.initialize()
        else:
            self.wfs.initialize_wave_functions_from_restart_file()

        no_wave_functions = (self.wfs.kpt_u[0].psit_nG is None)
        converged = self.scf.check_convergence(self.density,
                                               self.wfs.eigensolver)
        if no_wave_functions or not converged:
            self.wfs.eigensolver.error = np.inf
            self.scf.converged = False

            # is the density ok ?
            error = self.density.mixer.get_charge_sloshing()
            criterion = (self.input_parameters['convergence']['density']
                         * self.wfs.nvalence)
            if error < criterion:
                self.scf.fix_density()

            self.calculate()

    def check_atoms(self):
        """Check that atoms objects are identical on all processors."""
        if not mpi.compare_atoms(self.atoms, comm=self.wfs.world):
            raise RuntimeError('Atoms objects on different processors ' +
                               'are not identical!')


def kpts2ndarray(kpts):
    """Convert kpts keyword to 2d ndarray of scaled k-points."""
    if kpts is None:
        return np.zeros((1, 3))
    if isinstance(kpts[0], int):
        return monkhorst_pack(kpts)
    return np.array(kpts)
