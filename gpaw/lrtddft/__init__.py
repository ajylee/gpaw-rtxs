"""This module defines a linear response TDDFT-class.

"""
from math import sqrt
import sys

import numpy as np
from ase.units import Hartree

import _gpaw
import gpaw.mpi as mpi
MASTER = mpi.MASTER
from gpaw import debug
from gpaw.poisson import PoissonSolver
from gpaw.output import initialize_text_stream
from gpaw.lrtddft.excitation import Excitation, ExcitationList
from gpaw.lrtddft.kssingle import KSSingles
from gpaw.lrtddft.omega_matrix import OmegaMatrix
from gpaw.lrtddft.apmb import ApmB
##from gpaw.lrtddft.transition_density import TransitionDensity
from gpaw.utilities import packed_index
from gpaw.utilities.lapack import diagonalize
from gpaw.xc import XC
from gpaw.lrtddft.spectrum import spectrum

__all__ = ['LrTDDFT', 'photoabsorption_spectrum', 'spectrum']

class LrTDDFT(ExcitationList):
    """Linear Response TDDFT excitation class
    
    Input parameters:

    calculator:
    the calculator object after a ground state calculation
      
    nspins:
    number of spins considered in the calculation
    Note: Valid only for unpolarised ground state calculation

    eps:
    Minimal occupation difference for a transition (default 0.001)

    istart:
    First occupied state to consider
    jend:
    Last unoccupied state to consider
      
    xc:
    Exchange-Correlation approximation in the Kernel
    derivative_level:
    0: use Exc, 1: use vxc, 2: use fxc  if available

    filename:
    read from a file
    """
    def __init__(self,
                 calculator=None,
                 nspins=None,
                 eps=0.001,
                 istart=0,
                 jend=None,
                 energy_range=None,
                 xc=None,
                 derivative_level=1,
                 numscale=0.00001,
                 txt=None,
                 filename=None,
                 finegrid=2,
                 force_ApmB=False, # for tests
                 eh_comm=None # parallelization over eh-pairs
                 ):

        self.nspins = None
        self.istart = None
        self.jend = None

        if isinstance(calculator, str):
            ExcitationList.__init__(self, None, txt)
            return self.read(calculator)
        else:
            ExcitationList.__init__(self, calculator, txt)

        if filename is not None:
            return self.read(filename)

        self.filename = None
        self.calculator = None
        self.eps = None
        self.xc = None
        self.derivative_level = None
        self.numscale = numscale
        self.finegrid = finegrid
        self.force_ApmB = force_ApmB

        if eh_comm is None:
            eh_comm = mpi.serial_comm
        elif isinstance(eh_comm, (mpi.world.__class__,
                                mpi.serial_comm.__class__)):
            # Correct type already.
            pass
        else:
            # world should be a list of ranks:
            eh_comm = mpi.world.new_communicator(np.asarray(eh_comm))

        self.eh_comm = eh_comm
 
        if calculator is not None:
            calculator.converge_wave_functions()
            if calculator.density.nct_G is None:
                calculator.set_positions()
                
            self.update(calculator, nspins, eps, 
                        istart, jend, energy_range,
                        xc, derivative_level, numscale)

    def analyse(self, what=None, out=None, min=0.1):
        """Print info about the transitions.
        
        Parameters:
          1. what: I list of excitation indicees, None means all
          2. out : I where to send the output, None means sys.stdout
          3. min : I minimal contribution to list (0<min<1)
        """
        if what is None:
            what = range(len(self))
        elif isinstance(what, int):
            what = [what]

        if out is None:
            out = sys.stdout
            
        for i in what:
            print >> out, str(i) + ':', self[i].analyse(min=min)
            
    def update(self,
               calculator=None,
               nspins=None,
               eps=0.001,
               istart=0,
               jend=None,
               energy_range=None,
               xc=None,
               derivative_level=None,
               numscale=0.001):

        changed = False
        if self.calculator != calculator or \
           self.nspins != nspins or \
           self.eps != eps or \
           self.istart != istart or \
           self.jend != jend :
            changed = True

        if not changed: return

        self.calculator = calculator
        self.nspins = nspins
        self.eps = eps
        self.istart = istart
        self.jend = jend
        self.xc = xc
        self.derivative_level = derivative_level
        self.numscale = numscale
        self.kss = KSSingles(calculator=calculator,
                             nspins=nspins,
                             eps=eps,
                             istart=istart,
                             jend=jend,
                             energy_range=energy_range,
                             txt=self.txt)
        if not self.force_ApmB:
            Om = OmegaMatrix
            name = 'LrTDDFT'
            if self.xc:
                xc = XC(self.xc)
                if hasattr(xc, 'hybrid') and xc.hybrid > 0.0:
                    Om = ApmB
                    name = 'LrTDDFThyb'
        else:
            Om = ApmB
            name = 'LrTDDFThyb'
        self.Om = Om(self.calculator, self.kss,
                     self.xc, self.derivative_level, self.numscale,
                     finegrid=self.finegrid, eh_comm=self.eh_comm,
                     txt=self.txt)
        self.name = name
##        self.diagonalize()

    def diagonalize(self, istart=None, jend=None, energy_range=None):
        self.istart = istart
        self.jend = jend
        self.Om.diagonalize(istart, jend, energy_range)
        
        # remove old stuff
        while len(self): self.pop()

        for j in range(len(self.Om.kss)):
            self.append(LrTDDFTExcitation(self.Om,j))

    def get_Om(self):
        return self.Om

    def read(self, filename=None, fh=None):
        """Read myself from a file"""

        if fh is None:
            if filename.endswith('.gz'):
                try:
                    import gzip
                    f = gzip.open(filename)
                except:
                    f = open(filename, 'r')
            else:
                f = open(filename, 'r')
            self.filename = filename
        else:
            f = fh
            self.filename = None

        # get my name
        s = f.readline().replace('\n','')
        self.name = s.split()[1]

        self.xc = f.readline().replace('\n','').split()[0]
        values = f.readline().split()
        self.eps = float(values[0])
        if len(values) > 1:
            self.derivative_level = int(values[1])
            self.numscale = float(values[2])
            self.finegrid = int(values[3])
        else:
            # old writing style, use old defaults
            self.numscale = 0.001

        self.kss = KSSingles(filehandle=f)
        if self.name == 'LrTDDFT':
            self.Om = OmegaMatrix(kss=self.kss, filehandle=f,
                                  txt=self.txt)
        else:
            self.Om = ApmB(kss=self.kss, filehandle=f,
                                  txt=self.txt)
        self.Om.Kss(self.kss)

        # check if already diagonalized
        p = f.tell()
        s = f.readline()
        if s != '# Eigenvalues\n':
            # go back to previous position
            f.seek(p)
        else:
            # load the eigenvalues
            n = int(f.readline().split()[0])
            for i in range(n):
                l = f.readline().split()
                E = float(l[0])
                me = [float(l[1]), float(l[2]), float(l[3])]
                self.append(LrTDDFTExcitation(e=E, m=me))

        if fh is None:
            f.close()

        # update own variables
        self.istart = self.Om.fullkss.istart
        self.jend = self.Om.fullkss.jend


    def singlets_triplets(self):
        """Split yourself into a singlet and triplet object"""

        slr = LrTDDFT(None, self.nspins, self.eps,
                      self.istart, self.jend, self.xc, 
                      self.derivative_level, self.numscale)
        tlr = LrTDDFT(None, self.nspins, self.eps,
                      self.istart, self.jend, self.xc, 
                      self.derivative_level, self.numscale)
        slr.Om, tlr.Om = self.Om.singlets_triplets()
        for lr in [slr, tlr]:
            lr.kss = lr.Om.fullkss
        return slr, tlr

    def single_pole_approximation(self, i, j):
        """Return the excitation according to the
        single pole approximation. See e.g.:
        Grabo et al, Theochem 501 (2000) 353-367
        """
        for ij, kss in enumerate(self.kss):
            if kss.i == i and kss.j == j:
                return sqrt(self.Om.full[ij][ij]) * Hartree
                return self.Om.full[ij][ij] / kss.energy * Hartree

    def __str__(self):
        string = ExcitationList.__str__(self)
        string += '# derived from:\n'
        string += self.kss.__str__()
        return string

    def write(self, filename=None, fh=None):
        """Write current state to a file.

        'filename' is the filename. If the filename ends in .gz,
        the file is automatically saved in compressed gzip format.

        'fh' is a filehandle. This can be used to write into already
        opened files. 
        """
        if mpi.rank == mpi.MASTER:
            if fh is None:
                if filename.endswith('.gz'):
                    try:
                        import gzip
                        f = gzip.open(filename,'wb')
                    except:
                        f = open(filename, 'w')
                else:
                    f = open(filename, 'w')
            else:
                f = fh

            f.write('# ' + self.name + '\n')
            xc = self.xc
            if xc is None: xc = 'RPA'
            if self.calculator is not None:
                xc += ' ' + self.calculator.get_xc_functional()
            f.write(xc + '\n')
            f.write('%g %d %g %d' % (self.eps, int(self.derivative_level),
                                     self.numscale, int(self.finegrid)) + '\n')
            self.kss.write(fh=f)
            self.Om.write(fh=f)

            if len(self):
                f.write('# Eigenvalues\n')
                istart = self.istart
                if istart is None: 
                    istart = self.kss.istart
                jend = self.jend
                if jend is None: 
                    jend = self.kss.jend
                f.write('%d %d %d'%(len(self), istart, jend) + '\n')
                for ex in self:
                    f.write(ex.outstring())
                f.write('# Eigenvectors\n')
                for ex in self:
                    for w in ex.f:
                        f.write('%g '%w)
                    f.write('\n')

            if fh is None:
                f.close()

def d2Excdnsdnt(dup, ddn):
    """Second derivative of Exc polarised"""
    res = [[0, 0], [0, 0]]
    for ispin in range(2):
        for jspin in range(2):
            res[ispin][jspin]=np.zeros(dup.shape)
            _gpaw.d2Excdnsdnt(dup, ddn, ispin, jspin, res[ispin][jspin])
    return res

def d2Excdn2(den):
    """Second derivative of Exc unpolarised"""
    res = np.zeros(den.shape)
    _gpaw.d2Excdn2(den, res)
    return res

class LrTDDFTExcitation(Excitation):
    def __init__(self,Om=None,i=None,
                 e=None,m=None):
        # define from the diagonalized Omega matrix
        if Om is not None:
            if i is None:
                raise RuntimeError

            ev = Om.eigenvalues[i]
            if ev < 0:
                # we reached an instability, mark it with a negative value
                self.energy = -sqrt(-ev)
            else:
                self.energy = sqrt(ev)
            self.f = Om.eigenvectors[i]
            self.kss = Om.kss
            self.me = 0.
            for f,k in zip(self.f, self.kss):
                self.me += f * k.me

            return

        # define from energy and matrix element
        if e is not None:
            if m is None:
                raise RuntimeError
            self.energy = e
            self.me = m
            return

        raise RuntimeError

    def density_change(self,paw):
        """get the density change associated with this transition"""
        raise NotImplementedError

    def outstring(self):
        str = '%g ' % self.energy
        str += '  '
        for m in self.me:
            str += ' %g' % m
        str += '\n'
        return str
        
    def __str__(self):
        m2 = np.sum(self.me * self.me)
        m = sqrt(m2)
        if m > 0: 
            me = self.me/m
        else:   
            me = self.me
        str = "<LrTDDFTExcitation> om=%g[eV] |me|=%g (%.2f,%.2f,%.2f)" % \
              (self.energy * Hartree, m, me[0], me[1], me[2])
        return str

    def analyse(self,min=.1):
        """Return an analysis string of the excitation"""
        s='E=%.3f'%(self.energy * Hartree)+' eV, f=%.3g'\
           %(self.get_oscillator_strength()[0])+'\n'

        def sqr(x): return x*x
        spin = ['u','d'] 
        min2 = sqr(min)
        rest = np.sum(self.f**2)
        for f,k in zip(self.f,self.kss):
            f2 = sqr(f)
            if f2>min2:
                s += '  %d->%d ' % (k.i,k.j) + spin[k.pspin] + ' ' 
                s += '%.3g \n'%f2
                rest -= f2
        s+='  rest=%.3g'%rest
        return s
        
def photoabsorption_spectrum(excitation_list, spectrum_file=None,
                             e_min=None, e_max=None, delta_e = None,
                             folding='Gauss', width=0.1, comment=None):
    """Uniform absorption spectrum interface

    Parameters:
    ================= ===================================================
    ``exlist``        ExcitationList
    ``spectrum_file`` File name for the output file, STDOUT if not given
    ``e_min``         min. energy, set to cover all energies if not given
    ``e_max``         max. energy, set to cover all energies if not given
    ``delta_e``       energy spacing
    ``energyunit``    Energy unit, default 'eV'
    ``folding``       Gauss (default) or Lorentz
    ``width``         folding width in terms of the chosen energyunit
    ================= ===================================================
    all energies in [eV]
    """

    spectrum(exlist=excitation_list, filename=spectrum_file, 
             emin=e_min, emax=e_max,
             de=delta_e, energyunit='eV', 
             folding=folding, width=width,
             comment=comment)
