import os
import sys
import tempfile
import optparse
import traceback

import numpy as np
from ase.structure import bulk, estimate_lattice_constant
from ase.atoms import Atoms, string2symbols
from ase.data.molecules import molecule
from ase.visualize import view
from ase.io import read, write
from ase.constraints import FixAtoms

from gpaw.utilities import devnull
from gpaw.utilities.bulk2 import EMTRunner, GPAWRunner
from gpaw.parameters import InputParameters
from gpaw.mpi import world


defaults = InputParameters()


def build_parser():
    description = ('Run GPAW calculation for simple atoms, molecules or '
                   'bulk systems.')
    epilog = 'GPAW options: --%s.  ' % ', --'.join(defaults.keys())
    parser = optparse.OptionParser(usage='%prog [options] formula or filename',
                                   version='%prog 0.1',
                                   description=description + ' ' + epilog)

    struct = optparse.OptionGroup(parser, 'Structure')
    struct.add_option('-i', '--identifier',
                      help='String identifier added to filenames.')
    struct.add_option('-x', '--crystal-structure',
                      help='Crystal structure.',
                      choices=['sc', 'fcc', 'bcc', 'diamond', 'hcp',
                               'rocksalt', 'zincblende'])
    struct.add_option('-a', '--lattice-constant', type='float',
                      help='Lattice constant in Angstrom.')
    struct.add_option('--c-over-a', type='float',
                      help='c/a ratio.')
    struct.add_option('-v', '--vacuum', type='float', default=3.0,
                      help='Amount of vacuum to add around isolated systems '
                      '(in Angstrom).')
    struct.add_option('-O', '--orthorhombic', action='store_true',
                      help='Use orthorhombic unit cell.')
    struct.add_option('-C', '--cubic', action='store_true',
                      help='Use cubic unit cell.')
    struct.add_option('-r', '--repeat',
                      help='Repeat unit cell.  Use "-r 2" or "-r 2,3,1".')
    struct.add_option('-M', '--magnetic-moment',
                      help='Magnetic moment(s).  Use "-M 1" or "-M 2.3,-2.3".')
    parser.add_option_group(struct)

    behavior = optparse.OptionGroup(parser, 'Behavior')
    behavior.add_option('--read', action='store_true',
                        help="Don't alculate anything - read from file.")
    behavior.add_option('-p', '--plot', action='store_true',
                        help='Plot results.')
    behavior.add_option('-G', '--gui', action='store_true',
                        help="Pop up ASE's GUI.")
    behavior.add_option('-w', '--write-to-file', metavar='FILENAME',
                        help='Write configuration to file.')
    behavior.add_option('-F', '--fit', action='store_true',
                        help='Find optimal volume or bondlength.')
    behavior.add_option('-R', '--relax', type='float', metavar='FMAX',
                        help='Relax internal coordinates using L-BFGS '
                        'algorithm.')
    behavior.add_option('--constrain-tags', type='str', metavar='T1,T2,...',
                        help='Constrain atoms with tags T1, T2, ...')
    behavior.add_option('--parameters',
                        help='read input parameters from this file')
    behavior.add_option('-E', '--effective-medium-theory',
                        action='store_true',
                        help='Use EMT calculator.')
    behavior.add_option('-P', '--python-session',
                        action='store_true',
                        help='Run calculation inside interactive Python ' +
                        'session.  A possible $PYTHONSTARTUP script will be ' +
                        'imported and the "calc" variable refers to the ' +
                        'calculator.')
    parser.add_option_group(behavior)

    calc_opts = optparse.OptionGroup(parser, 'Calculator')
    for key in defaults:
        calc_opts.add_option('--%s' % key, type=str,
                             help=optparse.SUPPRESS_HELP)

    calc_opts.add_option('--write-gpw-file', metavar='MODE',
                         help='Write gpw file.')
    parser.add_option_group(calc_opts)

    return parser


def run(argv=None, ignore_python_session_option=False):
    if argv is None:
        argv = sys.argv[1:]
    elif isinstance(argv, str):
        argv = argv.split()

    parser = build_parser()
    opt, args = parser.parse_args(argv)

    if len(args) != 1:
        parser.error('incorrect number of arguments')
    name = args[0]

    if not ignore_python_session_option and opt.python_session:
        file = tempfile.NamedTemporaryFile()
        file.write('import os\n')
        file.write('if "PYTHONSTARTUP" in os.environ:\n')
        file.write('    execfile(os.environ["PYTHONSTARTUP"])\n')
        file.write('from gpaw.utilities.gpawscript import run\n')
        file.write('runner = run(%r, ignore_python_session_option=True)\n' %
                   ' '.join(argv))
        file.write('calc = runner.calc\n')
        file.flush()
        os.system('python -i %s' % file.name)
        return

    if world.rank == 0:
        out = sys.stdout
    else:
        out = devnull

    a = None
    try:
        symbols = string2symbols(name)
    except ValueError:
        # name was not a chemical formula - must be a file name:
        atoms = read(name)
    else:
        if opt.crystal_structure:
            a = opt.lattice_constant
            if a is None:
                a = estimate_lattice_constant(name, opt.crystal_structure,
                                              opt.c_over_a)
                out.write('Using an estimated lattice constant of %.3f Ang\n' %
                          a)

            atoms = bulk(name, opt.crystal_structure, a, covera=opt.c_over_a,
                         orthorhombic=opt.orthorhombic, cubic=opt.cubic)
        else:
            try:
                # Molecule?
                atoms = molecule(name)
            except NotImplementedError:
                if len(symbols) == 1:
                    # Atom
                    atoms = Atoms(name)
                elif len(symbols) == 2:
                    # Dimer
                    atoms = Atoms(name, positions=[(0, 0, 0),
                                                   (opt.bond_length, 0, 0)])
                else:
                    raise ValueError('Unknown molecule: ' + name)

    if opt.magnetic_moment:
        magmom = opt.magnetic_moment.split(',')
        atoms.set_initial_magnetic_moments(np.tile(magmom,
                                                   len(atoms) // len(magmom)))

    if opt.repeat is not None:
        r = opt.repeat.split(',')
        if len(r) == 1:
            r = 3 * r
        atoms = atoms.repeat([int(c) for c in r])

    if opt.gui:
        view(atoms)
        return

    if opt.write_to_file:
        write(opt.write_to_file, atoms)
        return

    if opt.effective_medium_theory:
        Runner = EMTRunner
    else:
        Runner = GPAWRunner

    if opt.fit:
        strains = np.linspace(0.98, 1.02, 5)
    else:
        strains = None

    if opt.constrain_tags:
        tags = [int(t) for t in opt.constrain_tags.split(',')]
        constrain = FixAtoms(mask=[t in tags for t in atoms.get_tags()])
        atoms.constraints = [constrain]

    runner = Runner(name, atoms, strains, tag=opt.identifier,
                    clean=not opt.read,
                    fmax=opt.relax, out=out)

    if not opt.effective_medium_theory:
        # Import stuff that eval() may need to know:
        from gpaw.wavefunctions.pw import PW
        from gpaw.occupations import FermiDirac, MethfesselPaxton
        from gpaw.mixer import Mixer, MixerSum
        from gpaw.poisson import PoissonSolver
        from gpaw.eigensolvers import RMM_DIIS

        if opt.parameters:
            input_parameters = eval(open(opt.parameters).read())
        else:
            input_parameters = {}
        for key in defaults:
            value = getattr(opt, key)
            if value is not None:
                try:
                    input_parameters[key] = eval(value)
                except (NameError, SyntaxError):
                    input_parameters[key] = value

        runner.set_parameters(vacuum=opt.vacuum,
                              write_gpw_file=opt.write_gpw_file,
                              **input_parameters)

    runner.run()

    runner.summary(plot=opt.plot, a0=a)

    return runner


def main():
    try:
        run(sys.argv[1:])
    except KeyboardInterrupt:
        print('Killed!')
        raise SystemExit(1)
    except SystemExit:
        raise
    except Exception:
        #traceback.print_exc()
        sys.stderr.write("""
An exception occurred!  Please report the issue to
gridpaw-developer@listserv.fysik.dtu.dk - thanks!  Please also report this
if it was a user error, so that a better error message can be provided
next time.""")
        raise
