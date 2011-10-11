import optparse

from ase.tasks.calcwrapper import ElectronicStructureCalculatorWrapper

from gpaw.utilities import h2gpts


class GPAWWrapper(ElectronicStructureCalculatorWrapper):
    def __init__(self, show_text_output=False, write_gpw_file=None,
                 **kwargs):
        self.show_text_output = show_text_output
        self.write_gpw_file = write_gpw_file

        ElectronicStructureCalculatorWrapper.__init__(self, 'GPAW', **kwargs)

    def __call__(self, name, atoms):
        kpts = self.calculate_kpts(atoms)

        if (not atoms.pbc.any() and len(atoms) == 1 and
            atoms.get_initial_magnetic_moments().any()):
            self.kwargs['hund'] = True

        if atoms.pbc.any() and 'gpts' not in self.kwargs:
            # Use fixed number of gpts:
            h = self.kwargs.get('h', 0.2)
            gpts = h2gpts(h, atoms.cell)
            self.kwargs['h'] = None
            self.kwargs['gpts'] = gpts

        if self.show_text_output:
            txt = '-'
        else:
            txt = name + '.txt'

        if self.write_gpw_file is not None:
            from gpaw.hooks import hooks
            hooks['converged'] = (
                lambda calc, name=name + '.gpw', mode=self.write_gpw_file:
                    calc.write(name, mode))

        from gpaw import GPAW
        return GPAW(txt=txt, kpts=kpts, xc=self.xc, **self.kwargs)
        
    def add_options(self, parser):
        ElectronicStructureCalculatorWrapper.add_options(self, parser)
        
        calc = optparse.OptionGroup(parser, 'GPAW')
        calc.add_option('-S', '--show-text-output', action='store_true',
                        help='Send text output from calculation to ' +
                        'standard out.')
        calc.add_option('-W', '--write-gpw-file', metavar='MODE',
                        help='Write gpw file.')
        parser.add_option_group(calc)

    def parse(self, opts, args):
        ElectronicStructureCalculatorWrapper.parse(self, opts, args)

        self.show_text_output = opts.show_text_output
        self.write_gpw_file = opts.write_gpw_file
