# this example shows how to append new calculated results to an already
# existing cmr file, illustrated for calculation of PBE energy on LDA density

import os

import cmr
# set True in order to use cmr in parallel jobs!
cmr.set_ase_parallel(enable=True)

from ase.structure import molecule
from ase.io import read, write
from ase.parallel import barrier, rank

from gpaw import GPAW, restart
from gpaw.test import equal

# define the project in order to find it in the database!
project_id = 'modify cmr file after gpw restart'

formula = 'H2'
vacuum = 2.0
xc = 'LDA'
mode = 'lcao'
h = 0.20

cmr_params = {
    'db_keywords': [project_id],
    # add project_id also as a field to support search across projects
    'project_id': project_id,
    # user's tags: xc tag will be set later for illustration purpose!
    'formula': formula,
    'vacuum': vacuum,
    'mode': mode,
    'h': h,
    }
cmrfile = formula + '.cmr'

system1 = molecule(formula)
system1.center(vacuum=vacuum)

# first calculation: LDA lcao
calc = GPAW(mode=mode, xc=xc, h=h, txt=None)
system1.set_calculator(calc)
e = system1.get_potential_energy()
calc.write(formula)

# read gpw file
system2, calc2 = restart(formula, txt=None)
# write the information 'as in' gpw file into db file
# (called *db to avoid conflict with the *cmr file below)
if 1: # not used in this example
    calc2.write(formula + '.db', cmr_params=cmr_params)
# write the information 'as in' corresponding trajectory file into cmr file
write(cmrfile, system2, cmr_params=cmr_params)

# add the xc tag to the cmrfile
assert os.path.exists(cmrfile)
data = cmr.read(cmrfile)
data.set_user_variable('xc', xc)
data.write(cmrfile)

# peform PBE calculation on LDA density
ediff = calc2.get_xc_difference('PBE')

# add new results to the cmrfile
assert os.path.exists(cmrfile)
data = cmr.read(cmrfile)
data.set_user_variable('PBE', data['ase_potential_energy'] + ediff)
data.write(cmrfile)

# analyse the results with CMR

# cmr readers work only in serial!

from cmr.ui import DirectoryReader

if rank == 0:
    reader = DirectoryReader(directory='.', ext='.cmr')
    # read all compounds in the project with lcao
    all = reader.find(name_value_list=[('mode', 'lcao')],
                      keyword_list=[project_id])
    results = all.get('formula', formula)

    print results['formula'], results['xc'], results['ase_potential_energy']

    # column_length=0 aligns data in the table (-1 : data unaligned is default)
    all.print_table(column_length=0,
                    columns=['formula', 'xc', 'h', 'ase_potential_energy', 'PBE'])

if rank == 0:
    equal(results['PBE'], e + ediff, 1e-6)

if rank == 0:
    for file in [formula + '.gpw', formula + '.db', cmrfile]:
        if os.path.exists(file): os.unlink(file)
