from ase import *
from ase.units import Ha
from ase.dft import monkhorst_pack
from ase.parallel import paropen
from ase.lattice.hexagonal import Graphite
from gpaw import *
from gpaw.response.cell import get_primitive_cell, set_Gvectors

kpts = monkhorst_pack((12,12,4))
kpts += np.array([1/24., 1/24., 1/8.])

a = 2.46 # Lattice parameter of graphene
d = 3.34
max_cut = 250

ds = [2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 4.0, 5.0, 6.0, 12.0]

for d in ds: 
    calc = GPAW(h=0.18,
                xc='PBE',
                #kpts=(12,12,4),
                kpts=kpts,
                maxiter=300,
                mixer=Mixer(beta=0.1, nmaxold=5, weight=50.0),
                txt='gs_%s.txt' % d,
                parallel={'domain': 1},
                idiotproof=False)

    bulk = Graphite(symbol='C', latticeconstant={'a':a, 'c':2*d}, pbc=True)
    bulk.set_calculator(calc)
    bulk.get_potential_energy()
    acell = bulk.cell / Bohr
    bcell = get_primitive_cell(acell)[1]
    gpts = calc.get_number_of_grid_points()
    max_bands = set_Gvectors(acell, bcell, gpts,
                             [max_cut/Ha, max_cut/Ha,max_cut/Ha])[0]

    calc.set(nbands=max_bands+200,
             fixdensity=True,
             eigensolver='cg',
             convergence={'bands': -200})
    bulk.get_potential_energy()

    calc.write('gs_%s.gpw' % d, mode='all')
