from ase import *
from ase.lattice.hexagonal import Graphite
from ase.parallel import paropen
from gpaw import *
from gpaw.xc.hybridk import HybridXC

a = 2.46 # Lattice parameter of graphene

ds = [2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 4.0, 5.0, 6.0, 12.0]

calc = GPAW(h=0.18,
            xc='PBE',
            kpts=(20, 20, 6),
            txt='gs.txt')

for d in ds:
    bulk = Graphite(symbol='C', latticeconstant={'a':a, 'c':2*d},
                    pbc=True)
    bulk.set_calculator(calc)
    E = bulk.get_potential_energy()
    exx = HybridXC('EXX', alpha=5.0)
    E_hf = E + calc.get_xc_difference(exx)

    f = paropen('hf_graph.dat', 'a')
    print >> f, d, E_hf
    f.close()
