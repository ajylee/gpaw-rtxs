from ase.parallel import paropen
from gpaw import *
from gpaw.xc.hybridk import HybridXC

calc = GPAW('Kr_gs.gpw', txt=None, parallel={'domain': 1})
E = calc.get_potential_energy()
exx = HybridXC('EXX')
E_hf = E + calc.get_xc_difference(exx)

f = paropen('hf_energy.dat', 'a')
print >> f, E_hf
f.close()
