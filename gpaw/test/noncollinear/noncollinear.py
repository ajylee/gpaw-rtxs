from ase import Atoms
from gpaw import GPAW, FermiDirac
from gpaw.xc.noncollinear import NonCollinearLDAKernel, \
     NonCollinearFunctional, NonCollinearLCAOEigensolver, NonCollinearMixer
from gpaw.xc import XC

h = Atoms('H', magmoms=[(1, 0, 0)])
h.center(vacuum=2)
xc = XC(NonCollinearLDAKernel())
#xc = NonCollinearFunctional(XC('PBE'))
c = GPAW(txt='nc.txt',
         mode='lcao',
         basis='dz(dzp)',
         #setups='ncpp',
         h=0.25,
         occupations=FermiDirac(0.0),
         xc=xc,
         mixer=NonCollinearMixer(),
         eigensolver=NonCollinearLCAOEigensolver())
c.set(nbands=1)
h.calc = c
h.get_potential_energy()
