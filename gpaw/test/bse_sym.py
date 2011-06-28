import numpy as np
from ase.structure import bulk
from gpaw import GPAW
from gpaw.response.df import DF
from ase.dft import monkhorst_pack
from ase.dft.kpoints import get_monkhorst_shape
from gpaw.response.bse import BSE


# kpoint must be Gamma centered if you use symmetry for bse
kpts =(2,2,2)
bzk_kc = monkhorst_pack(kpts)

Nk_c = get_monkhorst_shape(bzk_kc)

shift_c = []
for Nk in Nk_c:
    if Nk % 2 == 0:
        shift_c.append(0.5 / Nk)
    else:
        shift_c.append(0.)

bzk_kc += shift_c
        

# no symmetry GS
atoms = bulk('Si', 'diamond', a=5.431) 

calc = GPAW(h=0.20, kpts=bzk_kc, usesymm=None)      
atoms.set_calculator(calc)               
atoms.get_potential_energy()
calc.write('Si_nosym.gpw','all')

# no symmetry BSE
eshift = 0.8
bse = BSE('Si_nosym.gpw',w=np.linspace(0,10,201),
              q=np.array([0.0001,0,0.0]),optical_limit=True,ecut=150.,
              nc=np.array([4,6]), nv=np.array([2,4]), eshift=eshift,
              nbands=8,positive_w=True,use_W=True)
bse.get_dielectric_function('bse_nosymm.dat')


# with symmetry GS
calc = GPAW(h=0.20, kpts=bzk_kc)      
atoms.set_calculator(calc)               
atoms.get_potential_energy()          
calc.write('Si_sym.gpw','all')

# with symmetry BSE
eshift = 0.8
bse = BSE('Si_sym.gpw',w=np.linspace(0,10,201),
              q=np.array([0.0001,0,0.0]),optical_limit=True,ecut=150.,
              nc=np.array([4,6]), nv=np.array([2,4]), eshift=eshift,
              nbands=8,positive_w=True,use_W=True)
bse.get_dielectric_function('bse_symm.dat')

from pylab import *
d1 = np.loadtxt('bse_nosymm.dat')
d2 = np.loadtxt('bse_symm.dat')
print np.abs(d1[:,2] - d2[:,2]).max()
print np.abs(d1[:,2] - d2[:,2]).sum()
#29.080429065
#277.852820767

plot(d1[:,0],d1[:,2],'-k')
plot(d2[:,0],d2[:,2],'-r')


#show()
