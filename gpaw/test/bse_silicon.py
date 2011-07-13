import numpy as np
from ase import Atom, Atoms
from ase.structure import bulk
from ase.units import Hartree, Bohr
from gpaw import GPAW, FermiDirac
from gpaw.response.bse import BSE
from ase.dft import monkhorst_pack

GS = 1
bse = 1
check = 1

if GS:
    kpts = (4,4,4)
 
    a = 5.431 # From PRB 73,045112 (2006)
    atoms = bulk('Si', 'diamond', a=a)
    calc = GPAW(h=0.2,
                kpts=kpts,
                occupations=FermiDirac(0.001),
                nbands=8,
                convergence={'bands':'all'})

    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write('Si.gpw','all')


if bse:

    eshift = 0.8
    
    bse = BSE('Si.gpw',w=np.linspace(0,10,201),
              q=np.array([0.0001,0,0.0]),optical_limit=True,ecut=50.,
              nc=np.array([4,6]), nv=np.array([2,4]), eshift=eshift,
              nbands=8,positive_w=True,use_W=True,qsymm=True)
    
    bse.get_dielectric_function('Si_bse.dat')

if check:
    
    d = np.loadtxt('Si_bse.dat')

    Nw1 = 67
    Nw2 = 80
    if d[Nw1, 2] > d[Nw1-1, 2] and d[Nw1, 2] > d[Nw1+1, 2] \
            and d[Nw2, 2] > d[Nw2-1, 2] and d[Nw2, 2] > d[Nw2+1, 2]:
        pass
    else:
        raise ValueError('Absorption peak not correct ! ')

    if np.abs(d[Nw1, 2] - 52.7324347108) > 0.1 \
        or np.abs(d[Nw2, 2] -  61.3604077801) > 0.1:
        print d[Nw1, 2], d[Nw2, 2]
        raise ValueError('Please check spectrum strength ! ')


