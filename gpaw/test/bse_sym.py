import numpy as np
from ase.structure import bulk
from gpaw import GPAW
from gpaw.response.df import DF
from ase.dft import monkhorst_pack
from ase.dft.kpoints import get_monkhorst_shape
from gpaw.response.bse import BSE
from gpaw.mpi import rank, size

# generate kmesh
kpts =(2,2,2)
bzk_kc = monkhorst_pack(kpts)
Nk_c = get_monkhorst_shape(bzk_kc)

shift_c = []
for Nk in Nk_c:
    if Nk % 2 == 0:
        shift_c.append(0.5 / Nk)
    else:
        shift_c.append(0.)

atoms = bulk('Si', 'diamond', a=5.431) 

kpts1 = bzk_kc # not Gamma centered
kpts2 = bzk_kc + shift_c # Gamma centered

for kpts in (kpts1, kpts2):

    calc = GPAW(h=0.20, kpts=kpts)      
    atoms.set_calculator(calc)               
    atoms.get_potential_energy()
    calc.write('Si.gpw','all')
    
    # no symmetry BSE
    eshift = 0.8
    bse = BSE('Si.gpw',w=np.linspace(0,10,201),
                  q=np.array([0.0001,0,0.0]),optical_limit=True,ecut=150.,
                  nc=np.array([4,6]), nv=np.array([2,4]), eshift=eshift,
                  nbands=8,positive_w=True,use_W=True,qsymm=False)
    bse.get_dielectric_function('bse_nosymm.dat')
    
    
    # with symmetry BSE
    eshift = 0.8
    bse = BSE('Si.gpw',w=np.linspace(0,10,201),
                  q=np.array([0.0001,0,0.0]),optical_limit=True,ecut=150.,
                  nc=np.array([4,6]), nv=np.array([2,4]), eshift=eshift,
                  nbands=8,positive_w=True,use_W=True,qsymm=True)
    bse.get_dielectric_function('bse_symm.dat')

check = 1
if check:
    d1 = np.loadtxt('bse_nosymm.dat')
    d2 = np.loadtxt('bse_symm.dat')
    assert np.abs(np.abs(d1[:,2] - d2[:,2]).max() - 0.015336443) < 1e-4
    assert np.abs(np.abs(d1[:,2] - d2[:,2]).sum() - 0.217433477941) < 1e-2 


