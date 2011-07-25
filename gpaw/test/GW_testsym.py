from ase.structure import bulk
from gpaw import GPAW, FermiDirac
from ase.units import Hartree
import numpy as np
from gpaw.response.gw import GW

a = 5.431
atoms = bulk('Si', 'diamond', a=a)

kpts =(3,3,3)

calc = GPAW(
            h=0.24,
            kpts=kpts,
            xc='LDA',
            txt='Si_gs.txt',
            nbands=20,
            convergence={'bands':8},
            occupations=FermiDirac(0.001)
           )

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('Si_kpt3.gpw','all')

    

file='Si_kpt3.gpw'

gw = GW(
        file=file,
        nbands=8,
        bands=np.array([2,3]),
        kpoints=np.arange(27),
        w=np.linspace(0., 60., 601),
        ecut=100.,
        eta=0.1
       )

gw.get_QP_spectrum()

gw.Qp_kn *= Hartree
print gw.Qp_kn

checkQp_kn = np.zeros((gw.kd.nibzkpts, gw.gwnband))
nn = np.zeros(gw.kd.nibzkpts)
for k in range(gw.nkpt):
    ibzk = gw.kd.bz2ibz_k[k]
    checkQp_kn[ibzk,:] += gw.Qp_kn[k,:]
    nn[ibzk] += 1

for k in range(gw.kd.nibzkpts):
    checkQp_kn[k] /= nn[k]

for k in range(gw.nkpt):
    ibzk = gw.kd.bz2ibz_k[k]
    print np.abs(checkQp_kn[ibzk] - gw.Qp_kn[k])

# gw.Qp_kn    
#[[ 6.19524053  6.19534054]
# [ 3.99973253  3.98922797]
# [ 1.27697071  4.2831092 ]
# [ 3.9998115   3.98909071]
# [ 6.18608188  6.16522611]
# [ 1.27797568  4.28056473]
# [ 1.27693229  4.28323778]
# [ 1.27794372  4.28066576]
# [ 1.27710243  4.27761804]
# [ 4.00033611  3.98849131]
# [ 6.19055713  6.16984517]
# [ 1.27780668  4.2861514 ]
# [ 6.19045773  6.16980265]
# [ 7.11598243  7.11544747]
# [ 6.19471012  6.17383495]
# [ 1.27779523  4.28615135]
# [ 6.19474947  6.17384065]
# [ 4.0068166   3.98246908]
# [ 1.27710061  4.27767882]
# [ 1.27793113  4.28065861]
# [ 1.27692875  4.2832108 ]
# [ 1.27796959  4.28053065]
# [ 6.19020653  6.16916668]
# [ 4.00734748  3.98187612]
# [ 1.27696063  4.28310927]
# [ 4.0074272   3.98174002]
# [ 6.1909594   6.19114655]]
