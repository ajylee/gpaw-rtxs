import numpy as np
import pickle
from time import time, ctime
from datetime import timedelta
from ase.structure import bulk
from ase.units import Hartree
from gpaw import GPAW, FermiDirac
from gpaw.response.gw import GW
from gpaw.xc.hybridk import HybridXC
from gpaw.xc.tools import vxc
from gpaw.mpi import serial_comm, world, rank

starttime = time()

a = 5.431
atoms = bulk('Si', 'diamond', a=a)

kpts = (2,2,2)

calc = GPAW(
            h=0.24,
            kpts=kpts,
            xc='LDA',
            txt='Si_gs.txt',
            nbands=10,
            convergence={'bands':8},
            occupations=FermiDirac(0.001)
           )

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('Si_gs.gpw','all')

nbands=8
bands=np.array([3,4])
ecut=25./Hartree

gwnkpt = calc.wfs.kd.nibzkpts
gwnband = len(bands)

file='Si_gs.gpw'
calc = GPAW(
            file,
            communicator=serial_comm,
            parallel={'domain':1},
            txt=None
           )

v_xc = vxc(calc)

alpha = 5.0
exx = HybridXC('EXX', alpha=alpha, ecut=ecut, bands=bands)
calc.get_xc_difference(exx)

e_kn = np.zeros((gwnkpt, gwnband), dtype=float)
v_kn = np.zeros((gwnkpt, gwnband), dtype=float)
e_xx = np.zeros((gwnkpt, gwnband), dtype=float)

i = 0
for k in range(gwnkpt):
    j = 0
    for n in bands:
        e_kn[i][j] = calc.get_eigenvalues(kpt=k)[n] / Hartree
        v_kn[i][j] = v_xc[0][k][n] / Hartree
        e_xx[i][j] = exx.exx_skn[0][k][n]
        j += 1
    i += 1

data = {
        'e_kn': e_kn,         # in Hartree
        'v_kn': v_kn,         # in Hartree
        'e_xx': e_xx,         # in Hartree
       }
if rank == 0:
    pickle.dump(data, open('EXX.pckl', 'w'), -1)

exxfile='EXX.pckl'

gw = GW(
        file=file,
        nbands=8,
        bands=np.array([3,4]),
        w=np.linspace(0., 30., 601),
        ecut=25.,
        eta=0.1,
        hilbert_trans=False,
        exxfile=exxfile
       )

gw.get_QP_spectrum()

QP_False = gw.QP_kn * Hartree

gw = GW(
        file=file,
        nbands=8,
        bands=np.array([3,4]),
        w=np.linspace(0., 30., 601),
        ecut=25.,
        eta=0.1,
        hilbert_trans=True,
        exxfile=exxfile
       )

gw.get_QP_spectrum()

QP_True = gw.QP_kn * Hartree

if not (np.abs(QP_False - QP_True) < 0.01).all():
    raise AssertionError("method 1 not equal to method 2")

totaltime = round(time() - starttime)
print "GW test finished in %s " %(timedelta(seconds=totaltime))
