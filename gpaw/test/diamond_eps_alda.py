import numpy as np
import sys
import time

from ase.units import Bohr
from ase.structure import bulk
from gpaw import GPAW, FermiDirac
from gpaw.atom.basis import BasisMaker
from gpaw.response.df import DF
from gpaw.mpi import serial_comm, rank, size
from gpaw.utilities import devnull


if rank != 0:
  sys.stdout = devnull 

# GS Calculation One
a = 6.75 * Bohr
atoms = bulk('C', 'diamond', a=a)

nbands = 8

calc = GPAW(h=0.2,
            kpts=(2,2,2),
            nbands = nbands+5,
            eigensolver='cg',
            occupations=FermiDirac(0.001),
            convergence={'bands':nbands})

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('C2.gpw','all')

# Macroscopic dielectric constant calculation
q = np.array([0.0, 0.00001, 0.])

df = DF(calc='C2.gpw', q=q, w=(0.,), eta=0.001, nbands=nbands,
        ecut=50, hilbert_trans=False, optical_limit=True)
eM1, eM2 = df.get_macroscopic_dielectric_constant()

if np.abs(eM2[1]-7.914302)>1e-3:
    raise ValueError("Incorrect value from Diamond dielectric constant with ALDA Kernel %.4f" % (eM2))

# RPA:
# With kpts=(12,12,12) and bands=64, ecut=250eV, this script gives 5.56
# Value from PRB 73, 045112 with kpts=(12,12,12) and bands=64: 5.55
# ALDA:
# With kpts=(12,12,12) and bands=64, ecut=250eV, this script gives 5.82 
# Value from PRB 73, 045112 with kpts=(12,12,12) and bands=64: 5.82
