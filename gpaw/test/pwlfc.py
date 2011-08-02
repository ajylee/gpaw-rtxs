import numpy as np

from gpaw.wavefunctions.pw import PWDescriptor, PWLFC, RealSpacePWLFC
from gpaw.test import equal
from gpaw.grid_descriptor import GridDescriptor
from gpaw.spline import Spline
import gpaw.mpi as mpi
from gpaw.lfc import LocalizedFunctionsCollection as LFC

x = 2.0
rc = 3.0
r = np.linspace(0, rc, 100)
s = Spline(0, rc, 2 * x**1.5 / np.pi * np.exp(-x * r**2))

n = 40
a = 8.0
gd = GridDescriptor((n, n, n), (a, a, a), comm=mpi.serial_comm)
c = LFC(gd, [[s]], dtype=complex)

kpts = np.array([(0.25, 0.25, 0.0)])
spos_ac = np.array([(0.15, 0.5, 0.95)])

c.set_k_points(kpts)
c.set_positions(spos_ac)
b = gd.zeros(dtype=complex)
c_ai = {0: np.array([1.9 - 4.5j])}
c.add(b, c_ai, 0)
b = abs(b)

pd = PWDescriptor(45, gd)
c = RealSpacePWLFC(LFC(gd, [[s]]), pd)
c.set_k_points(kpts)
c.set_positions(spos_ac)
b2 = pd.zeros(1, dtype=complex)
c.add(b2, c_ai, 0)
b3 = abs(pd.ifft(b2))
print abs(b-b3).max()

c = PWLFC([[s]], pd)
c.set_k_points(kpts)
c.set_positions(spos_ac)
b2 = pd.zeros(1, dtype=complex)
c.add(b2, c_ai, 0)
b3 = abs(pd.ifft(b2))
print abs(b-b3).max()
