import numpy as np

from gpaw.test import equal
from gpaw.grid_descriptor import GridDescriptor
from gpaw.spline import Spline
import gpaw.mpi as mpi
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.wavefunctions.pw import PWDescriptor, PWLFC, RealSpacePWLFC


x = 2.0
rc = 3.5
r = np.linspace(0, rc, 100)

n = 40
a = 8.0
gd = GridDescriptor((n, n, n), (a, a, a), comm=mpi.serial_comm)

kpts = np.array([(0.25, 0.25, 0.0)])
spos_ac = np.array([(0.15, 0.5, 0.95)])

for l in range(3):
    s = Spline(l, rc, 2 * x**1.5 / np.pi * np.exp(-x * r**2))

    c = LFC(gd, [[s]], dtype=complex)
    c.set_k_points(kpts)
    c.set_positions(spos_ac)
    b = gd.zeros(dtype=complex)

    c_ai = {0: np.zeros(2 * l + 1, complex)}
    c_ai[0][0] = 1.9 - 4.5j

    c.add(b, c_ai, 0)

    pd = PWDescriptor(45, gd, kpts)
    c = RealSpacePWLFC(LFC(gd, [[s]]), pd)
    c.set_k_points(kpts)
    c.set_positions(spos_ac)
    b2 = pd.zeros(1, dtype=complex)
    c_axi = {0: np.array([c_ai[0]])}
    c.add(b2, c_axi, 0)
    b3 = pd.ifft(b2[0]) * c.expikr_qR[0]
    equal(abs(b-b3).max(), 0, 0.001)

    c3 = PWLFC([[s]], pd)
    c3.set_k_points(kpts)
    c3.set_positions(spos_ac)
    b2 = pd.zeros(dtype=complex)
    c3.add(b2, c_ai, 0)
    b4 = pd.ifft(b2) * c.expikr_qR[0]
    equal(abs(b4-b3).max(), 0, 2e-7)

