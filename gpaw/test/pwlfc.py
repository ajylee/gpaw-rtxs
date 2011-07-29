import numpy as np
from gpaw.test import equal
from gpaw.grid_descriptor import GridDescriptor
from gpaw.spline import Spline
import gpaw.mpi as mpi
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.wavefunctions.pw import PWDescriptor, PWLFC, RealSpacePWLFC

x = 5.0
rc = 2.0
r = np.linspace(0, rc, 100)
s = Spline(0, rc, 2 * x**1.5 / np.pi * np.exp(-x * r**2))
n = 40
a = 8.0
gd = GridDescriptor((n, n, n), (a, a, a), comm=mpi.serial_comm)
c = LFC(gd, [[s]])
c.set_positions([(0.5, 0.5, 0.5)])
b = gd.zeros()
c.add(b, {0: np.ones(1)})
x = gd.integrate(b)
print x

pd = PWDescriptor(25, gd)
c = RealSpacePWLFC(LFC(gd, [[s]]), pd)
c.set_k_points(np.zeros((1, 3)))
c.set_positions([(0.5, 0.5, 0.5)])
b = pd.zeros(1, dtype=complex)
c.add(b, {0: np.ones((1, 1), complex)}, 0)
x = gd.integrate(pd.ifft(b))
print x

c = PWLFC([[s]], pd)
c.set_k_points(np.zeros((1, 3)))
c.set_positions([(0.5, 0.5, 0.5)])
b = pd.zeros(1, dtype=complex)
c.add(b, {0: np.ones((1, 1), complex)}, 0)
x = gd.integrate(pd.ifft(b))
print x
