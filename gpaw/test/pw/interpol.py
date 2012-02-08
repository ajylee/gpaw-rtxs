import numpy as np
from gpaw.test import equal
from gpaw.grid_descriptor import GridDescriptor
from gpaw.transformers import Transformer
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.mpi import world

def test(gd1, gd2, pd1, pd2, R1, R2):
    a1 = gd1.zeros(dtype=pd1.dtype)
    a1[R1] = 1
    x = pd1.interpolate(a1, pd2)[0][R2]
    
    a2 = gd2.zeros(dtype=pd2.dtype)
    a2[R2] = 1
    y = pd2.restrict(a2, pd1)[0][R1] * 8

    equal(x, y, 1e-9)

if world.size == 1:
    n = 8
    gd2 = GridDescriptor((n, n, n))
    gd1 = gd2.coarsen()
    pd1 = PWDescriptor(10, gd1, complex)
    pd2 = PWDescriptor(10, gd2, complex)
    test(gd1, gd2, pd1, pd2, (0,0,0), (0,3,1))
