import numpy as np
from ase.dft.kpoints import monkhorst_pack
from gpaw.kpt_descriptor import KPointDescriptor
k = 70
k_kc = monkhorst_pack((k, k, 1))
kd = KPointDescriptor(k_kc + (0.5 / k, 0.5 / k, 0))
assert (kd.N_c == (k, k, 1)).all()
assert abs(kd.offset_c - (0.5 / k, 0.5 / k, 0)).sum() < 1e-9


