import os
from ase import Atom, Atoms
from gpaw import GPAW
from gpaw.test import equal

import cmr
a = 4.05
d = a / 2 ** 0.5
bulk = Atoms([Atom('Al', (0, 0, 0)),
              Atom('Al', (0.5, 0.5, 0.5))],
             pbc=True)
bulk.set_cell((d, d, a), scale_atoms=True)
h = 0.25
calc = GPAW(h=h,
            nbands=2 * 8,
            kpts=(2, 2, 2),
            convergence={'energy': 1e-5})
bulk.set_calculator(calc)
e0 = bulk.get_potential_energy()

for ext in [".db", ".cmr"]:
    calc.write("test1"+ext)
    assert os.path.exists("test1"+ext)
    calc.write("test2"+ext, cmr_params={"value":1, "keywords":["a", "b"]})
    assert os.path.exists("test2"+ext)
    data = cmr.read("test2"+ext)
    assert data["value"] == 1
    assert len(data["db_keywords"]) == 2
    os.unlink("test1"+ext)
    os.unlink("test2"+ext)
