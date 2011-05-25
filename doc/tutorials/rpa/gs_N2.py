from ase import *
from ase.units import Ha, Bohr
from ase.parallel import paropen
from gpaw import *
from gpaw.xc.hybrid import HybridXC
from gpaw.response.cell import get_primitive_cell, set_Gvectors

d = 6.
max_cut = 400.

# N -------------------------------------------

N = data.molecules.molecule('N')
N.set_pbc(True)
N.set_cell((d, d, d))
N.center()
calc = GPAW(h=0.18,
            maxiter=300,
            xc='PBE',
            hund=True,
            txt='N.txt',
            convergence={'density': 1.e-6})

N.set_calculator(calc)
N.set_calculator(calc)
E1_pbe = N.get_potential_energy()
E1_hf = E1_pbe + calc.get_xc_difference(HybridXC('EXX'))

acell = N.cell / Bohr
bcell = get_primitive_cell(acell)[1]
gpts = calc.get_number_of_grid_points()
max_bands = set_Gvectors(acell, bcell, gpts, [max_cut/Ha, max_cut/Ha,max_cut/Ha])[0]

calc.set(nbands=max_bands+500,
         fixdensity=True,
         eigensolver='cg',
         convergence={'bands': -500})
N.get_potential_energy()

calc.write('N.gpw', mode='all')

# N2 ------------------------------------------

N2 = data.molecules.molecule('N2')
N2.set_pbc(True)
N2.set_cell((d, d, d))
N2.center()
calc = GPAW(h=0.18,
            maxiter=300,
            xc='PBE',
            txt='N2.txt',
            convergence={'density': 1.e-6})

N2.set_calculator(calc)
dyn = optimize.BFGS(N2)
dyn.run(fmax=0.05)
E2_pbe = N2.get_potential_energy()
E2_hf = E2_pbe + calc.get_xc_difference(HybridXC('EXX'))
f = paropen('PBE_HF.dat', 'w')
print >> f, 'PBE: ', E2_pbe - 2*E1_pbe
print >> f, 'HF: ', E2_hf - 2*E1_hf
f.close()

calc.set(nbands=max_bands+500,
         fixdensity=True,
         eigensolver='cg',
         convergence={'bands': -500})
N2.get_potential_energy()

calc.write('N2.gpw', mode='all')
