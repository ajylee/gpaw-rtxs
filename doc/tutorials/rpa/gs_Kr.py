from ase import *
from ase.structure import bulk
from ase.units import Ha, Bohr
from gpaw import *
from gpaw.response.cell import get_primitive_cell, set_Gvectors

V = 40
a0 = (4.*V)**(1/3.)
Kr = bulk('Kr', 'fcc', a=a0)

calc = GPAW(h=0.18, xc='PBE', kpts=(6, 6, 6), nbands=8,
            txt='Kr_gs.txt')
Kr.set_calculator(calc)
Kr.get_potential_energy()

acell = Kr.cell / Bohr
bcell = get_primitive_cell(acell)[1]
gpts = calc.get_number_of_grid_points()
ecut = 300    
max_bands = set_Gvectors(acell, bcell, gpts, [ecut/Ha,ecut/Ha,ecut/Ha])[0]

calc.set(nbands=max_bands+50, 
         eigensolver='cg', 
         fixdensity=True,
         convergence={'bands': -50})
Kr.get_potential_energy()

calc.write('Kr_gs.gpw', mode='all')
