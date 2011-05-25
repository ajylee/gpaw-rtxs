"""Bandstructure exercise

Calculate the band structure of Na along the Gamma-X direction.
"""
from gpaw import GPAW, FermiDirac
from ase import Atoms

a = 4.23
atoms = Atoms('Na2',
              [(0, 0, 0),
               (0.5 * a, 0.5 * a, 0.5 * a)],
              pbc=True, cell=(a, a, a))

# Make self-consistent calculation and save results
h = 0.25
calc = GPAW(h=0.25,
            kpts=(8, 8, 8),
            occupations=FermiDirac(width=0.05),
            nbands=3,
            txt='out_Na_sc.txt')
atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('Na_sc.gpw')

# Calculate band structure along Gamma-X i.e. from 0 to 0.5
# Do not use symmetry in band structure calculation, 
# otherwise one may get less kpoints than one would expect 
nkpt = 50
kpts = [(k / float(2 * nkpt), 0, 0) for k in range(nkpt)]
calc = GPAW('Na_sc.gpw', txt='out_Na_harris.txt',
            kpts=kpts, fixdensity=True, usesymm=None, nbands=7,
            eigensolver='cg', convergence={'bands': 'all'})
ef = calc.get_fermi_level()
calc.get_potential_energy()

# Write the results to a file e.g. for plotting with gnuplot
f = open('Na_bands.txt', 'w')
for k, kpt_c in enumerate(calc.get_ibz_k_points()):
    for eig in calc.get_eigenvalues(kpt=k):
        print >> f, kpt_c[0], eig - ef
