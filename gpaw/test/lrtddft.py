import os
from ase import Atom, Atoms
from gpaw import GPAW
from gpaw.test import equal
from gpaw.lrtddft import LrTDDFT
from gpaw.mpi import world 

txt='-'
txt='/dev/null'
io_only=False
#io_only=True
load=False
#load=True
if not io_only:
    R=0.7 # approx. experimental bond length
    a = 3.0
    c = 4.0
    H2 = Atoms([Atom('H', (a / 2, a / 2, (c - R) / 2)),
                Atom('H', (a / 2, a / 2, (c + R) / 2))],
               cell=(a, a, c))
    calc = GPAW(xc='PBE', nbands=3, spinpol=False, txt=txt)
    H2.set_calculator(calc)
    H2.get_potential_energy()

    xc='LDA'

    # without spin
    lr = LrTDDFT(calc, xc=xc)
    lr.diagonalize()
    t1 = lr[0]

    # course grids
    for finegrid in [1,0]:
        lr = LrTDDFT(calc, xc=xc, finegrid=finegrid)
        lr.diagonalize()
        t3 = lr[0]
        print 'finegrid, t1, t3=', finegrid, t1 ,t3
        equal(t1.get_energy(), t3.get_energy(), 5.e-4)

    # with spin
    
    lr_vspin = LrTDDFT(calc, xc=xc, nspins=2)
    singlet, triplet = lr_vspin.singlets_triplets()
    lr_vspin.diagonalize()
    # the triplet is lower, so that the second is the first singlet
    # excited state
    t2 = lr_vspin[1]

    print 'with virtual/wo spin t2, t1=', t2.get_energy(), t1 .get_energy()
    equal(t1.get_energy(), t2.get_energy(), 5.e-7)

    if not load:
        c_spin = GPAW(xc='PBE', nbands=2, 
                      spinpol=True, parallel={'domain': world.size},
                      txt=txt)
        H2.set_calculator(c_spin)
        c_spin.calculate(H2)
##        c_spin.write('H2spin.gpw', 'all')
    else:
        c_spin = GPAW('H2spin.gpw', txt=txt)
    lr_spin = LrTDDFT(c_spin, xc=xc)
    lr_spin.diagonalize()
    for i in range(2):
        print 'i, real, virtual spin: ', i, lr_vspin[i], lr_spin[i]
        equal(lr_vspin[i].get_energy(), lr_spin[i].get_energy(), 5.e-6)

    # singlet/triplet separation
    precision = 1.e-5
    singlet.diagonalize()
    equal(singlet[0].get_energy(), lr_spin[1].get_energy(), precision)
    equal(singlet[0].get_oscillator_strength()[0],
          lr_spin[1].get_oscillator_strength()[0], precision)
    triplet.diagonalize()
    equal(triplet[0].get_oscillator_strength()[0], 0)
    equal(triplet[0].get_energy(), lr_spin[0].get_energy(), precision)
    equal(triplet[0].get_oscillator_strength()[0], 0)

# io
fname = 'lr.dat.gz'
if not io_only:
    lr.write(fname)
    world.barrier()
lr = LrTDDFT(fname)
lr.diagonalize()
t4 = lr[0]

if not io_only:
    equal(t3.get_energy(), t4.get_energy(), 1e-6)

e4 = t4.get_energy()
equal(e4, 0.675814, 1e-4)
