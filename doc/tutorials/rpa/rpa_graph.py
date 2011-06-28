from ase import *
from ase.parallel import paropen
from gpaw import *
from gpaw.mpi import serial_comm
from gpaw.xc.rpa_correlation_energy import RPACorrelation

ds = [2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 4.0, 5.0, 6.0, 12.0]
ecut = 250

for d in ds:
    calc = GPAW('gs_%s.gpw' % d, communicator=serial_comm, txt=None)
    rpa = RPACorrelation(calc, txt='rpa_%s.txt' % d)
    E = rpa.get_rpa_correlation_energy(ecut=ecut,
                                       directions=[[0, 2/3.], [2, 1/3.]],
                                       kcommsize=64,
                                       restart='restart_%s.txt' % d)
    f = paropen('rpa_graph.dat', 'a')
    print >> f, d, E
    f.close()
