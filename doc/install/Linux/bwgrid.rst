======
bwgrid
======

The `BWgrid <http://www.bw-grid.de/>`__
is an grid of machines located in Baden-Württemberg, Germany.
The installation in Freiburg is a cluster containing 139 dual socket,
quad-core Intel Xenon E5440 CPUs, 2.83GHz processors with 2 GB of memory
per core, 16 dual socket, quad-core Intel Xenon X5550 CPUs, 2.67GHz processors
with 3 GB of memory per core and eight dual socket, six-core Intel Xenon
X5650 CPUs, 2.66GHz processors with 2 GB of memory per core. For more
information visit `<http://www.bfg.uni-freiburg.de/doc/bwgrid>`_.

Building GPAW with Intel compiler
=================================

Use the compiler wrapper file :svn:`~doc/install/Linux/icc.py`

.. literalinclude:: icc.py

Instructions assume **bash**, installation under $HOME/opt.
Load the necessary modules::

  module load devel/python/2.7.2
  module load compiler/intel/12.0
  module load mpi/impi/4.0.2-intel-12.0
  module load numlib/mkl/10.3.5
  module load numlib/python_numpy/1.6.1-python-2.7.2

The installation of gpaw requires to modify customize.py to
:svn:`~doc/install/Linux/customize_bwgrid_icc.py`

.. literalinclude:: customize_bwgrid_icc.py

and build GPAW (``python setup.py build_ext | tee build_ext.log``)

A gpaw script :file:`test.py` can be submitted to run on 8 cpus like this::

  > gpaw-runscript test.py 8
  using bwg
  run.bwg written
  > qsub run.bwg

