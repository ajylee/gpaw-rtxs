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

Instructions assume **bash**, installation under $HOME/opt.
Load the necessary modules::

  module load devel/python/2.7.2
  module load compiler/gnu/4.5
  module load mpi/openmpi/1.4.3-gnu-4.5
  module load numlib/lapack/3.2.2-gnu-4.4
  module load numlib/atlas/3.8.3-gnu-4.4
  module load numlib/acml/4.4.0-gnu

We use ATLAS for numpy and acml for gpaw. This might change in future.
 
Build numpy. ATLAS will be detected automaticly::

  mkdir -p ${HOME}/opt/python/lib/python2.7/site-packages
  export PYTHONPATH=${HOME}/opt/python/lib/python2.7/site-packages

  wget http://dfn.dl.sourceforge.net/sourceforge/numpy/numpy-1.6.1.tar.gz
  wget http://python-nose.googlecode.com/files/nose-0.11.0.tar.gz
  tar zxf nose-0.11.0.tar.gz
  tar zxf numpy-1.6.1.tar.gz
  cd nose-0.11.0
  python setup.py install --prefix=$HOME/opt/python | tee install.log
  cd ../numpy-1.6.1
  python setup.py build --fcompiler=gnu95  | tee build.log
  python setup.py install --prefix=$HOME/opt/python | tee install.log
  cd ..
  python -c "import numpy; numpy.test()"

The installation of gpaw requires to modify customize.py to::

  libraries = ['acml', 'gfortran']
  library_dirs = ['/opt/bwgrid/numlib/acml/4.4.0-gnu/gfortran64/lib']


and build GPAW (``python setup.py build_ext | tee build_ext.log``)

A gpaw script :file:`test.py` can be submitted to run on 8 cpus like this::

  > gpaw-runscript test.py 8
  using pbs_bwg
  run.pbs_bwg written
  > qsub run.pbs_bwg

