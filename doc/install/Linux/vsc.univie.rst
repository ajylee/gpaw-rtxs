.. _vsc.univie:

==========
vsc.univie
==========

The vsc.univie machine is a cluster of dual socket, hexa-core Intel Xeon X5650
2.67 GHz processors with 2 GB of memory per core.

Instructions assume **bash**, installation under `${HOME}/opt`.

Setup the root directory::

  mkdir -p ${HOME}/opt
  cd ${HOME}/opt

Set the versions::

  export nose=0.11.3
  # Warning: version 1.6.0 seems inconsistent about C-, Fortran-contiguous
  export numpy=1.5.1
  export scipy=0.9.0

  export acml=4.0.1

  export ase=3.5.1.2175
  export gpaw=0.8.0.8092
  export setups=0.8.7929
  
and create sh startup script::

  cat <<EOF > ${HOME}/opt/campos.sh
  #!/bin/sh
  #
  export GPAW_PLATFORM=`python -c "from distutils import util, sysconfig; print util.get_platform()+'-'+sysconfig.get_python_version()"`
  #
  export LD_LIBRARY_PATH=\${HOME}/opt/acml-${acml}/gfortran64/lib:\${LD_LIBRARY_PATH}
  #
  export PYTHONPATH=\${HOME}/opt/nose-${nose}-1/usr/lib/python2.4/site-packages:\${PYTHONPATH}
  export PATH=\${HOME}/opt/nose-${nose}-1/usr/bin:\${PATH}
  #
  export PYTHONPATH=\${HOME}/opt/numpy-${numpy}-1/usr/lib64/python2.4/site-packages:\${PYTHONPATH}
  export PATH=numpy-\${HOME}/opt/${numpy}-1/usr/bin:\${PATH}
  #
  export PYTHONPATH=\${HOME}/opt/scipy-${scipy}-1/usr/lib64/python2.4/site-packages:\${PYTHONPATH}
  export PATH=\${HOME}/opt/scipy-${scipy}-1/usr/bin:\${PATH}
  #
  export PYTHONPATH=\${HOME}/opt/python-ase-${ase}:\${PYTHONPATH}
  export PATH=\${HOME}/opt/python-ase-${ase}/tools:\${PATH}
  #
  export GPAW_SETUP_PATH=\${HOME}/opt/gpaw-setups-${setups}
  #
  export GPAW_HOME=\${HOME}/opt/gpaw-${gpaw}
  export PYTHONPATH=\${GPAW_HOME}:\${PYTHONPATH}
  export PYTHONPATH=\${GPAW_HOME}/build/lib.${GPAW_PLATFORM}:\${PYTHONPATH}
  export PATH=\${GPAW_HOME}/build/bin.${GPAW_PLATFORM}:\${PATH}
  export PATH=\${GPAW_HOME}/tools:\${PATH}
  EOF

Download and install acml::

  acml-${acml} # download
  cd acml-${acml}
  tar zxf acml-*.tgz && tar zxf contents-acml-*.tgz

Build nose/numpy/scipy::

  wget --no-check-certificate https://downloads.sourceforge.net/project/numpy/NumPy/${numpy}/numpy-${numpy}.tar.gz
  wget --no-check-certificate https://downloads.sourceforge.net/project/scipy/scipy/${scipy}/scipy-${scipy}.tar.gz
  wget http://python-nose.googlecode.com/files/nose-${nose}.tar.gz
  tar zxf nose-${nose}.tar.gz
  tar zxf numpy-${numpy}.tar.gz
  tar zxf scipy-${scipy}.tar.gz
  cd nose-${nose}
  python setup.py install --root=${HOME}/opt/nose-${nose}-1 2>&1 | tee install.log

use the following ``site.cfg`` to build numpy::

  cat <<EOF > ${HOME}/opt/numpy-${numpy}/site.cfg
  [DEFAULT]
  library_dirs = /usr/lib64:${HOME}/opt/acml-${acml}/gfortran64/lib
  include_dirs = ${HOME}/opt/acml-${acml}/gfortran64/lib/../include:/usr/include/suitesparse
  [blas]
  libraries = acml
  library_dirs = ${HOME}/opt/acml-${acml}/gfortran64/lib
  [lapack]
  libraries = acml, gfortran
  library_dirs = ${HOME}/opt/acml-${acml}/gfortran64/lib
  EOF

continue with::

  cd ../numpy-${numpy}
  # force numpy to use internal blas + acml, note the double quotes! Don't use acml for dotblas,
  # see gpaw/test/numpy_core_multiarray_dot.py
  sed -i "s/_lib_names = \['lapack'\]/_lib_names = ['acml']/g"  numpy/distutils/system_info.py
  sed -i "s/_lib_names = \['blas'\]/_lib_names = ['acml']/g"  numpy/distutils/system_info.py
  # avoid "Both g77 and gfortran runtimes linked in lapack_lite !" setting --fcompiler=gnu95
  python setup.py build --fcompiler=gnu95 2>&1 | tee build.log
  python setup.py install --root=${HOME}/opt/numpy-${numpy}-1 2>&1 | tee install.log
  cd ..
  source ${HOME}/opt/campos.sh
  python -c "import numpy; numpy.test()"

  cd scipy-${scipy}
  python setup.py config_fc --fcompiler=gfortran install --root=${HOME}/opt/scipy-${scipy}-1 2>&1 | tee install.log
  cd ..
  python -c "import scipy; scipy.test()"

Make sure that you have the right mpicc::

  which mpicc
  /usr/mpi/qlogic/bin/mpicc

Install ASE/GPAW::

  wget https://wiki.fysik.dtu.dk/ase-files/python-ase-${ase}.tar.gz
  wget https://wiki.fysik.dtu.dk/gpaw-files/gpaw-${gpaw}.tar.gz
  wget http://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-${setups}.tar.gz
  tar zxf python-ase-${ase}.tar.gz
  tar zxf gpaw-${gpaw}.tar.gz
  tar zxf gpaw-setups-${setups}.tar.gz
  mkdir testase && cd testase && testase.py 2>&1 | tee ../testase.log
  wget https://svn.fysik.dtu.dk/projects/gpaw/trunk/doc/install/Linux/customize_vsc_univie.py
  cd ../gpaw-${gpaw}
  python setup.py --remove-default-flags --customize=../customize_vsc_univie.py build_ext 2>&1 | tee build_ext.log

The :file:`customize_vsc_univie.py` looks like:

.. literalinclude:: customize_vsc_univie.py

GPAW tests :file:`gpaw-test` can be submitted like this::

  qsub run.sh

where :file:`run.sh` looks like this::

  #!/bin/sh

  #$ -pe mpich 8
  #$ -V
  #$ -M my.name@example.at
  #$ -m be
  #$ -l h_rt=00:50:00

  if [ -z "${PYTHONPATH}" ]
  then
      export PYTHONPATH=""
  fi

  source ${HOME}/opt/campos.sh

  export OMP_NUM_THREADS=1

  mpirun -m $TMPDIR/machines -np $NSLOTS gpaw-python `which gpaw-test`

Please make sure that your jobs do not run multi-threaded, e.g. for a
job running on ``node02`` do from a login node::

  ssh node02 ps -fL

you should see **1** in the **NLWP** column. Numbers higher then **1**
mean multi-threaded job.

It's convenient to customize as described on the :ref:`parallel_runs` page.
