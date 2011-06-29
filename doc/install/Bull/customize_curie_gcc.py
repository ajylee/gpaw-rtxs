import os

scalapack = True
mpicompiler = 'mpicc'
libraries = []

# MKL flags
mkl_flags = os.environ['MKL_SCA_LIBS']
extra_link_args = [mkl_flags]

define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
define_macros += [("GPAW_ASYNC",1)]
define_macros += [("GPAW_MPI2",1)]

