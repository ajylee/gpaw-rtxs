.. _SUNCAT:

======
SUNCAT
======

Information about the cluster can be found at
`<https://confluence.slac.stanford.edu/display/SUNCAT/Computing>`_.

The installation of gpaw requires to modify customize.py to
:svn:`~doc/install/Linux/SUNCAT/customize.py`.

Note that this customize.py works only with MKL version 10.3.
Earlier versions have a problem with python using "dlopen" not working
with MKL circular dependencies.

.. literalinclude:: customize.py

The environment settings (valid at SUNCAT) to be able to link and run:

.. literalinclude:: setupenv
