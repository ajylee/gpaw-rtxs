.. _download:

========
Download
========

.. note::

   To determine which way of installation suits you best
   please read carefully :ref:`installationguide` first!

.. _latest_stable_release:

Latest stable release
=====================

The latest stable release can be obtained from ``svn`` or as a ``tarball``.

When using svn please set the following variable:

- bash::

   export GPAW_TAGS=https://svn.fysik.dtu.dk/projects/gpaw/tags/

- csh/tcsh::

   setenv GPAW_TAGS https://svn.fysik.dtu.dk/projects/gpaw/tags/

========= =========== ============================================== =======================
Release   Date        Retrieve as svn checkout                       Retrieve as tarball    
========= =========== ============================================== =======================
   0.8.0_ May 25 2011 ``svn co -r 8092 $GPAW_TAGS/0.8.0 gpaw-0.8.0`` gpaw-0.8.0.8092.tar.gz_
   0.7.2_ Aug 11 2010 ``svn co -r 6974 $GPAW_TAGS/0.7.2 gpaw-0.7.2`` gpaw-0.7.2.6974.tar.gz_
   0.7_   Apr 23 2010 ``svn co -r 6383 $GPAW_TAGS/0.7 gpaw-0.7``     gpaw-0.7.6383.tar.gz_  
   0.6_   Oct  9 2009 ``svn co -r 5147 $GPAW_TAGS/0.6 gpaw-0.6``     gpaw-0.6.5147.tar.gz_  
   0.5_   Apr  1 2009 ``svn co -r 3667 $GPAW_TAGS/0.5 gpaw-0.5``     gpaw-0.5.3667.tar.gz_  
   0.4_   Nov 16 2008 ``svn co -r 2734 $GPAW_TAGS/0.4 gpaw-0.4``     gpaw-0.4.2734.tar.gz_  
========= =========== ============================================== =======================

.. _0.8.0:
    https://trac.fysik.dtu.dk/projects/gpaw/browser/tags/0.8.0

.. _gpaw-0.8.0.8092.tar.gz:
    https://wiki.fysik.dtu.dk/gpaw-files/gpaw-0.8.0.8092.tar.gz

.. _0.7.2:
    https://trac.fysik.dtu.dk/projects/gpaw/browser/tags/0.7.2

.. _gpaw-0.7.2.6974.tar.gz:
    https://wiki.fysik.dtu.dk/gpaw-files/gpaw-0.7.2.6974.tar.gz

.. _0.7:
    https://trac.fysik.dtu.dk/projects/gpaw/browser/tags/0.7

.. _gpaw-0.7.6383.tar.gz:
    https://wiki.fysik.dtu.dk/gpaw-files/gpaw-0.7.6383.tar.gz

.. _0.6:
    https://trac.fysik.dtu.dk/projects/gpaw/browser/tags/0.6

.. _gpaw-0.6.5147.tar.gz:
    https://wiki.fysik.dtu.dk/gpaw-files/gpaw-0.6.5147.tar.gz

.. _0.5:
    https://trac.fysik.dtu.dk/projects/gpaw/browser/tags/0.5

.. _gpaw-0.5.3667.tar.gz:
    https://wiki.fysik.dtu.dk/gpaw-files/gpaw-0.5.3667.tar.gz

.. _0.4:
    https://trac.fysik.dtu.dk/projects/gpaw/browser/tags/0.4

.. _gpaw-0.4.2734.tar.gz:
    https://wiki.fysik.dtu.dk/gpaw-files/gpaw-0.4.2734.tar.gz

After getting the code :ref:`create_links`.

.. _latest_development_release:

Latest development release
==========================

The latest revision can be obtained from svn::

  $ svn checkout https://svn.fysik.dtu.dk/projects/gpaw/trunk gpaw

or from the daily snapshot: `<gpaw-snapshot.tar.gz>`_.

After getting the code :ref:`create_links`.

.. note::

   The recommended checkout path is :envvar:`$HOME`.

See :ref:`faq` in case of problems.

.. _create_links:

Create links
============

.. note::

   GPAW requires ASE.
   :ase:`Download and install ASE <download.html>`.

It is convenient to maintain several version of GPAW
with the help of links.
After downloading create the link to the requested version, e.g.:

- if retrieved from ``svn``::

   $ cd $HOME
   $ ln -s gpaw-0.8.0 gpaw

- if retrieved as ``tarball``::

   $ cd $HOME
   $ tar xtzf gpaw-0.8.0.8092.tar.gz
   $ ln -s gpaw-0.8.0.8092 gpaw

  .. note::

     The recommended installation path is :envvar:`$HOME`.

When you have the code, go back to the :ref:`installationguide`.
